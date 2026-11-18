"""Third modality clock + pair-label masked-diffusion loss + sampler hook on top of ProductSpaceFlowMatcher.

The pair-label modality is NOT a ``data_mode`` (``FLOW_MATCHER_FACTORY`` only knows ``bb_ca``/``local_latents``,
psfm:19 and :114); it rides in the ``t`` dict under the extra key ``'ring'`` and in ``batch`` as

  ring_state_gt  [b,n,n] long   clean labels: 0..21 RING states, 22 NO_INTERACTION, anything >= 23 = NOT OBSERVED
                                (the code alone encodes observation; a ``ring_obs`` key, if present, is diagnostic only:
                                a registered-but-absent key is collate-filled with False and must not silence the loss)
  ring_state     [b,n,n] long   NETWORK INPUT written here: revealed label | MASK (23) | PAD (24: padding and diagonal)

Input contract (no coverage leak): the set of revealed pairs is drawn over ALL valid pairs independent of coverage; an
UNOBSERVED pair that is revealed is shown as NO_INTERACTION (22) and never enters the loss, so neither the reveal
pattern nor the values carry information about which pairs RING covered beyond a (mostly true) prior of 22.

Fixes after the red-team of the spike (results/N002-*/redteam_claude.md): CUDA-safe fills; unobserved pairs never
trained towards NO_INTERACTION; missing ``ring_state_gt`` / ``t['ring']`` handled (val, inference); MDLM weight with
the per-sample VALID-pair normaliser (no heavy tail); class-weighted CE; third clock tied to the structure clock by a
lagged tube (structure leads), so training states match the sampler; diagonal handled identically in both;
``partial_simulation`` re-armed; last step reveals by argmax.
"""
from __future__ import annotations

import contextlib
import logging
import math
import os
import re

import torch
import torch.nn.functional as F

from rfproteina.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
from triadj_ext.pair_stream import RING_MASK, RING_N_STATES, RING_PAD

NO_INTERACTION = RING_N_STATES - 1  # 22

DEFAULT_RING_CFG = dict(
    clock_mode="tube",  # "tube" (t_ring = t_ca - lag + sigma*N(0,1), clipped) | "indep" (U(0,1), the spike's version)
    lag=0.10,  # structure leads: at a given step the ring clock is behind the CA clock by this much
    sigma=0.15,
    p_indep=0.20,  # fraction of examples that draw the ring clock independently from U(0,1) (coverage of all states)
    class_w_nointer=0.2,  # CE weight of NO_INTERACTION (TRIAD_J: [1]*22 + [0.2]); sampler adds -log(w) back (int_bias)
    w_clip=20.0,  # MDLM weight 1/(1-t) is clipped at this value
    loss_scale=1.0,
    # arm C (per-block heads, model_s2.TriadRTransformer): auxiliary CE of every block head on the same hidden pairs, logged as
    # ``ring_prediction_blk<i>`` (same scale as ``ring_prediction``); the objective weights are applied in ``optimization_terms``.
    final_weight=1.0,  # objective weight of ``ring_prediction`` (the final head)
    block_weight=0.5,  # objective weight of every ``ring_prediction_blk<i>`` ...
    block_weights=None,  # ... unless this mapping {"4": w, ...} names the block
    # PINNED rows (training-side counterpart of the inference condition `pinned`: k true positives revealed, EVERYTHING else MASK, at every noise level). Default OFF (p_pinned = 0: no extra RNG draw,
    # no extra batch key, no extra loss key: bit-identical to the tube-only behaviour).
    p_pinned=0.0,  # per-row probability (rows with >= 1 observed positive pair) that the input stream is a pinned row instead of a tube row
    pinned_min_k=3,  # k ~ log-uniform over [pinned_min_k, max(pinned_min_k, pinned_max_frac x #observed positives)]
    pinned_max_frac=0.5,
    pinned_hotspot_p=0.5,  # within pinned rows: probability that the k LONGEST-range positives (|i-j| >= pinned_min_sep, ties by class strength) are shown instead of k random positives
    pinned_min_sep=12,
)

log = logging.getLogger(__name__)

RING_VAL_MODES = {0: "tube", 1: "all-MASK", 2: "full reveal", 3: "shuffled full reveal"}  # batch['ring_val_mode'] [b] (validation only; absent/0 = training tube)
SAMPLER_MODES = ("generated", "masked", "oracle", "pinned", "hybrid")
#: states 11..21 = the eleven interaction types between DIFFERENT pn_units (ring_data: state = type + N_TYPES(11) * inter); they cannot occur inside one chain.
INTER_STATES = tuple(range(11, 22))
#: class strength rank of the 11 interaction types (ring_data: state = type + 11 * inter; COLLAPSE_PRIORITY strongest first): lower = stronger. Used to break ties between equally distant hotspots.
_TYPE_NAMES = ("hbond", "salt_bridge", "ionic", "hydrophobic", "pi_stack", "pi_cation", "disulfide", "vdw", "metal", "halogen", "pi_hbond")
_COLLAPSE = ("disulfide", "metal", "salt_bridge", "ionic", "pi_cation", "pi_stack", "pi_hbond", "hbond", "halogen", "hydrophobic", "vdw")
TYPE_RANK = tuple(_COLLAPSE.index(t) for t in _TYPE_NAMES)


def pair_mask_of(batch):
    pm = batch["padding_mask"]
    return pm[:, :, None] & pm[:, None, :]


def offdiag(like):
    n = like.shape[-1]
    return ~torch.eye(n, dtype=torch.bool, device=like.device)[None]


def symmetric_uniform(shape, device, generator=None):
    u = torch.rand(shape, device=device, generator=generator)
    u = torch.triu(u, 1)
    return u + u.transpose(-1, -2)  # symmetric, diagonal 0


def _sym_upper(x):
    """Take the upper triangle and mirror it: forces symmetry whatever the data builder delivered."""
    u = torch.triu(x, 1)
    return u + u.transpose(-1, -2)


def _has(batch, key):
    try:
        return key in batch and batch[key] is not None
    except TypeError:
        return False


def get_gt_and_obs(batch):
    """-> (gt long [b,n,n] symmetric with codes 0..22 valid, obs bool [b,n,n] symmetric) or (None, None)."""
    if not _has(batch, "ring_state_gt"):
        return None, None
    gt = batch["ring_state_gt"].long()
    obs = (gt < RING_N_STATES) & (gt >= 0)
    obs = torch.triu(obs, 1)
    obs = obs | obs.transpose(-1, -2)
    return _sym_upper(gt.clamp(0, RING_N_STATES - 1)), obs


class TriadJFlowMatcher(ProductSpaceFlowMatcher):
    ring_cfg = DEFAULT_RING_CFG

    def _rc(self, key):
        loss_cfg = getattr(self.cfg_exp, "loss", None)
        user = loss_cfg.get("ring") if loss_cfg is not None and hasattr(loss_cfg, "get") else None
        if user is not None and key in user:
            return user[key]
        return self.ring_cfg[key]

    # ---- (1) third clock: psfm:275 sample_t ------------------------------------------------------------------
    def _ring_clock(self, t_ca, generator=None):
        u = torch.rand(t_ca.shape, device=t_ca.device, generator=generator)
        if self._rc("clock_mode") == "indep":
            return u
        tube = (t_ca - self._rc("lag") + self._rc("sigma") * torch.randn(t_ca.shape, device=t_ca.device, generator=generator)).clamp(0.0, 1.0)
        indep = torch.rand(t_ca.shape, device=t_ca.device, generator=generator) < self._rc("p_indep")
        return torch.where(indep, u, tube)

    def sample_t(self, shape, device):
        t = super().sample_t(shape, device)  # {'bb_ca': [b], 'local_latents': [b]} (+ shared_groups handling)
        t["ring"] = self._ring_clock(t["bb_ca"])
        return t

    # ---- (2) corrupt the label map with the ring clock: psfm:222 corrupt_batch --------------------------------
    def corrupt_batch(self, batch, t=None):
        batch = super().corrupt_batch(batch, t=t)  # fills x_0/x_1/x_t and batch['t'] (= t)
        tt = batch["t"]
        had_ring_clock = "ring" in tt  # training passes the full t dict (sample_t); val / partial-diffusion paths pass data-mode clocks only (red-team #4)
        if "ring" not in tt:
            tt = {**tt, "ring": self._ring_clock(tt["bb_ca"])}
            batch["t"] = tt
        pair = pair_mask_of(batch)
        valid = pair & offdiag(pair)
        gt, obs = get_gt_and_obs(batch)
        mode = _val_mode(batch, pair)  # [b] long or None; validation-only per-row override of the ring clock
        if mode is not None:  # 1 = all-MASK (clock 0), 2/3 = full reveal (clock 1), 0 = whatever the tube drew; written back so
            t_eff = tt["ring"].to(pair.device)  # that the loss weight and the validation t-bins see the effective clock
            one, zero = torch.ones_like(t_eff), torch.zeros_like(t_eff)
            t_eff = torch.where(mode == 1, zero, torch.where((mode == 2) | (mode == 3), one, t_eff))
            tt = {**tt, "ring": t_eff}
            batch["t"] = tt
        if gt is None:  # inference / unlabelled data: nothing revealed
            st = torch.full(pair.shape, RING_MASK, dtype=torch.long, device=pair.device)
        else:
            t_ring = tt["ring"].to(gt.device)[:, None, None]  # t=1 clean, t=0 all MASK
            # reveal locations are drawn over ALL valid pairs, independent of coverage (a coverage-dependent reveal
            # pattern tells the network which residues RING touched, i.e. burial, and differs from the sampler,
            # which reveals everywhere; Codex red-team #1). Unobserved pairs are shown as NO_INTERACTION when
            # revealed (TRIAD_J precedent) and are never trained.
            has_ring = obs.flatten(1).any(1)[:, None, None]  # rows without any observed pair (replay/unlabelled data, collate fills 24) stay all-MASK
            reveal = (symmetric_uniform(gt.shape, gt.device) < t_ring) & has_ring
            shown = torch.where(obs, gt, torch.full_like(gt, NO_INTERACTION))
            st = torch.where(reveal, shown, torch.full_like(gt, RING_MASK))
            if mode is not None and bool((mode == 3).any()):  # one scalar sync per call; modes 0-2 take the old path unchanged
                # mode 3 = SHUFFLED control: the full label map of ANOTHER sample. Per row the donor is the nearest row in the cyclic
                # order (r-1, r-2, ...) whose map really differs (see _shuffle_donors); never the row's own map or a replicate.
                # Pair (i,j) takes the donor's (i,j); the receiver's own `valid` mask below crops/pads it (padding and diagonal stay PAD;
                # receiver pairs outside the donor's valid size are unobserved there and show NO_INTERACTION). Revealed everywhere,
                # whether or not the receiver has labels.
                donor = _shuffle_donors(batch, gt, obs, valid, mode)
                st = torch.where((mode == 3)[:, None, None], shown[donor], st)
            if had_ring_clock and float(self._rc("p_pinned")) > 0.0:  # training only (val / sampling paths carry no ring clock); p_pinned = 0 draws nothing
                pinned, st_pin, k_shown, t_pin = self._pinned_rows(gt, obs, valid, st)
                if mode is not None:  # a validation-style per-row mode (1 / 2 / 3) keeps its row exactly as the mode made it
                    pinned = pinned & (mode == 0)
                st = torch.where(pinned[:, None, None], st_pin, st)
                batch["ring_pinned"] = pinned.float()
                batch["ring_pinned_k"] = torch.where(pinned, k_shown, torch.zeros_like(k_shown)).float()
                tt = {**tt, "ring": torch.where(pinned, t_pin.to(tt["ring"].dtype), tt["ring"].to(pinned.device))}  # MDLM weight 1/(1-t) uses the REVEALED fraction (~0 for a pinned row)
                batch["t"] = tt
        batch["ring_state"] = torch.where(valid, st, torch.full_like(st, RING_PAD))  # network input key
        return batch

    def _pinned_rows(self, gt, obs, valid, st):
        """PINNED rows: per row, with probability ``p_pinned`` (rows with >= 1 observed positive pair), replace the tube input by k TRUE positive pairs shown with their labels and EVERYTHING else MASK
        (never NO_INTERACTION), independent of the noise level. k is log-uniform over [pinned_min_k, max(pinned_min_k, pinned_max_frac x #observed positives)] (capped by the number of positives);
        with probability ``pinned_hotspot_p`` the k positives with the largest |i-j| (>= pinned_min_sep; ties by class strength, then position) are shown instead of k random ones. Symmetric.
        The loss target stays "every OBSERVED pair that is MASK in the input" (hidden = observed & not shown), exactly as for tube rows. RNG: four draws (row, k, hotspot, pair keys), taken AFTER the tube
        draws, so the tube rows of the same seed are unchanged. -> (pinned [b] bool, st, k_shown [b] long, revealed fraction [b] float)"""
        B, n, _ = gt.shape
        dev = gt.device
        u_row, u_k, u_hot = torch.rand(B, device=dev), torch.rand(B, device=dev), torch.rand(B, device=dev)
        keys = torch.rand(B, n, n, device=dev)
        tri = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), 1)[None]
        pos = tri & valid & obs & (gt < NO_INTERACTION)  # observed positive pairs, i < j
        n_pos = pos.flatten(1).sum(1)
        kmin = float(self._rc("pinned_min_k"))
        hi = torch.clamp(float(self._rc("pinned_max_frac")) * n_pos.float(), min=kmin)
        k = torch.exp(math.log(kmin) + u_k * (torch.log(hi) - math.log(kmin))).round()
        k = torch.minimum(k, n_pos.float()).long()
        idx = torch.arange(n, device=dev)
        sep = (idx[None, :] - idx[:, None])[None].expand(B, n, n)
        rank = torch.tensor(TYPE_RANK, device=dev, dtype=torch.float32)[gt.clamp(0, NO_INTERACTION - 1) % 11]
        long_ok = pos & (sep >= int(self._rc("pinned_min_sep")))
        n_long = long_ok.flatten(1).sum(1)
        hot = (u_hot < float(self._rc("pinned_hotspot_p"))) & (n_long > 0)
        inf = float("inf")
        key = torch.where(hot[:, None, None], torch.where(long_ok, -(sep.float() * 16.0 - rank), torch.full_like(keys, inf)), torch.where(pos, keys, torch.full_like(keys, inf)))
        key = key.flatten(1)
        order = torch.sort(key, dim=1, stable=True).indices  # best first; ties (hotspots of equal separation and strength) by flat index = (i, j)
        rk = torch.empty_like(order)
        rk.scatter_(1, order, torch.arange(n * n, device=dev)[None].expand(B, -1))
        take = torch.where(hot, torch.minimum(k, n_long), k)
        sel = ((rk < take[:, None]) & torch.isfinite(key)).view(B, n, n)
        sel = sel | sel.transpose(1, 2)
        pinned = (u_row < float(self._rc("p_pinned"))) & obs.flatten(1).any(1) & (n_pos > 0)
        st_pin = torch.where(sel, gt, torch.full_like(gt, RING_MASK))
        st = torch.where(pinned[:, None, None], st_pin, st)
        k_shown = (sel.flatten(1).sum(1) // 2)
        t_pin = sel.flatten(1).sum(1).float() / valid.flatten(1).sum(1).clamp_min(1).float()
        return pinned, st, k_shown, t_pin

    # ---- (3) pair-label loss: psfm:367 compute_aux_loss + psfm:325 optimization_terms -------------------------
    def _ring_ce(self, logits, gt, masked, valid, w):
        """MDLM-weighted class-weighted CE of one head on the hidden observed pairs -> ([b] objective-scale term, [b] mean CE)."""
        cw = torch.ones(RING_N_STATES, device=logits.device)
        cw[NO_INTERACTION] = self._rc("class_w_nointer")
        ce = F.cross_entropy(logits.float().permute(0, 3, 1, 2), gt, weight=cw, reduction="none")  # [b,n,n]
        n_valid = valid.sum((1, 2)).clamp_min(1)  # MDLM: sum over masked / all valid, times 1/(1-t)
        base = (ce * masked).sum((1, 2)) / n_valid
        mean_ce = (ce * masked).sum((1, 2)) / masked.sum((1, 2)).clamp_min(1)  # logging: mean CE over hidden pairs
        return base * w * self._rc("loss_scale"), mean_ce

    def compute_aux_loss(self, batch, nn_out):
        losses = super().compute_aux_loss(batch, nn_out)
        inter = nn_out["interaction"]
        logits = inter["logits"]  # [b,n,n,23] from pair_stream.TriadJLocalLatentsTransformer
        blocks = inter.get("blocks") or {}  # {block index: [b,n,n,23]} from model_s2.TriadRTransformer (arm C)
        gt, obs = get_gt_and_obs(batch)
        if gt is None:  # keeps every ring parameter in the graph (DDP) and the term a [b] tensor
            losses["ring_prediction"] = 0.0 * logits.float().sum((1, 2, 3))
            for i, lg in blocks.items():
                losses[f"ring_prediction_blk{i}"] = 0.0 * lg.float().sum((1, 2, 3))
            return losses
        pair = pair_mask_of(batch)
        valid = pair & offdiag(pair) & obs  # observed, off-diagonal, real residues
        masked = (batch["ring_state"] == RING_MASK) & valid  # loss only on HIDDEN observed pairs (MDLM)
        t_ring = batch["t"]["ring"].to(logits.device)
        w = 1.0 / (1.0 - t_ring).clamp_min(1.0 / self._rc("w_clip"))
        if _has(batch, "ring_pinned"):  # p_pinned > 0: the fraction of pinned rows and their reveal count k are logged (never part of the objective)
            losses["ring_pinned_justlog"] = batch["ring_pinned"].to(logits.device).float()
            losses["ring_pinned_k_justlog"] = batch["ring_pinned_k"].to(logits.device).float()
            # the EFFECTIVE ring clock of every physical row (pinned rows carry their revealed fraction): as a per-sample loss key it is gathered by the exploration winner together with the losses, so a
            # logger can bucket winners by the clock of the attempt that actually won (outputs['t'] is thinned [::K], the FIRST attempt of each example, whose pinned status may differ)
            losses["ring_clock_justlog"] = batch["t"]["ring"].to(logits.device).float()
        losses["ring_prediction"], losses["ring_ce_justlog"] = self._ring_ce(logits, gt, masked, valid, w)  # [b] like every other term (winner gather psfm/proteina:676)
        for i, lg in blocks.items():  # per-block auxiliary heads: unweighted logs here, objective weight in optimization_terms
            losses[f"ring_prediction_blk{i}"], losses[f"ring_ce_blk{i}_justlog"] = self._ring_ce(lg, gt, masked, valid, w)
        return losses

    def _block_weight(self, key):
        bw = self._rc("block_weights")
        i = key[len("ring_prediction_blk"):]
        if bw is not None:
            for k in (i, int(i)):
                if k in bw:
                    return float(bw[k])
        return float(self._rc("block_weight"))

    def optimization_terms(self, losses):
        terms = super().optimization_terms(losses)  # hard-coded tuple at psfm:328-330
        if "ring_prediction" in losses:
            terms["ring_prediction"] = losses["ring_prediction"] * float(self._rc("final_weight"))
        for k in losses:
            if k.startswith("ring_prediction_blk"):
                terms[k] = losses[k] * self._block_weight(k)
        return terms


def _example_groups(batch, b):
    """Per-row identity of the source structure from ``example_id_batch`` (None when absent): the 4-character PDB id inside ``{...}`` of
    atomworks example ids (so two chains/crops of the same entry count as the SAME source), else the whole id string."""
    try:
        ids = batch.get("example_id_batch") if hasattr(batch, "get") else None
    except Exception:  # noqa: BLE001
        ids = None
    if ids is None or len(ids) != b:
        return None
    out = []
    for x in ids:
        x = str(x)
        m = re.search(r"\{([0-9A-Za-z]{4})\}", x)
        out.append(m.group(1).lower() if m else x)
    return out


def _shuffle_donors(batch, gt, obs, valid, mode):
    """Donor row per receiver for ``ring_val_mode=3`` -> LongTensor [b] (rows that are not mode 3 get themselves).
    For receiver r the candidates are j = r-1, r-2, ... (cyclic); the first j with (a) an observed label map (>= 1 observed pair),
    (b) a map that DIFFERS from r's own on the overlap of both valid regions (observed masks differ or labels differ where both
    observed; replicates / identical copies fail this) and (c) a different source entry when example ids are available, is taken.
    No such row in the micro-batch -> ValueError (never the own map)."""
    b = gt.shape[0]
    if b < 2:
        raise ValueError("ring_val_mode=3 (shuffled control) needs a donor: batch size >= 2")
    ov = valid[:, None] & valid[None]  # [b,b,n,n] pairs valid in both rows
    diff = ((obs[:, None] != obs[None]) | (obs[:, None] & obs[None] & (gt[:, None] != gt[None]))) & ov
    differs = diff.flatten(2).any(-1).cpu()  # [b,b]: map of row j differs from row r's on their overlap
    has_obs = obs.flatten(1).any(1).cpu()
    groups = _example_groups(batch, b)
    mode_c = mode.cpu()
    donor = list(range(b))
    for r in range(b):
        if int(mode_c[r]) != 3:
            continue
        for k in range(1, b):
            j = (r - k) % b
            if bool(has_obs[j]) and bool(differs[r, j]) and (groups is None or groups[j] != groups[r]):
                donor[r] = j
                break
        else:
            raise ValueError(
                f"ring_val_mode=3: row {r} has no usable donor in this micro-batch of {b} (need another source entry with an observed "
                f"label map that differs from its own; replicates/duplicates do not count)"
                + (f"; sources={groups}" if groups is not None else "; no example ids in the batch, compared label content only"))
    return torch.tensor(donor, dtype=torch.long, device=gt.device)


def single_chain_rows(batch):
    """bool [b]: rows whose valid tokens carry ONE chain id (a monomer: no inter-chain state can exist); None when the batch has no ``chain_iid`` (then nothing is masked)."""
    try:
        ci = batch.get("chain_iid") if hasattr(batch, "get") else None
    except Exception:  # noqa: BLE001
        ci = None
    if ci is None or not torch.is_tensor(ci):
        return None
    pm = batch["padding_mask"].bool()
    ci = ci.long()
    return torch.tensor([int(torch.unique(ci[r][pm[r]]).numel()) <= 1 for r in range(pm.shape[0])], dtype=torch.bool, device=pm.device)


def inter_forbid(batch):
    """bool [b,23] for ``reveal_step(forbid=...)``: True for the inter-chain classes on single-chain rows; None when no row is single-chain / chains unknown."""
    one = single_chain_rows(batch)
    if one is None or not bool(one.any()):
        return None
    cls = torch.zeros(RING_N_STATES, dtype=torch.bool, device=one.device)
    cls[list(INTER_STATES)] = True
    return one[:, None] & cls[None, :]


def class_forbid(batch, pairwise: bool = False):
    """Class mask for ``reveal_step(forbid=...)``. ``pairwise=False`` (default; env ``TRIADR_RING_PAIR_FORBID`` unset) and rows without a chain id / batches whose rows are ALL single-chain: exactly ``inter_forbid``
    (bool [b,23] / None, unchanged: multi-chain rows are NOT constrained). ``pairwise=True`` and at least one multi-chain row (a complex): bool [b,n,n,23] PER PAIR -- a pair inside one chain can never be given an
    inter-chain state (11..21), a pair across two chains can never be given an intra-chain state (0..10); NO_INTERACTION is always allowed; a single-chain row of such a batch keeps the whole inter range forbidden
    (same as ``inter_forbid``). design_with_contacts sets the env for two-chain specs."""
    one = single_chain_rows(batch)
    if not pairwise or one is None or bool(one.all()):
        return inter_forbid(batch)
    ci = batch["chain_iid"].long().to(one.device)
    same = ci[:, :, None] == ci[:, None, :]  # [b,n,n]
    inter_cls = torch.zeros(RING_N_STATES, dtype=torch.bool, device=ci.device)
    inter_cls[list(INTER_STATES)] = True
    intra_cls = torch.zeros(RING_N_STATES, dtype=torch.bool, device=ci.device)
    intra_cls[: INTER_STATES[0]] = True
    return torch.where(same[..., None], inter_cls, intra_cls)


def _val_mode(batch, like):
    """``batch['ring_val_mode']`` [b] (0 tube, 1 all-MASK, 2 full reveal) as a long tensor, or None when the key is absent."""
    if not _has(batch, "ring_val_mode"):
        return None
    m = batch["ring_val_mode"].to(like.device).long().reshape(-1)
    if m.shape[0] != like.shape[0]:
        raise ValueError(f"ring_val_mode must be one value per row: {tuple(m.shape)} vs batch {like.shape[0]}")
    return m


# ---- (4) sampler: where the pair-label state is updated each step --------------------------------------------------
def reveal_step(ring_state, logits, frac_target, pair, argmax=False, int_bias=None, temp=None, bias=0.0, min_prob=0.0, only_no=False, forbid=None):
    """Masked-diffusion reveal: each still-MASK off-diagonal pair is revealed with the probability that brings the
    revealed fraction to ``frac_target`` (scalar or [b]); its value is sampled (or argmax) from the symmetric head.
    ``int_bias`` [23]: added to the logits (= -log class weight, undoes the training class weighting).
    Sampler-only diagnosis knobs (defaults = the behaviour above, bit for bit; composed AFTER ``int_bias``):
      ``bias``     added to the NO_INTERACTION logit (positive = fewer contacts)
      ``temp``     softmax temperature on the logits before sampling (None = 1; 0 = argmax at every step)
      ``min_prob`` a pair is revealed as a POSITIVE class only if the (tempered; T=1 when temp==0) probability of that class >= min_prob,
                   else as NO_INTERACTION
      ``only_no``  reveal NO_INTERACTION for every revealed pair (never positives)
      ``forbid``   bool [b,23] (or None): classes that can never be sampled for row b, or bool [b,n,n,23] per pair (``class_forbid`` on a complex) (logit -> -inf after ``int_bias``/``bias``, before temperature / softmax / argmax / ``min_prob``)"""
    valid = pair & offdiag(pair)
    masked = (ring_state == RING_MASK) & valid
    n_tot = valid.sum((1, 2)).clamp_min(1)
    revealed = 1.0 - masked.sum((1, 2)) / n_tot
    ft = torch.as_tensor(frac_target, device=ring_state.device, dtype=revealed.dtype).expand_as(revealed)
    p = ((ft - revealed) / (1.0 - revealed).clamp_min(1e-6)).clamp(0.0, 1.0)[:, None, None]
    lg = logits.float()
    if int_bias is not None:
        lg = lg + int_bias.to(lg.device)
    if bias:
        lg = lg.clone()
        lg[..., NO_INTERACTION] += bias
    if forbid is not None:
        fb = forbid.to(lg.device)
        lg = lg.masked_fill(fb[:, None, None, :] if fb.dim() == 2 else fb, float("-inf"))
    argmax = argmax or (temp is not None and temp == 0)
    if temp is not None and temp > 0 and temp != 1.0:
        lg = lg / temp
    if argmax:
        sample = lg.argmax(-1)
    else:
        sample = torch.distributions.Categorical(logits=lg).sample()  # [b,n,n]
        sample = torch.triu(sample, 1)
        sample = sample + sample.transpose(1, 2)  # symmetric
    if only_no:
        sample = torch.full_like(sample, NO_INTERACTION)
    elif min_prob and min_prob > 0:
        pr = torch.softmax(lg, -1).gather(-1, sample.unsqueeze(-1)).squeeze(-1)  # probability of the class that was picked
        sample = torch.where((sample != NO_INTERACTION) & (pr < min_prob), torch.full_like(sample, NO_INTERACTION), sample)
    reveal = masked & (symmetric_uniform(ring_state.shape, ring_state.device) < p)
    return torch.where(reveal, sample, ring_state)


# ---- generated-label sink: a generate entry registers a callable and receives one record per sampled batch -------
RING_SINKS: list = []


def add_ring_sink(fn) -> None:
    """Register ``fn(record: dict)``; called at the end of every ``full_simulation``/``partial_simulation`` with
    ``{mode, example_ids, ring_state int8 [b,n,n] cpu, padding_mask bool [b,n] cpu, trace [steps] revealed fractions}``.
    Idempotent per function object. (Upstream's ``vanilla.generate`` drops ``fm.last_ring_state``; this is the hook.)"""
    if fn not in RING_SINKS:
        RING_SINKS.append(fn)


def remove_ring_sink(fn) -> None:
    if fn in RING_SINKS:
        RING_SINKS.remove(fn)


def sampler_mode(default: str = "generated") -> str:
    """``TRIADR_RING_SAMPLER`` = generated (reveal from the head, the model's own labels) | masked (never reveal: the
    stream stays all-MASK = the stream-masked condition of gate (a)) | oracle (``batch['ring_state_gt']`` shown from step 0, unobserved pairs as NO_INTERACTION) | pinned (ONLY the labelled pairs of
    ``batch['ring_state_gt']`` are shown, everything else stays MASK for the whole run: sparse constraints) | hybrid (the labelled pairs of ``batch['ring_state_gt']`` are PINNED exactly as in
    ``pinned`` -- shown from step 0 and re-written after every reveal -- while every other pair follows the ``generated`` schedule: the model's own labels, with the temperature / lag / inter-state masking knobs)."""
    m = os.environ.get("TRIADR_RING_SAMPLER", default).strip().lower()
    if m not in SAMPLER_MODES:
        raise ValueError(f"TRIADR_RING_SAMPLER must be one of {SAMPLER_MODES}, got {m!r}")
    return m


def _env_float(name):
    v = os.environ.get(name, "").strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError as e:
        raise ValueError(f"{name} must be a float, got {v!r}") from e


def sampler_knobs() -> dict:
    """Sampler-only diagnosis knobs from the environment (``generated`` mode only; unset = current behaviour exactly):
    TRIADR_RING_TEMP (>= 0; 0 = argmax every step), TRIADR_RING_MIN_PROB ([0,1]), TRIADR_RING_LAG (replaces loss.ring.lag in the reveal
    schedule frac = t_ca - lag), TRIADR_RING_BIAS (added to the NO_INTERACTION logit), TRIADR_RING_ONLYNO (1 = reveal only NO_INTERACTION),
    TRIADR_RING_ALLOW_INTER (1 = the OLD behaviour: sample inter-chain states 11..21 even for single-chain batches). Default: for every row of a batch that has ONE chain id the inter-chain classes are
    set to -inf before softmax / argmax / MIN_PROB (a monomer cannot have inter-chain contacts; the old sampler revealed ~150-200 of them per design). Multi-chain rows are untouched.
    LIMITATION of TRIADR_RING_LAG: the reveal schedule is frac = t_ca - lag, so at LAG >= 1 every network prediction during sampling sees ZERO revealed pairs; the exported final label map is still
    completed (argmax of the last step) AFTER the last prediction and cannot steer it -- such a run is a masked sampler whose side output is a map, not a conditioned one.
    TRIADR_RING_ISOLATE_RNG (1 = the label reveal draws its random numbers inside ``torch.random.fork_rng``: the global RNG that drives the structure
    noise is left exactly as if no label had been drawn, so ``generated`` and ``masked``/``oracle`` runs with one seed share the SAME structure noise)."""
    k = {"temp": _env_float("TRIADR_RING_TEMP"), "min_prob": _env_float("TRIADR_RING_MIN_PROB"), "lag": _env_float("TRIADR_RING_LAG"),
         "bias": _env_float("TRIADR_RING_BIAS"), "only_no": os.environ.get("TRIADR_RING_ONLYNO", "").strip().lower() in ("1", "true", "yes"),
         "isolate_rng": os.environ.get("TRIADR_RING_ISOLATE_RNG", "").strip().lower() in ("1", "true", "yes"),
         "allow_inter": os.environ.get("TRIADR_RING_ALLOW_INTER", "").strip().lower() in ("1", "true", "yes")}
    if k["temp"] is not None and k["temp"] < 0:
        raise ValueError(f"TRIADR_RING_TEMP must be >= 0, got {k['temp']}")
    if k["min_prob"] is not None and not 0.0 <= k["min_prob"] <= 1.0:
        raise ValueError(f"TRIADR_RING_MIN_PROB must be in [0,1], got {k['min_prob']}")
    return k


def stream_cfg_w() -> float:
    """``TRIADR_STREAM_CFG`` = w (default 1 = off). Stream classifier-free guidance: at every sampler step the denoiser is evaluated twice on the SAME noisy state, once with the current stream (pinned / oracle /
    generated labels) and once with the stream ALL-MASK, and each integrated quantity is combined as  pred = pred_masked + w * (pred_cond - pred_masked)  (w = 1: the conditional prediction, no second pass; the
    cost for w != 1 is 2x forward passes per step while any label is shown)."""
    v = _env_float("TRIADR_STREAM_CFG")
    w = 1.0 if v is None else v
    if not math.isfinite(w) or w < 0:  # nan / inf would turn every guided quantity into NaN (NaN * 0 = NaN, also where cond == masked)
        raise ValueError(f"TRIADR_STREAM_CFG must be a finite number >= 0, got {w}")
    return w


def stream_cfg_combine(nn_cond: dict, nn_masked: dict, w: float, data_modes, guided_modes) -> dict:
    """Combine two denoiser outputs (dict of dict of tensors, as the flow matcher returns them) in place on ``nn_cond``: for every data mode in ``guided_modes`` the INTEGRATED velocity
    ``v_guided = v_masked + w (v_cond - v_masked)`` (the key the integration step reads when guidance is enabled for the modality), and the sequence read-out logits
    ``sequence.logits = l_masked + w (l_cond - l_masked)``, and the CLEAN prediction of every guided mode ``x_1 = x1_masked + w (x1_cond - x1_masked)`` (= x_t + (1 - t) v_guided for the velocity parameterisation: the exported
    clean trajectory and the metric-guidance input are consistent with the integrated velocity and the sequence logits; ``v`` itself stays the conditional one). The interaction and insertion heads stay the conditional ones.
    Exact: cond == masked => result == cond for any finite w."""
    for dm in data_modes:
        if dm in guided_modes:
            vc, vu = nn_cond[dm]["v"], nn_masked[dm]["v"]
            nn_cond[dm]["v_guided"] = vu + w * (vc - vu)
            if "x_1" in nn_cond[dm] and "x_1" in nn_masked[dm]:
                xc, xu = nn_cond[dm]["x_1"], nn_masked[dm]["x_1"]
                nn_cond[dm]["x_1"] = xu + w * (xc - xu)
    if "sequence" in nn_cond and "sequence" in nn_masked and "logits" in nn_cond["sequence"] and "logits" in nn_masked["sequence"]:
        lc, lu = nn_cond["sequence"]["logits"], nn_masked["sequence"]["logits"]
        nn_cond["sequence"]["logits"] = lu + w * (lc - lu)
    return nn_cond


def _rng_snapshot(device) -> dict:
    """Torch RNG state (CPU + the batch's CUDA device) -- the stream-CFG second pass replays pass 1's random feature draws from it."""
    st = {"cpu": torch.get_rng_state()}
    if device is not None and getattr(device, "type", "") == "cuda":
        st["cuda"] = torch.cuda.get_rng_state(device)
    return st


def _rng_restore(st: dict, device) -> None:
    torch.set_rng_state(st["cpu"])
    if "cuda" in st:
        torch.cuda.set_rng_state(st["cuda"], device)


class TriadJSamplerFlowMatcher(TriadJFlowMatcher):
    """Adds the per-step pair-label update to the sampler loops WITHOUT copying them: ``full_simulation`` and
    ``partial_simulation`` call ``self.get_clean_pred_n_guided_vector(batch=..., ...)`` exactly once per step
    (psfm:809, :1030) with the batch object that carries ``batch['t']``, so the hook below sees every step.
    Beam search (search/) does not arm the hook: its ring_state stays whatever the caller put there (documented).
    The per-run policy is ``sampler_mode()`` (env ``TRIADR_RING_SAMPLER``, read at every ``_arm``)."""

    force_sampler_mode: str | None = None  # subclasses (control arm) pin the policy regardless of the environment

    def _arm(self, batch, nsteps):
        mode = self.force_sampler_mode or sampler_mode()
        pair = pair_mask_of(batch)
        valid = pair & offdiag(pair)
        mask_all = torch.full(pair.shape, RING_MASK, dtype=torch.long, device=pair.device)
        pad_all = torch.full(pair.shape, RING_PAD, dtype=torch.long, device=pair.device)
        if mode == "oracle":
            gt, obs = get_gt_and_obs(batch)
            if gt is None or not bool(obs.flatten(1).any(1).all()):  # collate zero-fills absent keys with 24 = "unobserved"
                raise ValueError("TRIADR_RING_SAMPLER=oracle needs batch['ring_state_gt'] with observed pairs in every row "
                                 "(a dataset that supplies RING labels)")
            shown = torch.where(obs, gt, torch.full_like(gt, NO_INTERACTION))  # same convention as a full reveal in training
            batch["ring_state"] = torch.where(valid, shown, pad_all)
        elif mode == "hybrid":
            # PINNED labels (like `pinned`) + the generated schedule for every other pair: the pinned pairs are re-written after every reveal, so a generated label never replaces them
            gt, obs = get_gt_and_obs(batch)
            if gt is None:
                raise ValueError("TRIADR_RING_SAMPLER=hybrid needs batch['ring_state_gt'] (codes 0..22 on the pinned pairs, 24 elsewhere)")
            pin = valid & obs
            batch["ring_state"] = torch.where(valid, torch.where(obs, gt, mask_all), pad_all)
            self._ring_pin = (pin, gt)
        elif mode == "pinned":
            # SPARSE map: ONLY the pairs with a label in batch['ring_state_gt'] (code < 23: 0..21 contacts, 22 NO_INTERACTION) are shown, with that label, from step 0 to the end; every other pair
            # stays MASK for the whole run (never revealed, never generated, never NO_INTERACTION). A row without any revealed pair is simply all-MASK.
            gt, obs = get_gt_and_obs(batch)
            if gt is None:
                raise ValueError("TRIADR_RING_SAMPLER=pinned needs batch['ring_state_gt'] (codes 0..22 on the revealed pairs, 24 elsewhere)")
            batch["ring_state"] = torch.where(valid, torch.where(obs, gt, mask_all), pad_all)
        else:
            batch["ring_state"] = torch.where(valid, mask_all, pad_all)
        # generated / hybrid: reveal hook armed for `nsteps` updates; masked/oracle/pinned: hook disarmed (0), the state never changes
        gen_like = mode in ("generated", "hybrid")
        if mode != "hybrid":
            self._ring_pin = None
        self._ring_i, self._ring_n, self._ring_trace, self._ring_mode = 0, (nsteps if gen_like else 0), [], mode
        self._stream_cfg = stream_cfg_w() if mode != "masked" else 1.0  # an all-MASK stream is its own unconditional pass: nothing to guide
        knobs = sampler_knobs()
        self._ring_knobs = knobs if gen_like else None  # diagnosis knobs act on the generated reveal only
        pair_forbid = os.environ.get("TRIADR_RING_PAIR_FORBID", "").strip().lower() in ("1", "true", "yes")  # complexes: per-pair class mask from the real chain ids (design_with_contacts, two-chain specs)
        self._ring_forbid = class_forbid(batch, pair_forbid) if (gen_like and (pair_forbid or not knobs["allow_inter"])) else None  # monomer rows: inter-chain classes can never be sampled
        if self._ring_forbid is not None and self._ring_forbid.dim() == 2 and knobs["allow_inter"]:
            self._ring_forbid = None  # TRIADR_RING_ALLOW_INTER drops the per-row MONOMER mask only; a complex's per-pair mask (TRIADR_RING_PAIR_FORBID=1) takes precedence over it
        one_ = single_chain_rows(batch) if self._ring_forbid is not None else None
        n_mono = int(one_.sum()) if one_ is not None else 0
        lag = knobs["lag"] if knobs["lag"] is not None else float(self._rc("lag"))
        eff = (f"temp={knobs['temp'] if knobs['temp'] is not None else 1.0} min_prob={knobs['min_prob'] or 0.0} lag={lag} "
               f"bias={knobs['bias'] or 0.0} onlyno={int(knobs['only_no'])} isolate_rng={int(knobs['isolate_rng'])} allow_inter={int(knobs['allow_inter'])} inter_masked_rows={n_mono}/{pair.shape[0]} stream_cfg={self._stream_cfg:g} class_w_nointer={self._rc('class_w_nointer')} nsteps={nsteps}" + (" pair_forbid=1" if pair_forbid else ""))
        ignored = "" if gen_like or not any(v for k, v in knobs.items() if k not in ("isolate_rng", "allow_inter")) else " (knobs set but IGNORED: not generated mode)"
        print(f"RING sampler knobs mode={mode} {eff}{ignored}", flush=True)

    def _disarm(self, batch):
        self.last_ring_state = batch["ring_state"].detach().clone() if _has(batch, "ring_state") else None
        self._ring_n = None
        if self.last_ring_state is not None:
            try:  # one line per sampled batch: the share of pairs revealed as a positive class (0..21) among all valid pairs
                pair = pair_mask_of(batch)
                valid = pair & offdiag(pair)
                rs = self.last_ring_state
                nv = valid.sum().clamp_min(1)
                n_inter = int(((rs >= INTER_STATES[0]) & (rs <= INTER_STATES[-1]) & valid).sum()) // 2
                print(f"RING sampler done mode={getattr(self, '_ring_mode', '?')} positive_frac={float(((rs < NO_INTERACTION) & valid).sum() / nv):.5f} inter_chain_pairs={n_inter} "
                      f"no_interaction_frac={float(((rs == NO_INTERACTION) & valid).sum() / nv):.5f} mask_frac={float(((rs == RING_MASK) & valid).sum() / nv):.5f}", flush=True)
            except Exception:  # noqa: BLE001
                log.exception("ring sampler summary failed; continuing")
        if RING_SINKS and self.last_ring_state is not None:
            try:
                ids = batch.get("example_id_batch") if hasattr(batch, "get") else None
                rec = {
                    "mode": getattr(self, "_ring_mode", "generated"),
                    "example_ids": [str(x) for x in ids] if ids is not None else None,
                    "ring_state": self.last_ring_state.to("cpu", torch.int8),
                    "padding_mask": batch["padding_mask"].detach().to("cpu"),
                    "trace": list(getattr(self, "_ring_trace", [])),
                }
                for fn in list(RING_SINKS):
                    fn(rec)
            except Exception:  # noqa: BLE001  (never let a logging sink kill a generation run)
                log.exception("ring sink failed; continuing")

    def full_simulation(self, batch, *args, nsteps, **kwargs):
        self._arm(batch, nsteps)
        try:
            return super().full_simulation(batch, *args, nsteps=nsteps, **kwargs)
        finally:
            self._disarm(batch)

    def partial_simulation(self, batch, *args, nsteps, **kwargs):
        self._arm(batch, nsteps)
        try:
            return super().partial_simulation(batch, *args, nsteps=nsteps, **kwargs)
        finally:
            self._disarm(batch)

    def _stream_cfg_prediction(self, batch, predict_for_sampling, guidance_w, w):
        """Conditional prediction + (unless the stream is all-MASK already) the same prediction with the stream all-MASK, combined by ``stream_cfg_combine`` (see ``stream_cfg_w``)."""
        if guidance_w != 1.0:
            raise ValueError("TRIADR_STREAM_CFG != 1 cannot be combined with the upstream CFG (guidance_w != 1)")

        def one(b):
            d = predict_for_sampling(b, mode="full")
            d = self.nn_out_add_clean_sample_prediction(b, d)
            return self.nn_out_add_simulation_tensor(b, d)

        rs = batch["ring_state"]
        dev = rs.device
        rng0 = _rng_snapshot(dev)  # the forward draws random features (e.g. coord_group_seq randperm over known coordinate groups): both passes must see the SAME draws
        nn_c = one(batch)
        pair = pair_mask_of(batch)
        mask_state = torch.where(pair & offdiag(pair), torch.full_like(rs, RING_MASK), torch.full_like(rs, RING_PAD))
        guided = {dm for dm in self.data_modes if getattr(self.base_flow_matchers[dm], "guidance_enabled", False)}
        if torch.equal(rs, mask_state):  # nothing shown yet (or masked sampler): cond == masked exactly, no second pass
            return stream_cfg_combine(nn_c, nn_c, w, self.data_modes, guided)
        rng1 = _rng_snapshot(dev)  # after exactly ONE forward: what the w = 1 path leaves behind, so the integration noise that follows is identical across w
        _rng_restore(rng0, dev)  # pass 2 replays pass 1's draws: the stream is the ONLY thing that differs
        batch["ring_state"] = mask_state
        try:
            nn_u = one(batch)
        finally:
            batch["ring_state"] = rs
            _rng_restore(rng1, dev)
        return stream_cfg_combine(nn_c, nn_u, w, self.data_modes, guided)

    def get_clean_pred_n_guided_vector(self, batch, predict_for_sampling, guidance_w):
        w = getattr(self, "_stream_cfg", 1.0)
        if w == 1.0:
            nn_out = super().get_clean_pred_n_guided_vector(batch, predict_for_sampling, guidance_w)
        else:
            nn_out = self._stream_cfg_prediction(batch, predict_for_sampling, guidance_w, w)
        if getattr(self, "_ring_n", None) and "interaction" in nn_out:
            self._ring_i += 1
            last = self._ring_i >= self._ring_n
            pair = pair_mask_of(batch)
            kn = getattr(self, "_ring_knobs", None) or {}
            lag = kn["lag"] if kn.get("lag") is not None else self._rc("lag")  # TRIADR_RING_LAG overrides loss.ring.lag in the sampler only
            try:  # structure leads: ring clock = structure clock - lag, same rule as the training tube
                t_ca = batch["t"]["bb_ca"].to(pair.device).float()
                frac = (t_ca - lag).clamp(0.0, 1.0)
            except (KeyError, TypeError):
                frac = torch.full((pair.shape[0],), self._ring_i / self._ring_n, device=pair.device)
            if last:
                frac = torch.ones_like(frac)
            bias = -torch.log(torch.cat([torch.ones(NO_INTERACTION), torch.tensor([self._rc("class_w_nointer")])]))
            dev = batch["ring_state"].device
            iso = (torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else [], enabled=True) if kn.get("isolate_rng") else contextlib.nullcontext())
            with iso:
                batch["ring_state"] = reveal_step(
                    batch["ring_state"], nn_out["interaction"]["logits"], frac, pair, argmax=last, int_bias=bias,
                    temp=kn.get("temp"), bias=kn.get("bias") or 0.0, min_prob=kn.get("min_prob") or 0.0, only_no=bool(kn.get("only_no")),
                    forbid=getattr(self, "_ring_forbid", None),
                )
            pin = getattr(self, "_ring_pin", None)
            if pin is not None:  # hybrid: the pinned labels always overwrite whatever was generated
                batch["ring_state"] = torch.where(pin[0], pin[1], batch["ring_state"])
            valid = pair & offdiag(pair)
            self._ring_trace.append(float(((batch["ring_state"] != RING_MASK) & valid).sum() / valid.sum().clamp_min(1)))
        return nn_out


class TriadJControlFlowMatcher(TriadJSamplerFlowMatcher):
    """Arm Z (control): same model/optimiser/data as the other arms, but the interaction stream is forced all-MASK in the
    input (training and sampling) and the ring losses are multiplied by 0.0 (kept as [b] tensors in the graph, so a DDP run
    sees the same parameter usage). Isolates the effect of the stream from LoRA/replay/K drift."""

    force_sampler_mode = "masked"

    def corrupt_batch(self, batch, t=None):
        batch = super().corrupt_batch(batch, t=t)
        st = batch["ring_state"]
        batch["ring_state"] = torch.where(st == RING_PAD, st, torch.full_like(st, RING_MASK))
        return batch

    def compute_aux_loss(self, batch, nn_out):
        losses = super().compute_aux_loss(batch, nn_out)
        return {k: (0.0 * v if k.startswith("ring_") else v) for k, v in losses.items()}


def register_flow() -> None:
    """Proteina.__init__ builds ``self.fm = ProductSpaceFlowMatcher(cfg_exp)`` (proteina.py:418, module global)."""
    import rfproteina.proteina as P

    P.ProductSpaceFlowMatcher = TriadJSamplerFlowMatcher


def register_optimizer_split(pattern: str = r"^nn\.ring_") -> None:
    """Keep the new embedding/head matrices on Adam: Muon takes every 2-D parameter unless a name matches
    ``opt.muon_exclude_param_patterns`` (optim_utils:52-56), and a Hydra list override REPLACES the bundle's list.
    Appends ``pattern`` in code (red-team #6). Idempotent."""
    import rfproteina.proteina as P

    if getattr(P.build_optimizers, "_triadj_wrapped", False):
        return
    orig = P.build_optimizers

    def wrapped(named_parameters, cfg_opt):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(OmegaConf.to_container(cfg_opt, resolve=True))
        pats = list(cfg.get("muon_exclude_param_patterns") or [])
        if pattern not in pats:
            pats.append(pattern)
        cfg["muon_exclude_param_patterns"] = pats
        return orig(named_parameters, cfg)

    wrapped._triadj_wrapped = True
    P.build_optimizers = wrapped
