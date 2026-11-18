"""P0 FROZEN-TRUNK PROBE (TRIAD-R gate P0, docs/TRIAD_R_PLAN.md section 4).

Question: does the FROZEN released RFD4-Proteina trunk already encode our 23-state RING pair interactions in its final pair
state z, at which noise levels, and by how much more than a distance-only baseline?

Everything except the probe heads is frozen (eval, no grad, bf16 autocast on GPU / fp32 on CPU). The trunk is the released
network run through ``pair_stream.load_triadj_proteina`` (zero-initialised ``ring_embed``), so it IS the released model; the
probe input carries ``ring_state`` = all-MASK (and never ``ring_state_gt`` / ``ring_obs``), asserted in code
(``make_probe_input`` / ``assert_probe_input_clean``) and in tests.

Variants, all trained in ONE run on identical (noisy x_t, trunk) features:
  V0  distance-only baseline: embeddings of binned CA-CA distance of the NOISY x_t (64 bins), bond-path seq-sep bin (64),
      same-chain flag, noise-level embedding -> 384-d -> shared head.
  V1  head on z0 (pre-trunk pair input, ``pair_repr_builder`` output).
  V2  head on final z (last recycle pass).
  V3  head on final z + outer-sum of the final single state s: z_ij + P(s_i) + P(s_j), P = LayerNorm(768)->Linear(768,384).
  V4  (extra, not in the spec) head on z0 + outer-sum of the PRE-trunk single state s0: separates "the trunk computed
      something" (V2/V3 > V4) from "the trunk's inputs (noisy latents, token types) already carry it".
  V5  (extra, ORACLE reference, not a trunk probe) V0 fed the CLEAN x_1 CA distances: what any CA-geometry-only reader could do
      with perfect denoising. V2/V3 > V0 alone may just be geometry denoising; V2/V3 > V5 means info beyond CA geometry
      (side-chain / sequence / type content), i.e. what S2 would otherwise have to teach.
Besides all observed pairs, AUC/AP are also reported on the HARD stratum (non-local spatial neighbours, see HARD_*).
Every head: LayerNorm(384) -> Linear(384,512) -> GELU -> Linear(512,23), logits symmetrised as 0.5*(l + l^T) (= (l + l^T) up
to a factor 2, as in pair_stream's ring head), class-weighted CE over OBSERVED pairs (ring_obs) with i<j only.
Class weights: TRIAD_J scheme [1]*22 + [0.2] (default, mirrors the planned S2 loss) or --class-weights freq ((n_c+1000)^-0.5,
normalised to expected weight 1). Evaluation removes the weight tilt (logits - log w) so P(any) and ECE are on the data prior.
The loss = sum_pairs w_y * CE / (UNWEIGHTED number of observed pairs in the batch): a target-independent normaliser, so the optimum is
the tilted posterior p(c|x) w_c / Z and subtracting log w at evaluation is exact (dividing by sum w_y would cancel the weights).
--init-heads restores the checkpoint's class weights/scheme. Validation rows are read in STRICT mode (a failing row is counted by
reason in result.json, never replaced; duplicates are skipped); train rows may be substituted and their counts are reported.
Top-L precision counts examples without a positive as 0 (``topl``); ``topl_pos`` is over examples with >= 1 positive.

Noise grid: paired (t_ca, t_lat) levels (t=0 noise, t=1 data), applied with the flow matcher's own ``corrupt_batch`` on the clean
x_1 from ``add_clean_samples`` (AE posterior sample). Train: one random level per example. Eval: every example at ALL levels with
common random numbers (same AE sample and same x_0 noise across levels, variants and recycle depths).

Data interface (Builder A's ``triadj_ext.ring_data``): batches are RFD4 training batches (dict-like of tensors, built with ALL
conditioning off) plus ``ring_state_gt`` [b,n,n] (0..21 RING types, 22 NO_INTERACTION, anything else = UNOBS) and ``ring_obs``
[b,n,n] bool. Wire it in ``build_loaders`` below (or pass --loader-factory path.py:func for tests / smokes).

Outputs (--out): result.json (all metrics, CIs, gains, decision rule), curves.npz (loss curves + metric arrays), heads.pt.
Log lines (stdout) are single ``P0 key=value ...`` lines, parseable for the login-node wandb cron.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------------------------- constants
N_CLS, N_INT, NO_INT = 23, 22, 22  # 0..21 RING states, 22 = NO_INTERACTION
RING_MASK, RING_PAD = 23, 24  # identical to triadj_ext.pair_stream (asserted at model load)
VARIANTS = ("V0", "V1", "V2", "V3", "V4", "V5")
DEFAULT_LEVELS = ((0.2, 0.02), (0.5, 0.1), (0.8, 0.4), (0.95, 0.8), (0.995, 0.98))  # (t_ca, t_lat); t=1 is clean data
NB_ANY, RNG_ANY = 4096, 16.0  # histogram bins over logit P(any), clipped to +-RNG
NB_CLS, RNG_CLS = 384, 12.0  # per-class one-vs-rest histograms over logit P(class)
N_ECE = 15
MIN_POS_CLASS = 200  # classes with >= this many held-out positives enter per-class AP / macro AP
V0_EDGES = torch.linspace(0.1, 3.2, 63)  # nm -> 64 distance bins (~0.5 A wide)
SEP_BINS = 64  # token_bond_path_length clamp (0 = other chain / disconnected / self)
D_PAIR, D_SINGLE, D_HID = 384, 768, 512
LABEL_KEYS = ("ring_state_gt", "ring_obs", "ring_state", "ring_val_mode")  # ring_val_mode: validation-only switch (S2), zero-filled by collate once registered; never a trunk input
GAIN_PAIRS = (("V2", "V0"), ("V2", "V1"), ("V3", "V0"), ("V3", "V1"), ("V3", "V2"), ("V2", "V4"), ("V3", "V4"), ("V4", "V0"), ("V2", "V5"), ("V3", "V5"), ("V5", "V0"))
GAIN_METRICS = ("auc", "ap", "macro_ap", "topl", "topl_pos", "auc_hard", "ap_hard")  # topl: all examples (no-positive = 0); topl_pos: examples with >= 1 positive
# "hard" stratum = non-local spatial neighbours: CLEAN CA-CA <= 12 A and (other chain / disconnected, or >= 4 bond hops apart).
# Among these proximity is no longer decisive, so AUC/AP here measure interaction content beyond coarse geometry.
HARD_MAX_DIST_NM, HARD_MIN_SEP = 1.2, 4
# Pre-registered decision rule (see decide()). Changing these after seeing results defeats the purpose.
RULE = dict(pivot_t_ca=0.5, min_gain_auc=0.03, min_gain_ap=0.03, min_gain_beyond_inputs=0.01, pc_min_auc=0.93, pc_tol=0.01)


def log(*a):
    print(*a, flush=True)


def kv(prefix: str, **kw) -> str:
    return prefix + " " + " ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items())


# --------------------------------------------------------------------------------------------------------- batch utilities
def unpack(batch, device):
    """dict-like RFD4 batch -> (plain dict of tensors on ``device``, meta). Non-tensor entries are dropped."""
    t = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
    for k in ("ring_state_gt", "ring_obs"):  # dataclass-style batches may keep extra keys outside .items()
        if k not in t and hasattr(batch, "get") and torch.is_tensor(batch.get(k)):
            t[k] = batch.get(k).to(device, non_blocking=True)
    get = batch.get if hasattr(batch, "get") else (lambda k, d=None: d)
    return t, {"example_ids": get("example_id_batch"), "clusters": get("cluster_id_batch"), "n_requested": get("n_requested_batch"),
               "n_failed": get("n_failed_batch"), "drops": get("drop_counts_batch")}


def index_batch(b: dict, keep: list, bsz: int) -> dict:
    idx = torch.as_tensor(keep, dtype=torch.long)
    return {k: (v[idx.to(v.device)] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == bsz else v) for k, v in b.items()}


def slice_batch(b: dict, i: int, j: int, bsz: int) -> dict:
    return {k: (v[i:j] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == bsz else v) for k, v in b.items()}


def label_masks(b: dict):
    """(y [b,n,n] long, m [b,n,n] bool): observed upper-triangle pairs of valid tokens; y in 0..22 where m."""
    y = b["ring_state_gt"].long()
    obs = b["ring_obs"].bool()
    pm = b["padding_mask"].bool()
    n = y.shape[1]
    tri = torch.ones(n, n, dtype=torch.bool, device=y.device).triu(1)
    m = obs & obs.transpose(1, 2) & pm[:, :, None] & pm[:, None, :] & tri[None]
    if bool((m & ((y < 0) | (y > NO_INT))).any()):
        raise ValueError("ring_obs pair with ring_state_gt outside 0..22 (UNOBS code under an observed mask)")
    return torch.where(m, y, torch.zeros_like(y)), m


def hard_mask(b: dict, ca_clean_nm: torch.Tensor) -> torch.Tensor:
    """[b,n,n] bool: clean CA-CA <= HARD_MAX_DIST_NM and (path 0 = other chain/disconnected, or >= HARD_MIN_SEP bond hops)."""
    d = (ca_clean_nm[:, :, None, :].float() - ca_clean_nm[:, None, :, :].float()).norm(dim=-1)
    pl = b["token_bond_path_length"].long()
    return (d <= HARD_MAX_DIST_NM) & ((pl == 0) | (pl >= HARD_MIN_SEP))


def assert_probe_input_clean(nb: dict) -> None:
    """The trunk input must not contain label information: no ring_state_gt / ring_obs / any other key starting with 'ring', and
    ring_state == MASK on every valid pair, PAD elsewhere."""
    bad = [k for k in nb if str(k).lower().startswith("ring") and k != "ring_state"]
    assert not bad, f"label keys reach the trunk: {bad}"
    rs, pm = nb["ring_state"], nb["padding_mask"].bool()
    pair = pm[:, :, None] & pm[:, None, :]
    assert bool((rs[pair] == RING_MASK).all()), "ring_state is not all-MASK on valid pairs (label leak)"
    assert bool((rs[~pair] == RING_PAD).all()), "ring_state is not PAD outside valid pairs"


def make_probe_input(batch: dict) -> dict:
    """Trunk input for probing: everything except label keys, plus ring_state = all-MASK / PAD (asserted)."""
    nb = {k: v for k, v in batch.items() if k not in LABEL_KEYS}
    pm = nb["padding_mask"].bool()
    pair = pm[:, :, None] & pm[:, None, :]
    nb["ring_state"] = torch.where(pair, RING_MASK, RING_PAD).to(torch.int16)
    assert_probe_input_clean(nb)
    return nb


_COND_FALSE = ("condition_hotspot_1_atom", "condition_pistack_1_atom", "condition_hydrophobic_1_atom")
_COND_ZERO = ("condition_hbond_1_atom", "condition_pication_1_atom", "condition_saltbridge_1_atom",
              "condition_metal-coord_1_atom", "condition_rasa-binned_1_atom", "condition_secondary-structure_1_residue")


def conditioning_problems(b: dict) -> list[str]:
    """1-body interaction / structure conditioning flags that would leak label-like information into s (unconditional task
    = unspecified: bools False, int codes 0, cath -1, coordinate conditioning absent)."""
    out = [k for k in _COND_FALSE if k in b and bool(b[k].any())]
    out += [k for k in _COND_ZERO if k in b and bool((b[k] != 0).any())]
    if "condition_cath-label_1_residue" in b and bool((b["condition_cath-label_1_residue"] >= 0).any()):
        out.append("condition_cath-label_1_residue")
    if "coord_cond_mask" in b and bool(b["coord_cond_mask"].any()):
        out.append("coord_cond_mask")
    return out


# -------------------------------------------------------------------------------------------------------- frozen trunk
class FeatureTap:
    """Captures (z0, s0, final z, final s) of the last trunk call by wrapping the instance's ``_run_trunk`` (a plain method,
    not a Module, so forward hooks cannot see it). ``z0``/``s0`` are the pre-trunk inputs (before any ring_embed addition)."""

    def __init__(self, nn_mod):
        self.nn, self.store = nn_mod, {}

    def install(self):
        orig = self.nn._run_trunk

        def wrapped(input, s_init, z_init, c, seq_mask):
            s, z = orig(input, s_init, z_init, c, seq_mask)
            self.store = {"z0": z_init, "s0": s_init, "z": z, "s": s}
            return s, z

        self.nn._run_trunk = wrapped
        return self

    def remove(self):
        self.nn.__dict__.pop("_run_trunk", None)


def load_frozen_trunk(bundle: str, device: torch.device):
    from triadj_ext import common  # sets CLUSTER/DATA_PATH/HF offline defaults
    from triadj_ext import pair_stream as ps

    assert (ps.RING_MASK, ps.RING_PAD) == (RING_MASK, RING_PAD), "state ids drifted from pair_stream"
    ov = None if device.type == "cuda" else dict(common.CPU_NN_OVERRIDES)
    model, _ = ps.load_triadj_proteina(bundle, nn_overrides=ov, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert float(model.nn.ring_embed.weight.abs().max()) == 0.0, "ring_embed must be zero: P0 probes the RELEASED trunk"
    return model


@torch.no_grad()
def clean_batch(model, b: dict) -> dict:
    b = dict(b)
    b["coords_nm"] = b["coords"] / 10.0  # proteina.py:625
    b["coord_mask"] = b["atom37_mask"]  # proteina.py:626
    return model.add_clean_samples(b)  # x_1 = {bb_ca, local_latents (AE posterior sample)}


@torch.no_grad()
def noisy_batch(model, clean: dict, lvl: torch.Tensor, levels_t, seed: int | None = None) -> dict:
    """fm.corrupt_batch exactly as in training, with per-example clocks (t_ca, t_lat) = levels[lvl]."""
    if seed is not None:
        torch.manual_seed(seed)
    t = {"bb_ca": levels_t[0][lvl], "local_latents": levels_t[1][lvl]}
    return model.fm.corrupt_batch(dict(clean), t=t)


def v0_inputs(noisy: dict, lvl: torch.Tensor, clean: bool = False) -> dict:
    x = (noisy["x_1"] if clean else noisy["x_t"])["bb_ca"].float()  # [b,n,3] nm: the noisy CA the trunk sees (V0) / clean CA (V5 oracle)
    d = (x[:, :, None, :] - x[:, None, :, :]).norm(dim=-1)
    ch = noisy["chain_iid"].long()
    return {"dbin": torch.bucketize(d, V0_EDGES.to(d.device)), "sep": noisy["token_bond_path_length"].long().clamp(0, SEP_BINS - 1),
            "same": (ch[:, :, None] == ch[:, None, :]).long(), "lvl": lvl}


@torch.no_grad()
def extract_features(model, tap: FeatureTap, noisy: dict, lvl: torch.Tensor, n_recycle: int) -> dict:
    nb = make_probe_input(noisy)
    tap.store = {}
    model.call_nn(nb, n_recycle=n_recycle)
    st = tap.store
    assert set(st) == {"z0", "s0", "z", "s"}, "trunk tap did not fire"
    tap.store = {}  # the returned dict is the only owner of the big tensors from here on
    return {**st, "v0": v0_inputs(noisy, lvl), "v0c": v0_inputs(noisy, lvl, clean=True)}


# -------------------------------------------------------------------------------------------------------------- probes
def symmetrise(l: torch.Tensor) -> torch.Tensor:
    return 0.5 * (l + l.transpose(1, 2))


class PairHead(nn.Module):
    """LayerNorm -> Linear(384,512) -> GELU -> Linear(512,23), symmetrised."""

    def __init__(self, d_in=D_PAIR, d_hid=D_HID, n_cls=N_CLS):
        super().__init__()
        self.norm, self.fc1, self.fc2 = nn.LayerNorm(d_in), nn.Linear(d_in, d_hid), nn.Linear(d_hid, n_cls)

    def forward(self, x):
        return symmetrise(self.fc2(F.gelu(self.fc1(self.norm(x)))).float())


class V0Features(nn.Module):
    def __init__(self, n_levels: int):
        super().__init__()
        self.ed, self.es, self.ec, self.el = nn.Embedding(64, 192), nn.Embedding(SEP_BINS, 128), nn.Embedding(2, 64), nn.Embedding(n_levels, D_PAIR)

    def forward(self, v0: dict):
        x = torch.cat([self.ed(v0["dbin"]), self.es(v0["sep"]), self.ec(v0["same"])], dim=-1)
        return x + self.el(v0["lvl"])[:, None, None, :]


class ProbeSet(nn.Module):
    """All probe variants (the ONLY trainable parameters of the run)."""

    def __init__(self, variants, n_levels: int):
        super().__init__()
        self.variants = tuple(variants)
        self.heads = nn.ModuleDict({v: PairHead() for v in self.variants})
        self.v0 = V0Features(n_levels) if "V0" in self.variants else None
        self.v0c = V0Features(n_levels) if "V5" in self.variants else None
        mk = lambda: nn.Sequential(nn.LayerNorm(D_SINGLE), nn.Linear(D_SINGLE, D_PAIR))
        self.proj_s = mk() if "V3" in self.variants else None
        self.proj_s0 = mk() if "V4" in self.variants else None

    @staticmethod
    def _outer_sum(z, s, proj):
        p = proj(s)  # [b,n,384]
        zn = F.layer_norm(z, (z.shape[-1],))  # unit-RMS pair part so z0 (RMS ~0.1) is not swamped by the projected single state
        return zn + p[:, :, None, :] + p[:, None, :, :]  # symmetric in (i, j)

    def inputs(self, f: dict, v: str):
        if v == "V0":
            return self.v0(f["v0"])
        if v == "V1":
            return f["z0"]
        if v == "V2":
            return f["z"]
        if v == "V3":
            return self._outer_sum(f["z"], f["s"], self.proj_s)
        if v == "V4":
            return self._outer_sum(f["z0"], f["s0"], self.proj_s0)
        if v == "V5":
            return self.v0c(f["v0c"])
        raise KeyError(v)

    def forward(self, f: dict) -> dict:
        return {v: self.heads[v](self.inputs(f, v)) for v in self.variants}  # {variant: [b,n,n,23] symmetric float32}


# --------------------------------------------------------------------------------------------------- loss, class weights
def class_weights(kind: str, counts: torch.Tensor | None = None) -> torch.Tensor:
    if kind == "triadj":
        w = torch.ones(N_CLS)
        w[NO_INT] = 0.2
        return w
    if kind == "freq":
        c = counts.double()
        w = (c + 1000.0) ** -0.5
        return (w / ((c / c.sum()) * w).sum()).float()
    raise ValueError(kind)


def loss_normaliser(m: torch.Tensor) -> float:
    """Divisor of the summed weighted CE of a batch = the (UNWEIGHTED) number of observed pairs. It must not depend on the targets:
    then the minimiser of E[w_y * CE] is the tilted posterior p(c|x) * w_c / Z(x), and subtracting log w at evaluation is exact.
    (Dividing by sum w_y, as F.cross_entropy(weight=...) does, cancels the weights on e.g. a one-pair batch.)"""
    return float(m.sum())


def weighted_ce_sum(lg: torch.Tensor, yy: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (F.cross_entropy(lg.float(), yy, reduction="none") * w[yy]).sum()


# ------------------------------------------------------------------------------------------------------ histogram metrics
def _bin(x: torch.Tensor, nb: int, rng: float) -> torch.Tensor:
    return ((x.clamp(-rng, rng - 1e-3) + rng) * (nb / (2 * rng))).long().clamp_(0, nb - 1)


def _log1mexp(x: torch.Tensor) -> torch.Tensor:  # log(1 - exp(x)), x < 0
    x = x.clamp(max=-1e-7)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def auc_from_hist(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """ROC AUC from per-bin counts (bins ascending in score; ties inside a bin count 1/2). [..., B] -> [...]"""
    P, N = pos.sum(-1), neg.sum(-1)
    below = neg.cumsum(-1) - neg
    num = (pos * (below + 0.5 * neg)).sum(-1)
    return torch.where((P > 0) & (N > 0), num / (P * N).clamp_min(1), torch.full_like(num, float("nan")))


def ap_from_hist(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Average precision (sklearn definition, one threshold per bin) from per-bin counts. [..., B] -> [...]"""
    pr, nr = pos.flip(-1), neg.flip(-1)
    tp, fp = pr.cumsum(-1), nr.cumsum(-1)
    prec = tp / (tp + fp).clamp_min(1)
    P = pos.sum(-1)
    ap = (pr * prec).sum(-1) / P.clamp_min(1)
    return torch.where(P > 0, ap, torch.full_like(ap, float("nan")))


def pair_scores(lg: torch.Tensor):
    """lg [N,23] logits with the class-weight tilt already removed -> (logit P(any) [N], logit P(class c) one-vs-rest [N,22])."""
    lse = torch.logsumexp(lg, -1, keepdim=True)
    s_any = torch.logsumexp(lg[:, :N_INT], -1) - lg[:, NO_INT]
    lc = lg[:, :N_INT]
    s_cls = lc - (lse + _log1mexp(lc - lse))  # logit(p_c) = l_c - log sum_{k != c} exp(l_k)
    return s_any, s_cls


def batch_stats(lg: torch.Tensor, ex: torch.Tensor, yy: torch.Tensor, n_ex: int, L_ex: torch.Tensor, hard: torch.Tensor | None = None) -> dict:
    """Per-example sufficient statistics of one variant at one level. lg [N,23] (tilt removed), ex [N] chunk example index,
    yy [N] labels, L_ex [n_ex] = #residues (tokens with >=1 observed pair), hard [N] bool = hard stratum (None: all)."""
    dev = lg.device
    s_any, s_cls = pair_scores(lg)
    pos = yy < NO_INT
    ia = ex * 2 * NB_ANY + (~pos).long() * NB_ANY + _bin(s_any, NB_ANY, RNG_ANY)
    h_any = torch.bincount(ia, minlength=n_ex * 2 * NB_ANY)
    h_hard = h_any if hard is None else torch.bincount(ia[hard], minlength=n_ex * 2 * NB_ANY)
    cidx = torch.arange(N_INT, device=dev)
    is_c = yy[:, None] == cidx[None]
    idx = ((ex[:, None] * N_INT + cidx[None]) * 2 + (~is_c).long()) * NB_CLS + _bin(s_cls, NB_CLS, RNG_CLS)
    h_cls = torch.bincount(idx.reshape(-1), minlength=n_ex * N_INT * 2 * NB_CLS)
    p_any = torch.sigmoid(s_any)
    eb = (p_any * N_ECE).long().clamp_(0, N_ECE - 1)
    base = ex * N_ECE + eb
    sz = n_ex * N_ECE
    ece = torch.stack([torch.bincount(base, minlength=sz).float(), torch.bincount(base, weights=p_any.float(), minlength=sz).float(),
                       torch.bincount(base, weights=pos.float(), minlength=sz).float()], -1).reshape(n_ex, N_ECE, 3)
    topl = torch.zeros(n_ex)  # zero-positive examples score 0 (their top-L list contains no positive)
    haspos = torch.zeros(n_ex, dtype=torch.bool)
    for e in range(n_ex):
        sel = ex == e
        pe = pos[sel]
        if bool(pe.any()):
            k = int(min(int(L_ex[e]), pe.numel()))
            topl[e] = float(pe[s_any[sel].topk(k).indices].float().mean())
            haspos[e] = True
    return {"any": h_any.reshape(n_ex, 2, NB_ANY).int().cpu(), "any_hard": h_hard.reshape(n_ex, 2, NB_ANY).int().cpu(), "cls": h_cls.reshape(n_ex, N_INT, 2, NB_CLS).int().cpu(),
            "ece": ece.cpu(), "topl": topl, "haspos": haspos}


class EvalStore:
    """Per (level, variant) per-example statistics + per-example labels/units, accumulated chunk by chunk."""

    def __init__(self, variants, n_levels):
        self.variants, self.L = tuple(variants), n_levels
        self.buf = {(l, v): {"any": [], "any_hard": [], "cls": [], "ece": [], "topl": [], "haspos": []} for l in range(n_levels) for v in variants}
        self.ids, self.unit_keys, self.cls_counts = [], [], []
        self.n_requested = self.n_failed = self.n_no_obs = self.n_dup = 0  # load accounting (see load_report)
        self.drops: dict = {}
        self.clusters_given = False

    def note_loaded(self, meta: dict, n_loaded_ok: int):
        self.n_requested += int(meta["n_requested"]) if meta.get("n_requested") is not None else n_loaded_ok
        self.n_failed += int(meta["n_failed"] or 0)
        for k, v in (meta.get("drops") or {}).items():
            self.drops[k] = self.drops.get(k, 0) + int(v)

    def load_report(self) -> dict:
        return {"requested": self.n_requested, "evaluated": self.n_examples, "failed_to_load": self.n_failed,
                "no_observed_pairs": self.n_no_obs, "duplicates_skipped": self.n_dup, "drop_reasons": dict(sorted(self.drops.items()))}

    @property
    def n_examples(self):
        return len(self.ids)

    def add_examples(self, ids, units, counts):
        self.ids += list(ids)
        self.unit_keys += list(units)
        self.cls_counts.append(counts.cpu())

    def add(self, l, v, st):
        for k, x in st.items():
            self.buf[(l, v)][k].append(x)

    def finalize(self):
        self.cat = {key: {k: torch.cat(x, 0) for k, x in d.items()} for key, d in self.buf.items()}
        self.counts = torch.cat(self.cls_counts, 0).double()  # [E,23]
        uniq = {u: i for i, u in enumerate(dict.fromkeys(self.unit_keys))}
        self.unit_idx = torch.tensor([uniq[u] for u in self.unit_keys], dtype=torch.long)
        self.n_units = len(uniq)
        self.n_with_pos = int(next(iter(self.cat.values()))["haspos"].sum())
        return self


def summarise(store: EvalStore, n_boot: int = 1000, seed: int = 0, device="cpu") -> dict:
    """Pooled point estimates + bootstrap (resampling units = clusters if provided else examples) of every metric.
    out[(l, v)] -> dict(auc, ap, macro_ap, topl: np arrays [1 + n_boot] (row 0 = point estimate); per_class_ap [22]; ece float)."""
    store.finalize()
    U, uidx = store.n_units, store.unit_idx
    g = torch.Generator().manual_seed(seed)
    W = torch.ones(1, U, dtype=torch.float64)
    if n_boot > 0:
        draw = torch.randint(0, U, (n_boot, U), generator=g)
        Wb = torch.zeros(n_boot, U, dtype=torch.float64)
        Wb.scatter_add_(1, draw, torch.ones_like(draw, dtype=torch.float64))
        W = torch.cat([W, Wb], 0)
    W = W.to(device)
    cls_total = store.counts.sum(0)
    cls_ok = [c for c in range(N_INT) if float(cls_total[c]) >= MIN_POS_CLASS]

    def unit_sum(x):
        out = torch.zeros((U,) + tuple(x.shape[1:]), dtype=torch.float64)
        return out.index_add_(0, uidx, x.double())

    res = {}
    for (l, v), d in store.cat.items():
        A = (W @ unit_sum(d["any"]).reshape(U, -1).to(device)).reshape(W.shape[0], 2, NB_ANY)
        Ahard = (W @ unit_sum(d["any_hard"]).reshape(U, -1).to(device)).reshape(W.shape[0], 2, NB_ANY)
        C = (W @ unit_sum(d["cls"]).reshape(U, -1).to(device)).reshape(W.shape[0], N_INT, 2, NB_CLS)
        apc = ap_from_hist(C[:, :, 0], C[:, :, 1])  # [1+R, 22]
        macro = torch.nanmean(apc[:, cls_ok], dim=1) if cls_ok else torch.full((W.shape[0],), float("nan"), dtype=torch.float64, device=device)
        valid = d["haspos"].bool()
        tl = d["topl"]
        topl = (W @ unit_sum(tl).to(device)) / (W @ unit_sum(torch.ones_like(tl)).to(device)).clamp_min(1e-12)  # ALL examples (no-positive = 0)
        topl_pos = (W @ unit_sum(torch.where(valid, tl, torch.zeros_like(tl))).to(device)) / (W @ unit_sum(valid.double()).to(device)).clamp_min(1e-12)
        ece = d["ece"].double().sum(0)  # [15,3]
        ece_v = float((ece[:, 1] - ece[:, 2]).abs().sum() / ece[:, 0].sum().clamp_min(1))
        res[(l, v)] = dict(auc=auc_from_hist(A[:, 0], A[:, 1]).cpu().numpy(), ap=ap_from_hist(A[:, 0], A[:, 1]).cpu().numpy(),
                           macro_ap=macro.cpu().numpy(), topl=topl.cpu().numpy(), topl_pos=topl_pos.cpu().numpy(),
                           auc_hard=auc_from_hist(Ahard[:, 0], Ahard[:, 1]).cpu().numpy(), ap_hard=ap_from_hist(Ahard[:, 0], Ahard[:, 1]).cpu().numpy(), per_class_ap=apc[0].cpu().numpy(), ece=ece_v)
    res["_meta"] = dict(cls_ok=cls_ok, class_counts=cls_total.tolist(), n_examples=store.n_examples, n_units=U, n_boot=n_boot,
                        n_pairs=float(cls_total.sum()), n_examples_with_pos=store.n_with_pos)
    return res


def _ci(x: np.ndarray):
    if len(x) < 3:
        return None
    lo, hi = np.nanpercentile(x[1:], [2.5, 97.5])
    return [float(lo), float(hi)]


def _f(x):
    x = float(x)
    return None if math.isnan(x) else x


def build_result(summ: dict, variants, levels) -> dict:
    """JSON-ready dict: per-level per-variant metrics (+CI), gains with bootstrap CIs."""
    meta = summ["_meta"]
    out = {"meta": meta, "levels": [], "gains": []}
    for l, (tca, tlat) in enumerate(levels):
        row = {"level": l, "t_ca": tca, "t_lat": tlat}
        for v in variants:
            r = summ[(l, v)]
            row[v] = {m: {"value": _f(r[m][0]), "ci95": _ci(r[m])} for m in GAIN_METRICS}
            row[v]["ece"] = r["ece"]
            row[v]["per_class_ap"] = [_f(x) for x in r["per_class_ap"]]
        out["levels"].append(row)
        for a, b in GAIN_PAIRS:
            if a in variants and b in variants:
                for m in GAIN_METRICS:
                    d = summ[(l, a)][m] - summ[(l, b)][m]
                    out["gains"].append({"level": l, "t_ca": tca, "a": a, "b": b, "metric": m, "delta": _f(d[0]), "ci95": _ci(d)})
    return out


def gain_of(res: dict, level: int, a: str, b: str, metric: str):
    for g in res["gains"]:
        if g["level"] == level and g["a"] == a and g["b"] == b and g["metric"] == metric:
            return g
    return None


def decide(res: dict) -> dict:
    """Pre-registered P0 rule, evaluated on the R=3 held-out results (see RULE).
    positive control : at the cleanest level V0 AUC >= pc_min_auc AND V2 AUC >= V0 AUC - pc_tol (else the run is uninterpretable)
    P0_PASS          : at t_ca ~ 0.5, V2 beats V0 by >= min_gain_auc (AUC) AND >= min_gain_ap (AP), both 95% CI lower bounds > 0
    beyond_inputs    : at the same level, V2 beats V4 by >= min_gain_beyond_inputs AUC with CI lower bound > 0
    high_noise_signal: at the noisiest level V2 - V0 AUC CI lower bound > 0 (trunk adds something even at t_ca = 0.2).
    Reported, not part of P0_PASS: hard_gain_auc_ok (V2 - V0 on the hard stratum, same threshold), beyond_geometry (V2 - V5 AUC on the
    hard stratum >= 0.01 with CI lower bound > 0: info beyond CA geometry, even versus oracle clean distances)."""
    lv = res["levels"]
    pivot = int(np.argmin([abs(r["t_ca"] - RULE["pivot_t_ca"]) for r in lv]))
    clean = int(np.argmax([r["t_ca"] for r in lv]))
    noisy = int(np.argmin([r["t_ca"] for r in lv]))
    g = lambda lev, a, b, m: gain_of(res, lev, a, b, m)
    ok = lambda gg, thr: bool(gg and gg["delta"] is not None and gg["ci95"] and gg["delta"] >= thr and gg["ci95"][0] > 0)
    v0c, v2c = lv[clean]["V0"]["auc"]["value"], lv[clean]["V2"]["auc"]["value"]
    pc = bool(v0c is not None and v2c is not None and v0c >= RULE["pc_min_auc"] and v2c >= v0c - RULE["pc_tol"])
    g_auc, g_ap, g_bi, g_hn = g(pivot, "V2", "V0", "auc"), g(pivot, "V2", "V0", "ap"), g(pivot, "V2", "V4", "auc"), g(noisy, "V2", "V0", "auc")
    d_auc = ok(g_auc, RULE["min_gain_auc"])
    d_ap = ok(g_ap, RULE["min_gain_ap"])
    return {"rule": RULE, "pivot_level": pivot, "clean_level": clean, "noisy_level": noisy,
            "positive_control": {"V0_auc_clean": v0c, "V2_auc_clean": v2c, "ok": pc},
            "pivot_gain_auc": g_auc, "pivot_gain_ap": g_ap, "gain_auc_ok": d_auc, "gain_ap_ok": d_ap,
            "P0_PASS": bool(pc and d_auc and d_ap),
            "beyond_inputs": ok(g_bi, RULE["min_gain_beyond_inputs"]), "pivot_V2_minus_V4_auc": g_bi,
            "hard_gain_auc_ok": ok(g(pivot, "V2", "V0", "auc_hard"), RULE["min_gain_auc"]), "pivot_gain_auc_hard": g(pivot, "V2", "V0", "auc_hard"),
            "beyond_geometry": ok(g(pivot, "V2", "V5", "auc_hard"), RULE["min_gain_beyond_inputs"]), "pivot_V2_minus_V5_auc_hard": g(pivot, "V2", "V5", "auc_hard"),
            "high_noise_signal": bool(g_hn and g_hn["ci95"] and g_hn["ci95"][0] > 0), "noisy_V2_minus_V0_auc": g_hn}


# ------------------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, tap, probes, loader, device, levels, levels_t, n_recycle, logw, micro, seed=1234, max_examples=0, amp=False):
    """Every example at ALL levels (common random numbers). Returns an EvalStore (not finalized). Only examples that really loaded
    are evaluated (failed rows arrive as sentinels and are counted, never replaced); duplicate example ids are skipped and counted."""
    probes.eval()
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None  # evaluation reseeds; do not disturb the training stream
    store = EvalStore(probes.variants, len(levels))
    n_seen, bidx = 0, 0
    seen_ids: set = set()
    for batch in loader:
        if max_examples and n_seen >= max_examples:
            break
        full, meta = unpack(batch, device)
        bsz = full["padding_mask"].shape[0] if "padding_mask" in full else 0
        store.note_loaded(meta, bsz)
        if bsz == 0:
            continue
        ids_all = list(meta["example_ids"]) if meta["example_ids"] else [f"ex{n_seen + k}" for k in range(bsz)]
        clu_all = list(meta["clusters"]) if meta["clusters"] is not None else ids_all
        keep = [k for k, e in enumerate(ids_all) if e not in seen_ids]
        if len(keep) < bsz:
            store.n_dup += bsz - len(keep)
            if not keep:
                continue
            full, ids_all, clu_all, bsz = index_batch(full, keep, bsz), [ids_all[k] for k in keep], [clu_all[k] for k in keep], len(keep)
        seen_ids.update(ids_all)
        store.clusters_given = store.clusters_given or meta["clusters"] is not None
        for i in range(0, bsz, micro):
            b = slice_batch(full, i, i + micro, bsz)
            B = b["padding_mask"].shape[0]
            if max_examples and n_seen >= max_examples:
                break
            y, m = label_masks(b)
            if not bool(m.any()):
                store.n_no_obs += B
                continue
            ex_ids, clu = ids_all[i:i + B], clu_all[i:i + B]
            ex, yy = m.nonzero()[:, 0], y[m]
            counts = torch.bincount(ex * N_CLS + yy, minlength=B * N_CLS).reshape(B, N_CLS)
            L_ex = (m | m.transpose(1, 2)).any(-1).sum(-1)
            store.add_examples(ex_ids, [str(c) for c in clu], counts)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                torch.manual_seed(seed + bidx)
                clean = clean_batch(model, b)
                hrd = hard_mask(b, clean["x_1"]["bb_ca"])[m]
                for l in range(len(levels)):
                    noisy = noisy_batch(model, clean, torch.full((B,), l, device=device, dtype=torch.long), levels_t, seed=seed + bidx + 1)
                    feats = extract_features(model, tap, noisy, torch.full((B,), l, device=device, dtype=torch.long), n_recycle)
                    logits = probes(feats)
                    for v, lv in logits.items():
                        store.add(l, v, batch_stats(lv[m].float() - logw, ex, yy, B, L_ex, hrd))
            n_seen += B
            bidx += 1
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state_all(rng_cuda)
    return store


# ------------------------------------------------------------------------------------------------------------- training
def lr_at(args, step: int, frac_time: float) -> float:
    prog = min(1.0, max(step / max(1, args.max_steps), frac_time))
    warm = min(1.0, (step + 1) / max(1, args.warmup))
    return args.lr * warm * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))


def train_step_once(model, tap, probes, opt, b: dict, args, levels_t, n_levels, w, gen, micro: int, device, amp):
    """One optimiser-step worth of gradients. Each example is corrupted at ``args.reps`` independent random levels (reps > 1 reuses an
    expensive-to-load example for several noise levels; the AE sample is shared)."""
    y, m = label_masks(b)
    n_obs = loss_normaliser(m)
    if n_obs == 0.0:
        return None
    bsz = b["padding_mask"].shape[0]
    reps = max(1, args.reps)
    lvl_all = torch.randint(0, n_levels, (reps, bsz), generator=gen).to(device)
    losses = {v: 0.0 for v in probes.variants}
    for i in range(0, bsz, micro):
        c = slice_batch(b, i, i + micro, bsz)
        cm, cy = m[i:i + micro], y[i:i + micro]
        if not bool(cm.any()):
            continue
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            clean = clean_batch(model, c)
        for r in range(reps):
            lvl = lvl_all[r, i:i + micro]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                noisy = noisy_batch(model, clean, lvl, levels_t)
                feats = extract_features(model, tap, noisy, lvl, args.n_recycle)
            for v in probes.variants:  # variants are independent: backward one at a time so only ONE head's activations are alive
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    lv = probes.heads[v](probes.inputs(feats, v))
                lo = weighted_ce_sum(lv[cm], cy[cm], w) / (n_obs * reps)
                lo.backward()
                losses[v] += float(lo.detach())
                del lv, lo
            del feats, noisy
        del clean
    return losses


def save_heads(path, probes, opt, step, args, levels, w):
    tmp = path + ".tmp"
    torch.save({"probes": probes.state_dict(), "opt": opt.state_dict() if opt is not None else None, "step": step,
                "levels": list(levels), "class_weights": w.tolist(), "variants": list(probes.variants), "args": vars(args)}, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------- data (ADAPTER stub)
def _load_factory(spec: str):
    path, func = spec.rsplit(":", 1)
    s = importlib.util.spec_from_file_location("p0_loader_factory", path)
    mod = importlib.util.module_from_spec(s)
    sys.modules["p0_loader_factory"] = mod
    s.loader.exec_module(mod)
    return getattr(mod, func)


try:  # Builder A's dataset (needs rfproteina at import: only available inside the SIF)
    from triadj_ext.ring_data import RingDataset as _RingDataset

    class P0RingDataset(_RingDataset):
        """RingDataset + the manifest cluster id of the row actually used (needed for the cluster bootstrap and the disjointness check).

        Interface assumed from A's ``ring_data.RingDataset`` (present as of 2026-10-21 16:3x, checked by a test): constructor kwarg
        ``substitute_on_failure`` (True = train: a failing row is replaced by a deterministic other row; False = STRICT, used for
        validation: the row is never replaced and ``__getitem__`` returns ``{'failed': True, 'index', 'example_id', 'reason',
        'drop_counts_snapshot'}``), static ``split_failed(batch) -> (valid, failed)``, static ``collate(valid)``, ``counts()`` =
        run-wide counters shared across workers (requested/ok/substituted/failed + one per skip reason), ``build_sample(i, seed)``,
        ``row(i)``."""

        def build_sample(self, i: int, seed: int | None = None) -> dict:
            sample = super().build_sample(i, seed)
            sample["cluster"] = self.row(i)["cluster"] or sample["example_id"]
            return sample
except Exception as _e:  # pragma: no cover
    _RingDataset, P0RingDataset, _RING_IMPORT_ERROR = None, None, repr(_e)


def p0_collate(batch: list) -> dict:
    """RFD4 collate of the samples that loaded -> plain dict of tensors + example_id_batch / cluster_id_batch lists + load accounting
    (n_requested_batch, n_failed_batch, drop_counts_batch = reasons of the strict-mode sentinels in this batch). Failed sentinels are
    removed and counted, never replaced; an all-failed batch carries no tensors. Module level: picklable for spawn workers."""
    good, failed = _RingDataset.split_failed(batch)
    drops: dict = {}
    for x in failed:
        r = str(x.get("reason", "unknown"))
        drops[r] = drops.get(r, 0) + 1
    d = {"n_requested_batch": len(batch), "n_failed_batch": len(failed), "drop_counts_batch": drops}
    if good:
        out = _RingDataset.collate(good)
        d.update({k: v for k, v in out.items() if torch.is_tensor(v)})
        d["example_id_batch"] = [str(x["example_id"]) for x in good]
        d["cluster_id_batch"] = [str(x.get("cluster") or x["example_id"]) for x in good]
    return d


def loader_counts(loader) -> dict:
    """Run-wide dataset counters (shared across DataLoader workers) of a loader built by ``build_loaders``; {} for other loaders."""
    ds = getattr(loader, "dataset", None)
    try:
        return dict(ds.counts()) if hasattr(ds, "counts") else {}
    except Exception:
        return {}


def counts_delta(c1: dict, c0: dict) -> dict:
    return {k: v - c0.get(k, 0) for k, v in c1.items() if v - c0.get(k, 0) != 0}


def _dedupe_then_subset(n: int, seed: int):
    """row_filter: drop duplicate manifest rows (same pdb/assembly/pn_unit(s)), then a random subset of n (fixed seed)."""

    def f(df):
        cols = [c for c in ("pdb_id", "assembly_id", "pn_unit_iid", "pn_unit_1_iid", "pn_unit_2_iid") if c in df]
        idx = np.flatnonzero(~df.duplicated(subset=cols).to_numpy())
        if 0 < n < len(idx):
            idx = np.random.default_rng(seed).choice(idx, n, replace=False)
        keep = np.zeros(len(df), dtype=bool)
        keep[idx] = True
        return keep

    return f


def build_loaders(args):
    """Wired to Builder A's ``triadj_ext.ring_data.RingDataset`` (PDB rows with RING labels; RFD4 pipeline with ALL conditioning off).
    Returns ``(train_loader, val_loader, info)``. Val = random ``--n-val`` de-duplicated rows of the val manifest (fixed seed,
    deterministic crops, no shuffling) read in STRICT mode (a failing row is reported, never replaced by another row); train = the
    train manifest MINUS every row whose cluster occurs in the val subset, so train and val share no cluster by construction
    (``train_val_cluster_overlap == 0`` is re-verified); train rows that fail are substituted by A's retry logic and their drop
    reasons are aggregated. ``--loader-factory path.py:func`` replaces all of this (tests, GPU smoke on tests/data)."""
    if args.loader_factory:
        return _load_factory(args.loader_factory)(args)
    if P0RingDataset is None:
        raise SystemExit(f"triadj_ext.ring_data not importable: {_RING_IMPORT_ERROR}")
    import lmdb  # noqa: F401  (RingStore needs it inside the workers; not in the SIF: put a pip --target dir on PYTHONPATH)
    from triadj_ext import ring_data as rd

    mdir = rd.default_path("manifest_dir")
    kw = dict(rows=args.rows, crop_size=args.crop_size, seed=args.seed)
    val_ds = P0RingDataset(os.path.join(mdir, args.val_manifest), row_filter=_dedupe_then_subset(args.n_val, args.seed),
                           substitute_on_failure=False, **kw)
    vclu = {c for c in val_ds._table()["cluster"].tolist() if c}
    train_ds = P0RingDataset(os.path.join(mdir, args.train_manifest), row_filter=lambda df: ~df["cluster"].fillna("").isin(vclu),
                             substitute_on_failure=True, **kw)
    tclu = {c for c in train_ds._table()["cluster"].tolist() if c}
    info = {"train_manifest": args.train_manifest, "val_manifest": args.val_manifest, "rows": args.rows, "crop_size": args.crop_size,
            "train_examples": len(train_ds), "val_examples": len(val_ds), "train_clusters": len(tclu), "val_clusters": len(vclu),
            "train_val_cluster_overlap": len(tclu & vclu)}
    assert info["train_val_cluster_overlap"] == 0
    dl = dict(collate_fn=p0_collate, drop_last=False, num_workers=args.workers, pin_memory=False)
    if args.workers > 0:
        dl.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl)
    vdl = dict(dl, num_workers=min(4, args.workers))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=max(1, args.micro_batch), shuffle=False, **vdl)
    return train_loader, val_loader, info


def infinite(loader):
    while True:
        n = 0
        for b in loader:
            n += 1
            yield b
        if n == 0:
            raise RuntimeError("empty training loader")


# ------------------------------------------------------------------------------------------------------------------ main
def parse_levels(s: str):
    return tuple(tuple(float(x) for x in p.split(":")) for p in s.split(","))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=f"/tmp/{os.environ.get('USER', 'u')}/p0_probe")
    p.add_argument("--bundle", default="/w/rfd4_weights")
    p.add_argument("--run-name", default="p0_probe")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--levels", default=",".join(f"{a}:{b}" for a, b in DEFAULT_LEVELS), help="t_ca:t_lat,... (t=0 noise, 1 data)")
    p.add_argument("--n-recycle", type=int, default=3, help="trunk passes at train time and the main eval")
    p.add_argument("--eval-recycles", default="3,6")
    p.add_argument("--eval6-max-examples", type=int, default=300, help="cap on val examples for recycle depths other than --n-recycle (0 = all)")
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=64, help="loaded examples per optimiser step (x --reps noise levels each)")
    p.add_argument("--reps", type=int, default=2, help="independent noise levels per loaded example per step (data-bound loaders)")
    p.add_argument("--micro-batch", type=int, default=16, help="trunk micro-batch (grad accumulation inside a step); halves itself on CUDA OOM")
    p.add_argument("--autotune-batch", action="store_true", help="double the micro-batch on a real batch until the measured peak reaches --autotune-target-gib (or OOM)")
    p.add_argument("--autotune-target-gib", type=float, default=60.0)
    p.add_argument("--budget-gib", type=float, default=68.0, help="print a warning when the measured peak exceeds this")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--class-weights", choices=("triadj", "freq"), default="triadj")
    p.add_argument("--freq-batches", type=int, default=30)
    p.add_argument("--time-limit", type=float, default=0.0, help="seconds for the WHOLE script (0 = none)")
    p.add_argument("--eval-reserve", type=float, default=2400.0, help="seconds kept for final evals when --time-limit is set")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--log-early-every", type=int, default=10, help="telemetry cadence for the first --log-early-steps steps")
    p.add_argument("--log-early-steps", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=300)
    p.add_argument("--mid-val-every", type=int, default=400, help="quick held-out check every k steps (0 = off)")
    p.add_argument("--mid-val-n", type=int, default=48)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rows", choices=("monomer", "interface", "all"), default="all")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--n-val", type=int, default=1000, help="random held-out rows of the val manifest")
    p.add_argument("--train-manifest", default="pdb_afdb_train_clustersplit.parquet")
    p.add_argument("--val-manifest", default="pdb_afdb_val_clustersplit.parquet")
    p.add_argument("--init-heads", default="", help="heads.pt to start from")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--allow-conditioning", action="store_true", help="downgrade the conditioning-off check to a warning")
    p.add_argument("--loader-factory", default="", help="path.py:func(args) -> (train_loader, val_loader, info)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args(argv)


def resolve_class_weights(args, ck: dict | None, count_fn) -> tuple[torch.Tensor, str]:
    """Class weights used for training AND for removing the tilt at evaluation. With --init-heads the checkpoint's weights (and
    scheme) win: the saved heads were trained against exactly those, so nothing is re-estimated and --class-weights is overridden."""
    if ck is not None and ck.get("class_weights") is not None:
        saved = (ck.get("args") or {}).get("class_weights", args.class_weights)
        if saved != args.class_weights:
            log(kv("P0 warn", class_weights_overridden_by_checkpoint=saved, requested=args.class_weights))
        return torch.tensor(ck["class_weights"], dtype=torch.float32), saved
    if args.class_weights == "freq":
        return class_weights("freq", count_fn()), "freq"
    return class_weights("triadj"), "triadj"


def gpu_util_pct() -> float:
    try:
        return float(torch.cuda.utilization())  # NVML (needs pynvml); a proxy only
    except Exception:
        return float("nan")


def autotune_search(try_fn, bsz: int, start: int, target_gib: float, budget_gib: float, oom_exc=torch.cuda.OutOfMemoryError):
    """Pure search logic (unit-tested on CPU). ``try_fn(micro) -> peak_gib`` (raises ``oom_exc`` on OOM). Doubles ``micro`` from ``start``
    until the peak reaches ``target_gib``, ``micro >= bsz`` or OOM; returns (largest micro with peak <= budget_gib, report). If even
    ``start`` does not fit it halves downwards."""
    micro, best, report = max(1, min(start, bsz)), None, []
    while True:
        try:
            peak = float(try_fn(micro))
        except oom_exc:
            report.append({"micro": micro, "peak_gib": None, "oom": True})
            if best is None and micro > 1:
                micro //= 2
                continue
            break
        report.append({"micro": micro, "peak_gib": round(peak, 2), "oom": False})
        if peak <= budget_gib:
            best = micro
        if peak >= target_gib or micro >= bsz:
            break
        micro = min(bsz, micro * 2)
    return best or 1, report


def autotune_micro(model, tap, probes, opt, b: dict, args, levels_t, n_levels, w, device, amp, target_gib: float, budget_gib: float):
    """Autotune the trunk micro-batch on a REAL training batch with the measured peak (no optimiser step is taken)."""
    gen = torch.Generator().manual_seed(4242)

    def try_fn(micro):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            opt.zero_grad(set_to_none=True)
            train_step_once(model, tap, probes, opt, b, args, levels_t, n_levels, w, gen, micro, device, amp)
            return torch.cuda.max_memory_allocated() / 2**30
        finally:
            opt.zero_grad(set_to_none=True)

    return autotune_search(try_fn, b["padding_mask"].shape[0], args.micro_batch, target_gib, budget_gib)


def maybe_wandb(args):
    """Optional: only when WANDB_MODE != disabled; offline by default (compute nodes have no internet)."""
    if args.no_wandb or os.environ.get("WANDB_MODE", "") == "disabled":
        return None
    try:
        import wandb

        return wandb.init(project="TRIAD-R", name=args.run_name, config=vars(args), mode=os.environ.get("WANDB_MODE") or "offline",
                          dir=args.out)
    except Exception as e:  # never let logging kill the run
        log(f"P0 warn wandb_unavailable={type(e).__name__}")
        return None


def _wlog(wb, d, step):
    if wb is not None:
        try:
            wb.log(d, step=step)
        except Exception:
            pass


def main(argv=None):
    args = parse_args(argv)
    if args.eval_only and not args.init_heads:
        raise SystemExit("P0 FATAL --eval-only needs --init-heads")
    t_start = time.time()
    device = torch.device(args.device)
    amp = device.type == "cuda"
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    levels = parse_levels(args.levels)
    levels_t = (torch.tensor([a for a, _ in levels], dtype=torch.float32, device=device),
                torch.tensor([b for _, b in levels], dtype=torch.float32, device=device))
    variants = tuple(args.variants.split(","))
    wb = maybe_wandb(args)

    model = load_frozen_trunk(args.bundle, device)
    tap = FeatureTap(model.nn).install()
    probes = ProbeSet(variants, len(levels)).to(device)
    n_probe = sum(p.numel() for p in probes.parameters())
    n_trunk = sum(p.numel() for p in model.parameters())
    assert not any(p.requires_grad for p in model.parameters())
    log(kv("P0 init", device=str(device), probe_params=n_probe, frozen_params=n_trunk, variants=",".join(variants), levels=args.levels,
           load_s=time.time() - t_start))

    ck = None
    if args.init_heads:  # load FIRST: its class weights / levels / variants define what the heads were trained against
        ck = torch.load(args.init_heads, map_location=device, weights_only=False)
        if [tuple(x) for x in ck["levels"]] != [tuple(x) for x in levels] or list(ck["variants"]) != list(variants):
            raise SystemExit(f"P0 FATAL --init-heads was trained with levels={ck['levels']} variants={ck['variants']}")
        probes.load_state_dict(ck["probes"])

    train_loader, val_loader, info = build_loaders(args)
    info = dict(info or {})
    log(kv("P0 split", **{k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}))
    train_iter = infinite(train_loader) if not args.eval_only else iter(())
    train_clusters: set = set()

    # first-batch sanity: leak guards + label statistics (skip all-failed batches)
    first = None
    for raw in val_loader:
        first, _ = unpack(raw, device)
        if "padding_mask" in first:
            break
    if first is None or "padding_mask" not in first:
        raise SystemExit("P0 FATAL: no validation example could be loaded")
    bad = conditioning_problems(first)
    if bad:
        msg = f"conditioning is ON in the probe batches (leaks label-like information into s): {bad}"
        if not args.allow_conditioning:
            raise SystemExit("P0 FATAL " + msg)
        log("P0 warn " + msg)
    y0, m0 = label_masks(first)
    c0 = torch.bincount(y0[m0], minlength=N_CLS)
    log(kv("P0 data", n=first["padding_mask"].shape[1], b=first["padding_mask"].shape[0], obs_pairs=int(m0.sum()), pos_frac=float(c0[:NO_INT].sum() / c0.sum().clamp_min(1))))
    make_probe_input({k: v for k, v in first.items()})  # runs the all-MASK assertion on a real batch
    del first

    def _count_train():
        cnt = torch.zeros(N_CLS, dtype=torch.long)
        n = 0
        while n < args.freq_batches:
            bb, _ = unpack(next(train_iter), device)
            if "padding_mask" not in bb:
                continue
            yy, mm = label_masks(bb)
            cnt += torch.bincount(yy[mm], minlength=N_CLS).cpu()
            n += 1
        return cnt

    w, scheme = resolve_class_weights(args, ck, _count_train)
    args.class_weights = scheme
    w = w.to(device)
    logw = torch.log(w)
    log(kv("P0 weights", scheme=scheme, from_checkpoint=bool(ck), w_int_mean=float(w[:N_INT].mean()), w_noint=float(w[NO_INT]), loss_norm="n_observed_pairs"))

    opt = torch.optim.AdamW(probes.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
    step0 = 0
    if ck is not None:
        if ck.get("opt") and not args.eval_only:
            opt.load_state_dict(ck["opt"])
        step0 = int(ck.get("step", 0))
        log(kv("P0 resume", step=step0))

    # ------------------------------------------------------------------ train
    curve = {"step": [], **{f"loss_{v}": [] for v in variants}}
    gen = torch.Generator().manual_seed(args.seed + 17)
    micro = max(1, min(args.micro_batch, args.batch_size))
    autotune_report = None
    if args.autotune_batch and device.type == "cuda" and not args.eval_only:
        raw0 = None
        while raw0 is None:
            r_ = next(train_iter)
            b0, _ = unpack(r_, device)
            raw0 = r_ if "padding_mask" in b0 else None
        micro, autotune_report = autotune_micro(model, tap, probes, opt, b0, args, levels_t, len(levels), w, device, amp, args.autotune_target_gib, args.budget_gib)
        log(kv("P0 autotune", micro=micro, batch=b0["padding_mask"].shape[0], reps=args.reps, probes=json.dumps(autotune_report).replace(" ", "")))
        del b0
        train_iter = itertools.chain([raw0], train_iter)  # the probe batch is still trained on
    train_deadline = t_start + args.time_limit - args.eval_reserve if args.time_limit > 0 else float("inf")
    t_train0, step, acc, n_skip = time.time(), step0, {v: 0.0 for v in variants}, 0
    n_log, wait_s, comp_s, ok_samples = 0, 0.0, 0.0, 0
    win = dict(t=t_train0, step=step0, wait=0.0, comp=0.0, samples=0)
    tr_load = {"requested": 0, "failed_to_load": 0, "drop_reasons": {}}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stop_reason = "max_steps"
    while not args.eval_only and step < args.max_steps:
        if time.time() > train_deadline:
            stop_reason = "time_limit"
            break
        frac_t = (time.time() - t_train0) / max(1.0, train_deadline - t_train0) if args.time_limit > 0 else 0.0
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(args, step, frac_t)
        t_w = time.time()
        raw = next(train_iter)
        wait_s += time.time() - t_w
        b, tmeta = unpack(raw, device)
        n_ok = b["padding_mask"].shape[0] if "padding_mask" in b else 0
        tr_load["requested"] += int(tmeta["n_requested"]) if tmeta["n_requested"] is not None else n_ok
        tr_load["failed_to_load"] += int(tmeta["n_failed"] or 0)
        for k_, v_ in (tmeta["drops"] or {}).items():
            tr_load["drop_reasons"][k_] = tr_load["drop_reasons"].get(k_, 0) + int(v_)
        if n_ok == 0:
            n_skip += 1
            continue
        ok_samples += n_ok
        if tmeta["clusters"] is not None:
            train_clusters.update(str(c) for c in tmeta["clusters"])
        if step - step0 < 3:  # guard the TRAIN stream too (the first val batch was checked above)
            bad = conditioning_problems(b)
            if bad and not args.allow_conditioning:
                raise SystemExit(f"P0 FATAL conditioning ON in train batches: {bad}")
        probes.train()
        t_c = time.time()
        while True:
            opt.zero_grad(set_to_none=True)
            try:
                losses = train_step_once(model, tap, probes, opt, b, args, levels_t, len(levels), w, gen, micro, device, amp)
                break
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if micro <= 1:
                    raise
                micro = max(1, micro // 2)
                log(kv("P0 oom", step=step, micro=micro))
        if losses is None:
            n_skip += 1
            continue
        for h in probes.heads.values():
            torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)
        opt.step()
        comp_s += time.time() - t_c
        step += 1
        n_log += 1
        for v in variants:
            acc[v] += losses[v]
        if (step - step0 <= args.log_early_steps and step % args.log_early_every == 0) or step % args.log_every == 0:
            now = time.time()
            el, wel = now - t_train0, max(now - win["t"], 1e-9)
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            row = {f"loss_{v}": acc[v] / n_log for v in variants}
            d_samples = ok_samples - win["samples"]
            log(kv("P0 step", step=step, **row, lr=opt.param_groups[0]["lr"], sps_win=(step - win["step"]) / wel, sps_all=(step - step0) / max(el, 1e-9),
                   samples_s=d_samples / wel, rep_ex_s=d_samples * max(1, args.reps) / wel, data_wait_frac_win=(wait_s - win["wait"]) / wel,
                   data_wait_frac_all=wait_s / max(el, 1e-9), compute_frac_win=(comp_s - win["comp"]) / wel, gpu_util=gpu_util_pct(),
                   peak_gib=peak, micro=micro, bsz=n_ok, n=b["padding_mask"].shape[1], skipped=n_skip, load_failed=tr_load["failed_to_load"], el_s=el))
            win = dict(t=now, step=step, wait=wait_s, comp=comp_s, samples=ok_samples)
            if peak > args.budget_gib:
                log(kv("P0 warn", peak_gib_over_budget=peak))
            curve["step"].append(step)
            for v in variants:
                curve[f"loss_{v}"].append(acc[v] / n_log)
            _wlog(wb, {**row, "peak_gib": peak, "lr": opt.param_groups[0]["lr"], "samples_s": d_samples / wel}, step)
            acc, n_log = {v: 0.0 for v in variants}, 0
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_heads(os.path.join(args.out, "heads.pt"), probes, opt, step, args, levels, w)
        if args.mid_val_every and step % args.mid_val_every == 0:
            st = evaluate(model, tap, probes, val_loader, device, levels, levels_t, args.n_recycle, logw, micro, max_examples=args.mid_val_n, amp=amp)
            sm = summarise(st, n_boot=0, device="cpu")
            row = {f"auc_L{l}_{v}": float(sm[(l, v)]["auc"][0]) for l in range(len(levels)) for v in variants}
            log(kv("P0 val", step=step, n=st.n_examples, **row))
            _wlog(wb, {f"val/{k}": x for k, x in row.items()}, step)
            probes.train()
    train_s = time.time() - t_train0
    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    if not args.eval_only:
        save_heads(os.path.join(args.out, "heads.pt"), probes, opt, step, args, levels, w)
    tr_load["dataset_counts"] = loader_counts(train_loader)  # run-wide, shared across workers (requested/ok/substituted/failed/reasons)
    log(kv("P0 train_done", steps=step, seconds=train_s, stop=stop_reason, peak_gib=peak_gib, micro=micro, samples=ok_samples,
           data_wait_frac=wait_s / max(train_s, 1e-9), load_failed=tr_load["failed_to_load"]))

    # ------------------------------------------------------------------ final held-out evaluation
    out = {"schema": "p0_probe/v1", "run_name": args.run_name, "args": vars(args), "variants": list(variants),
           "levels": [list(x) for x in levels], "class_weights": {"scheme": scheme, "w": w.tolist(), "loss_normaliser": "n_observed_pairs",
                                                                  "from_checkpoint": bool(ck)},
           "split": {**info, "n_val_requested": info.get("val_examples")},
           "train": {"steps": step, "seconds": train_s, "stop": stop_reason, "peak_gib": peak_gib, "micro_batch_final": micro,
                     "skipped_batches": n_skip, "frozen_params": n_trunk, "probe_params": n_probe, "reps": args.reps,
                     "batch_size": args.batch_size, "samples_loaded": ok_samples, "data_wait_frac": wait_s / max(train_s, 1e-9),
                     "autotune": autotune_report, "load": tr_load}, "eval": {}}
    arrays = {"curve_step": np.array(curve["step"]), **{f"curve_{k}": np.array(v) for k, v in curve.items() if k != "step"}}
    for R in [int(r) for r in args.eval_recycles.split(",")]:
        t_e = time.time()
        if args.time_limit > 0 and out["eval"] and t_e - t_start > args.time_limit - 180:
            log(kv("P0 warn", skipped_eval_R=R, reason="time_limit"))
            continue
        cap = 0 if R == args.n_recycle else args.eval6_max_examples
        cnt0 = loader_counts(val_loader)
        st = evaluate(model, tap, probes, val_loader, device, levels, levels_t, R, logw, micro, max_examples=cap, amp=amp)
        sm = summarise(st, n_boot=args.n_boot, seed=args.seed, device=device)
        res = build_result(sm, variants, levels)
        res["decision"] = decide(res) if all(v in variants for v in ("V0", "V2")) else None
        res["load"] = {**st.load_report(), "dataset_counts_delta": counts_delta(loader_counts(val_loader), cnt0)}
        out["eval"][f"R{R}"] = res
        out["split"][f"val_R{R}"] = res["load"]
        if st.clusters_given and train_clusters:  # train/val must not share a cluster (only checkable when the loader supplies cluster ids)
            out["split"][f"train_val_cluster_overlap_R{R}"] = len(train_clusters & set(st.unit_keys))
            out["split"]["train_clusters_seen"] = len(train_clusters)
        for key in ("auc", "ap", "macro_ap", "topl", "topl_pos", "auc_hard", "ap_hard"):
            arrays[f"{key}_R{R}"] = np.array([[float(sm[(l, v)][key][0]) for v in variants] for l in range(len(levels))])
            arrays[f"{key}_boot_R{R}"] = np.stack([np.stack([sm[(l, v)][key][1:] for v in variants]) for l in range(len(levels))]).astype(np.float32) if args.n_boot else np.zeros(0)
        arrays[f"ece_R{R}"] = np.array([[sm[(l, v)]["ece"] for v in variants] for l in range(len(levels))])
        arrays[f"per_class_ap_R{R}"] = np.array([[sm[(l, v)]["per_class_ap"] for v in variants] for l in range(len(levels))])
        log(kv("P0 eval", R=R, seconds=time.time() - t_e, n_examples=st.n_examples, n_units=st.n_units, pairs=sm["_meta"]["n_pairs"],
               n_with_pos=sm["_meta"]["n_examples_with_pos"], requested=st.n_requested, failed=st.n_failed, no_obs=st.n_no_obs, dup=st.n_dup))
        for l, (tca, tlat) in enumerate(levels):
            log(kv(f"P0 res", R=R, level=l, t_ca=tca, t_lat=tlat,
                   **{f"auc_{v}": float(sm[(l, v)]["auc"][0]) for v in variants}, **{f"ap_{v}": float(sm[(l, v)]["ap"][0]) for v in variants},
                   **{f"aucHard_{v}": float(sm[(l, v)]["auc_hard"][0]) for v in variants}))
        if res["decision"]:
            d = res["decision"]
            log(kv("P0 decision", R=R, positive_control=d["positive_control"]["ok"], gain_auc_ok=d["gain_auc_ok"], gain_ap_ok=d["gain_ap_ok"],
                   P0_PASS=d["P0_PASS"], beyond_inputs=d["beyond_inputs"], beyond_geometry=d["beyond_geometry"],
                   hard_gain_auc_ok=d["hard_gain_auc_ok"], high_noise_signal=d["high_noise_signal"]))
            _wlog(wb, {f"decision/R{R}/{k}": float(bool(d[k])) for k in ("P0_PASS", "beyond_inputs", "beyond_geometry", "high_noise_signal")}, step)
    out["total_seconds"] = time.time() - t_start
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez_compressed(os.path.join(args.out, "curves.npz"), **arrays)
    log(kv("P0 done", out=args.out, total_s=out["total_seconds"]))
    tap.remove()
    if wb is not None:
        try:
            wb.finish()
        except Exception:
            pass
    return out


if __name__ == "__main__":
    main()
