"""TRIAD-R S2 ORACLE MATRIX (gate (b) of results/N002-*/S2_harness_spec.md section 6): what does the interaction stream buy, per noise level?

For one S2 checkpoint (EMA weights; any arm A/B/C/Z) and >= 200 held-out chains (val split of ``RingDataset``, STRICT mode: a row that cannot be built is
dropped and counted, never replaced; ``requested`` / ``evaluated`` are reported) every example is run at the four paired noise levels (t_ca, t_lat) of
``p0_probe.DEFAULT_LEVELS`` (0.2/0.02, 0.5/0.1, 0.8/0.4, 0.95/0.8) under four stream conditions with COMMON RANDOM NUMBERS (same AE sample, same x_0 noise,
same x_t for every mode: asserted); the interactions|structure contrast adds ONE extra level (t_ca=0.9, t_lat=0.1) in mode 1 only (see below):

    ring_val_mode 0 tube (the training distribution) | 1 all-MASK (stream carries nothing) | 2 full reveal (true map) | 3 shuffled (another chain's map)

Per-example terms of the S2 model: ``bb_ca`` velocity loss, ``local_latents`` loss, ``struct_sum`` (their sum = the in-loop ``d_structure``), ``ca_x0_rmsd``
(Kabsch-aligned CA RMSD of the x_1 prediction, Angstrom), ``seq_ce`` (the sequence readout's CE: the "sequence probe"), ``ring_ce`` (class-weighted CE of the
final interaction head on the HIDDEN observed pairs, as in training) and ``ring_ce_unw`` (unweighted).

Cells (all deltas are oriented so that POSITIVE = the information helps; 95 % CIs from a CLUSTER bootstrap over sequence clusters, ``--n-boot`` resamples):
  structure|interactions  per level, per term in {bb_ca, local_latents, struct_sum, ca_x0_rmsd}:
        delta = loss(all-MASK) - loss(true map)            "true map helps"
        shuffled = loss(all-MASK) - loss(shuffled map)     "any map helps"
        margin = loss(shuffled) - loss(true map)           "TRUE map beats a wrong one" (= delta - shuffled; must be > 0)
        verdict POS (delta CI > 0 and margin CI > 0) | WEAK (delta CI > 0 but the shuffled map does as well) | NS | NEG (delta CI < 0)
  sequence|interactions   the same on ``seq_ce``
  interactions|structure  ring head on all-MASK, t_ca changes and t_lat is HELD FIXED (``--ring-contrasts 0.5:0.9@0.1`` = (0.5, 0.1) vs (0.9, 0.1); changing the
        latent noise too would let a head that ignores the CA improve through cleaner latents): CE(t_ca=0.5) - CE(t_ca=0.9) (> 0: better structure -> better
        interaction prediction), AUC / AP of the head on hidden observed pairs (all pairs and the HARD stratum, P0 definitions) at 0.9 minus at 0.5, plus the
        full per-level table.
Bootstrap units are SEQUENCE clusters: the part of the manifest cluster string before the first '+' (``ring_data.cluster_group``; ``abc+HEM`` and ``abc`` are one
unit); ``n_units`` is that count, ``n_units_raw`` the number of raw cluster strings. The shuffled donor is another chain of a DIFFERENT sequence cluster whenever the micro-batch
allows it (rows are reordered so that cyclic neighbours differ; rows left with a same-cluster donor are excluded from the shuffled cells and counted).
Orientation check: an S2 model whose ``ring_embed`` is still zero (the released model at step 0, ``--bundle /w/rfd4_weights --arm A``) must give delta = 0 exactly.

Outputs ONE small JSON (``--out``, schema ``oracle_s2/v1``) and stdout lines::

    ORACLE cell=structure|interactions term=bb_ca t_ca=0.5 delta=... ci=[lo,hi] shuffled=... margin=... margin_ci=[lo,hi] rel_pct=... verdict=POS n=256

Run (container, 1 H100, ~0.3 GPU-h; see oracle_s2.sbatch and EVAL_README.md)::

    python -m triadj_ext.s2.oracle_s2 --ckpt /w/triadr_s2/B/checkpoints/chk_..-ema.ckpt --arm B --tag B_s1000 --out /out/oracle.json --n-val 384

The pure statistics (bootstrap, cells, Kabsch) are import-light (numpy + torch) and unit-tested without the model: triadj_ext/tests/test_s2_eval.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from triadj_ext.s2.gates_report import sequence_cluster  # noqa: F401  (stdlib-only module shared with the gate report)

SCHEMA = "oracle_s2/v1"
MODE_NAMES = {0: "tube", 1: "all-MASK", 2: "full", 3: "shuffled"}
DEFAULT_LEVELS = ((0.2, 0.02), (0.5, 0.1), (0.8, 0.4), (0.95, 0.8))  # (t_ca, t_lat) = p0_probe levels
DEFAULT_CONTRASTS = "0.5:0.9@0.1"                                       # interactions|structure: t_ca 0.5 -> 0.9 at the FIXED t_lat 0.1 (extra level (0.9, 0.1), mode 1 only)
STRUCT_TERMS = ("bb_ca", "local_latents", "struct_sum", "ca_x0_rmsd")
SEQ_TERMS = ("seq_ce",)
RING_TERMS = ("ring_ce", "ring_ce_unw")
ALL_TERMS = STRUCT_TERMS + SEQ_TERMS + RING_TERMS + ("n_hidden",)
RING_AUC_METRICS = ("auc", "ap", "auc_hard", "ap_hard")
N_CLS, NO_INT, RING_MASK = 23, 22, 23


def log(*a):
    print(*a, flush=True)


def _num(x):
    """float or None (NaN / inf -> None: JSON has no NaN)."""
    x = float(x)
    return x if math.isfinite(x) else None


def _fmt(x, spec="%.5g"):
    return "nan" if x is None else spec % x


# ------------------------------------------------------------------------------------------------------------------ statistics
def boot_weights(unit_idx: np.ndarray, n_boot: int, seed: int = 0) -> np.ndarray:
    """[1 + n_boot, U] cluster resampling counts; row 0 = the point estimate (all ones). ``unit_idx`` [E] ints in 0..U-1."""
    U = int(unit_idx.max()) + 1 if len(unit_idx) else 0
    W = np.ones((1, U), dtype=np.float64)
    if n_boot > 0 and U > 0:
        draw = np.random.default_rng(seed).integers(0, U, size=(n_boot, U))
        W = np.concatenate([W, np.stack([np.bincount(r, minlength=U) for r in draw]).astype(np.float64)], 0)
    return W


def _unit_sums(x: np.ndarray, unit_idx: np.ndarray, U: int) -> np.ndarray:
    return np.bincount(unit_idx, weights=x, minlength=U)


def boot_ratio(num: np.ndarray, den: np.ndarray, unit_idx: np.ndarray, W: np.ndarray) -> np.ndarray:
    """[1 + n_boot] ratio of sums sum(num)/sum(den) under every cluster resample (row 0 = the point estimate)."""
    U = W.shape[1]
    N, D = W @ _unit_sums(num, unit_idx, U), W @ _unit_sums(den, unit_idx, U)
    return np.where(D > 0, N / np.where(D > 0, D, 1.0), np.nan)


def stat(arr: np.ndarray) -> dict:
    """[1 + n_boot] bootstrap array -> {"value": point estimate, "ci95": [2.5 %, 97.5 %] or None}."""
    v = _num(arr[0])
    ci = None
    if len(arr) >= 20:
        fin = arr[1:][np.isfinite(arr[1:])]
        if len(fin) >= 20:
            ci = [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))]
    return {"value": v, "ci95": ci}


def mean_stat(a: np.ndarray, unit_idx: np.ndarray, W: np.ndarray) -> dict:
    ok = np.isfinite(a)
    return stat(boot_ratio(np.where(ok, a, 0.0), ok.astype(np.float64), unit_idx, W))


def paired_stat(a: np.ndarray, b: np.ndarray, unit_idx: np.ndarray, W: np.ndarray, valid: np.ndarray | None = None) -> tuple[dict, dict, int]:
    """(mean of a - b, relative mean (a - b) / mean(a), n) over the examples where both are finite (or ``valid``), cluster bootstrap."""
    ok = (np.isfinite(a) & np.isfinite(b)) if valid is None else valid
    d = np.where(ok, a - b, 0.0)
    den = ok.astype(np.float64)
    delta = stat(boot_ratio(d, den, unit_idx, W))
    rel = stat(boot_ratio(d, np.where(ok, a, 0.0), unit_idx, W))
    return delta, rel, int(ok.sum())


def verdict_of(delta: dict, margin: dict | None) -> str:
    """POS: the true map helps AND beats the shuffled map | WEAK: it helps but the shuffled map does as well | NEG: it hurts | NS."""
    lo, hi = (delta["ci95"] or [None, None])
    if lo is None:
        return "NA"
    if hi < 0:
        return "NEG"
    if lo > 0:
        mlo = None if margin is None or margin["ci95"] is None else margin["ci95"][0]
        return "POS" if (mlo is not None and mlo > 0) else "WEAK"
    return "NS"


def simple_verdict(delta: dict) -> str:
    """POS (CI above 0) | NEG (CI below 0) | NS | NA, for cells without a shuffled control."""
    if not delta["ci95"]:
        return "NA"
    lo, hi = delta["ci95"]
    return "POS" if lo > 0 else ("NEG" if hi < 0 else "NS")


def kabsch_rmsd(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B] RMSD (units of the inputs) of ``pred`` rigidly aligned to ``ref`` over the ``mask`` tokens. pred/ref [B,N,3], mask [B,N] bool."""
    p, r, m = pred.double(), ref.double(), mask.double()[..., None]
    n = m[..., 0].sum(1).clamp_min(1.0)
    p = (p - (p * m).sum(1, keepdim=True) / n[:, None, None]) * m
    r = (r - (r * m).sum(1, keepdim=True) / n[:, None, None]) * m
    U, _, Vh = torch.linalg.svd(p.transpose(1, 2) @ r)
    d = torch.sign(torch.linalg.det(U @ Vh))
    d = torch.where(d == 0, torch.ones_like(d), d)
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    rot = U @ D @ Vh
    return (((p @ rot - r) ** 2 * m).sum((1, 2)) / n).sqrt()


# ------------------------------------------------------------------------------------------------------------- units / donors / levels
def index_units(units: list) -> tuple:
    """-> (int index per example [E], n_units): units are already normalised cluster ids."""
    uniq = {u: i for i, u in enumerate(dict.fromkeys(units))}
    return np.array([uniq[u] for u in units], dtype=np.int64), len(uniq)


def donor_permutation(clusters: list) -> tuple:
    """Order the rows of a micro-batch so that the cyclic neighbours (row r gets row r-1's map in ``ring_val_mode`` 3) belong to different sequence clusters.
    -> (permutation list, bad rows after the permutation = rows whose donor is still in their own cluster). Largest group first, filled into the even slots and
    then the odd slots: no two equal neighbours whenever the largest group has <= ceil(n/2) rows."""
    n = len(clusters)
    groups: dict = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    order = [i for g in sorted(groups.values(), key=lambda g: -len(g)) for i in g]
    slots = list(range(0, n, 2)) + list(range(1, n, 2))
    perm = [0] * n
    for slot, row in zip(slots, order):
        perm[slot] = row
    bad = [r for r in range(n) if n > 1 and clusters[perm[r]] == clusters[perm[(r - 1) % n]]]
    return perm, bad


def parse_contrasts(s: str) -> list:
    """'0.5:0.9@0.1,0.5:0.95@0.1' -> [(0.5, 0.9, 0.1), ...] = (t_ca_lo, t_ca_hi, t_lat held fixed)."""
    out = []
    for part in s.split(","):
        ca, tl = part.split("@")
        lo, hi = ca.split(":")
        out.append((float(lo), float(hi), float(tl)))
    return out


def resolve_levels(levels, contrasts) -> tuple:
    """Main levels + the extra (t_ca, t_lat) levels a contrast needs. -> (all levels, set of ring-only level indices: run in mode 1 only)."""
    out = list(levels)
    ring_only = set()
    for lo, hi, tl in contrasts:
        for t in (lo, hi):
            if _level_index(out, t, tl) is None:
                out.append((t, tl))
                ring_only.add(len(out) - 1)
    return tuple(out), ring_only


# ---------------------------------------------------------------------------------------------------------------- cells (pure)
class Table:
    """Per-example terms: ``v[term]`` = float64 [n_levels, 4 modes, E] (NaN where a mode was not run / is undefined)."""

    def __init__(self, n_levels: int, terms=ALL_TERMS):
        self.n_levels = n_levels
        self.parts: dict = {t: [[[] for _ in range(4)] for _ in range(n_levels)] for t in terms}
        self.terms = tuple(terms)

    def add(self, level: int, mode: int, term: str, values: torch.Tensor | np.ndarray) -> None:
        v = values.detach().float().cpu().numpy() if torch.is_tensor(values) else np.asarray(values, dtype=np.float32)
        self.parts[term][level][mode].append(v.astype(np.float64))

    def finalize(self) -> dict:
        out = {}
        for t in self.terms:
            rows = [[np.concatenate(self.parts[t][l][m]) if self.parts[t][l][m] else np.zeros(0) for m in range(4)] for l in range(self.n_levels)]
            E = max((len(x) for r in rows for x in r), default=0)
            arr = np.full((self.n_levels, 4, E), np.nan)
            for l in range(self.n_levels):
                for m in range(4):
                    arr[l, m, : len(rows[l][m])] = rows[l][m]
            out[t] = arr
        return out


def _level_index(levels, t_ca: float, t_lat: float | None = None) -> int | None:
    for i, (a, b) in enumerate(levels):
        if abs(a - t_ca) < 1e-6 and (t_lat is None or abs(b - t_lat) < 1e-6):
            return i
    return None


def stream_cells(tab: dict, levels, unit_idx: np.ndarray, W: np.ndarray, terms=STRUCT_TERMS + SEQ_TERMS) -> list[dict]:
    """structure|interactions and sequence|interactions cells (module docstring). ``tab`` = Table.finalize()."""
    cells = []
    for term in terms:
        if term not in tab:
            continue
        for li, (tca, tlat) in enumerate(levels):
            m1, m2, m3 = tab[term][li, 1], tab[term][li, 2], tab[term][li, 3]
            ok = np.isfinite(m1) & np.isfinite(m2) & np.isfinite(m3)  # joint support: margin = delta - shuffled holds exactly
            if not ok.any():
                continue
            delta, rel, n = paired_stat(m1, m2, unit_idx, W, ok)
            shuf, shuf_rel, _ = paired_stat(m1, m3, unit_idx, W, ok)
            margin, _, _ = paired_stat(m3, m2, unit_idx, W, ok)
            cells.append({"cell": "structure|interactions" if term in STRUCT_TERMS else "sequence|interactions", "term": term, "t_ca": tca, "t_lat": tlat, "n": n,
                          "n_units": int(len(np.unique(unit_idx[ok]))), "delta": delta, "delta_rel": rel, "shuffled": shuf, "shuffled_rel": shuf_rel, "margin": margin,
                          "mean_all_mask": mean_stat(np.where(ok, m1, np.nan), unit_idx, W)["value"], "verdict": verdict_of(delta, margin)})
    return cells


def ring_cells(tab: dict, levels, unit_idx: np.ndarray, W: np.ndarray, ring_summ: dict | None, contrasts=((0.5, 0.9, 0.1),)) -> list[dict]:
    """interactions|structure: the ring head on the all-MASK stream at t_ca=lo vs hi with t_lat HELD FIXED (CE from ``tab``, AUC/AP from the P0 histogram summary).
    A contrast whose two levels (same t_lat) were not evaluated yields no cell (never a (t_ca, t_lat)-confounded one)."""
    cells = []
    for lo_t, hi_t, tlat in contrasts:
        il, ih = _level_index(levels, lo_t, tlat), _level_index(levels, hi_t, tlat)
        if il is None or ih is None:
            continue
        for term in RING_TERMS:
            if term not in tab:
                continue
            lo, hi = tab[term][il, 1], tab[term][ih, 1]
            delta, rel, n = paired_stat(lo, hi, unit_idx, W)  # CE(lo_t) - CE(hi_t): > 0 = a cleaner structure gives a better interaction prediction
            if n:
                cells.append({"cell": "interactions|structure", "term": term, "t_ca": lo_t, "t_ca_ref": hi_t, "t_lat": tlat, "n": n, "delta": delta, "delta_rel": rel,
                              "verdict": simple_verdict(delta)})
        if ring_summ is not None:
            for m in RING_AUC_METRICS:
                a_lo, a_hi = ring_summ[(il, "M1")][m], ring_summ[(ih, "M1")][m]
                d = a_hi - a_lo  # AUC/AP(hi_t) - AUC/AP(lo_t): same resamples (one summarise call), > 0 = structure helps
                cells.append({"cell": "interactions|structure", "term": m, "t_ca": lo_t, "t_ca_ref": hi_t, "t_lat": tlat, "n": int(ring_summ["_meta"]["n_examples"]),
                              "delta": stat(d), "at_lo": stat(a_lo), "at_hi": stat(a_hi), "verdict": simple_verdict(stat(d))})
    return cells


def table_means(tab: dict, levels, unit_idx: np.ndarray, W: np.ndarray) -> dict:
    """{term: {t_ca: {mode name: [mean, ci_lo, ci_hi]}}}: raw per-condition means (the cells are differences of these)."""
    out = {}
    for term, arr in tab.items():
        out[term] = {}
        for li, (tca, _) in enumerate(levels):
            row = {}
            for m in range(4):
                if np.isfinite(arr[li, m]).any():
                    s = mean_stat(arr[li, m], unit_idx, W)
                    row[MODE_NAMES[m]] = [s["value"]] + (s["ci95"] or [None, None])
            out[term][f"{tca:g}"] = row
    return out


def cell_line(c: dict) -> str:
    ci = lambda s: "ci=[%s,%s]" % tuple(_fmt(x, "%.4g") for x in (s["ci95"] or (None, None)))
    if c["cell"] == "interactions|structure":
        return (f"ORACLE cell={c['cell']} term={c['term']} t_ca={c['t_ca']:g}->{c['t_ca_ref']:g} t_lat={c['t_lat']:g} delta={_fmt(c['delta']['value'], '%.4g')} {ci(c['delta'])} "
                f"verdict={c['verdict']} n={c['n']}")
    rel = c["delta_rel"]["value"]
    return (f"ORACLE cell={c['cell']} term={c['term']} t_ca={c['t_ca']:g} delta={_fmt(c['delta']['value'], '%.4g')} {ci(c['delta'])} "
            f"shuffled={_fmt(c['shuffled']['value'], '%.4g')} margin={_fmt(c['margin']['value'], '%.4g')} margin_{ci(c['margin'])} "
            f"rel_pct={_fmt(None if rel is None else 100 * rel, '%.3g')} verdict={c['verdict']} n={c['n']}")


# ------------------------------------------------------------------------------------------------------------ model + data (GPU/CPU)
def load_model(args, device: torch.device):
    """The S2 model with EMA weights: a training/EMA checkpoint (``--ckpt``, rebuilt strictly from its own hparams, as ``generate`` does) or a bundle
    (``--bundle``: the released one = arm at step 0, or an expanded base bundle). The arm class MUST be registered before anything is built."""
    from triadj_ext import common
    from triadj_ext.model_s2 import arm_nn_overrides, guard_arm_for_args, register_arm

    guard = guard_arm_for_args(args.arm, ckpt=args.ckpt or "", bundle=args.bundle or "")  # raises when --arm disagrees with the checkpoint (Z on an A ckpt, C on A, ...)
    arm = register_arm(args.arm)
    cpu_ov = dict(common.CPU_NN_OVERRIDES) if device.type != "cuda" else {}
    if args.bundle:
        cfg = json.load(open(os.path.join(args.bundle, "config.json")))["config"]
        expanded = "ring" in (cfg.get("nn") or {})
        if expanded:
            from rfproteina.proteina import Proteina

            model = Proteina.from_pretrained(args.bundle, nn_overrides=cpu_ov or None)
        else:
            from triadj_ext import pair_stream as ps

            model, _ = ps.load_triadj_proteina(args.bundle, nn_overrides={**cpu_ov, **arm_nn_overrides(arm)})
        what = f"bundle:{args.bundle} ({'expanded' if expanded else 'released + zero-init ring modules'})"
    else:
        import rfproteina.generate as G
        from rfproteina.proteina import Proteina
        from rfproteina.utils.weights import inference_state_dict

        cfg_exp = G._inference_cfg_exp(args.ckpt, cpu_ov)  # None when there is nothing to override
        model = Proteina.load_from_checkpoint(args.ckpt, strict=True, autoencoder_ckpt_path=None, load_training_metrics=False, weights_only=False,
                                              map_location="cpu", **({} if cfg_exp is None else {"cfg_exp": cfg_exp}))
        if args.ckpt.endswith("-ema.ckpt"):
            what = f"ckpt:{args.ckpt} (EMA export)"
        else:
            raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(inference_state_dict(raw, args.ckpt), strict=True)  # the SimpleEMA state of a training checkpoint
            what = f"ckpt:{args.ckpt} (EMA weights of a training checkpoint)"
            del raw
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"arm": arm, "arm_check": guard["verified"], "arm_declared": guard["declared"], "weights": what, "ring_embed_absmax": float(model.nn.ring_embed.weight.abs().max()), "nn_class": type(model.nn).__name__,
                   "fm_class": type(model.fm).__name__, "peft": getattr(model.nn, "peft_config", None) is not None}


def build_loader(args):
    """-> (iterable of batches, info). Strict mode, deterministic: the SAME rows as the in-loop S2 validation (``split='val'``, ``n_val`` rows in cluster-hash order)."""
    if args.loader_factory:
        from triadj_ext import p0_probe as P0

        tr, va, info = P0._load_factory(args.loader_factory)(args)
        return va, dict(info or {})
    from triadj_ext import p0_probe as P0

    if P0.P0RingDataset is None:
        raise SystemExit(f"triadj_ext.ring_data not importable: {P0._RING_IMPORT_ERROR}")
    import lmdb  # noqa: F401  (RING reader inside the workers; put the pip --target dir on PYTHONPATH)

    ds = P0.P0RingDataset(None, rows="all", crop_size=args.crop_size, seed=5, split="val", n_val=args.n_val, ring_labels=True, substitute_on_failure=False,
                          transform={"path_length_search_cutoff": 64})
    dl = dict(collate_fn=P0.p0_collate, drop_last=False, num_workers=args.workers, pin_memory=False, batch_size=args.micro_batch, shuffle=False)
    if args.workers > 0:
        dl.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    loader = torch.utils.data.DataLoader(ds, **dl)
    return loader, {"val_examples": len(ds), "split": "val", "rows": "all", "crop_size": args.crop_size, "n_val": args.n_val}


def ring_terms(nb: dict, out: dict, w_nointer: float = 0.2) -> dict:
    """[B] ring CE on the hidden observed pairs (class-weighted like training / unweighted) and the number of hidden pairs; NaN without a head or hidden pairs."""
    import torch.nn.functional as F

    from triadj_ext import ring_flow as RF

    B = nb["padding_mask"].shape[0]
    nan = torch.full((B,), float("nan"))
    inter = out.get("interaction")
    gt, obs = RF.get_gt_and_obs(nb)
    if inter is None or gt is None:
        return {"ring_ce": nan, "ring_ce_unw": nan, "n_hidden": nan}
    pair = RF.pair_mask_of(nb)
    valid = pair & RF.offdiag(pair) & obs
    masked = (nb["ring_state"] == RF.RING_MASK) & valid
    lg = inter["logits"].float().permute(0, 3, 1, 2)
    cw = torch.ones(N_CLS, device=lg.device)
    cw[NO_INT] = w_nointer
    ce_w = F.cross_entropy(lg, gt, weight=cw, reduction="none")
    ce_u = F.cross_entropy(lg, gt, reduction="none")
    n = masked.sum((1, 2))
    nz = n > 0
    nn_ = n.clamp_min(1)
    r1 = torch.where(nz, (ce_w * masked).sum((1, 2)) / nn_, torch.full_like(n, float("nan"), dtype=lg.dtype))
    r2 = torch.where(nz, (ce_u * masked).sum((1, 2)) / nn_, torch.full_like(n, float("nan"), dtype=lg.dtype))
    return {"ring_ce": r1.cpu(), "ring_ce_unw": r2.cpu(), "n_hidden": n.float().cpu()}


def ca_x0_rmsd(model, nb: dict, out: dict) -> torch.Tensor:
    """Kabsch-aligned CA RMSD (Angstrom) of the model's x_1 prediction (supervised tokens); NaN if the model cannot give one."""
    B = nb["padding_mask"].shape[0]
    try:
        from rfproteina.flow_matching.product_space_flow_matcher import compute_flow_supervision_masks

        o = model.fm.nn_out_add_clean_sample_prediction(nb, dict(out))
        mask = compute_flow_supervision_masks(nb)["bb_ca"].bool()
        return (kabsch_rmsd(o["bb_ca"]["x_1"].float(), nb["x_1"]["bb_ca"].float(), mask) * 10.0).cpu()
    except Exception as e:  # noqa: BLE001  (a diagnostic term must never kill the run)
        log(f"ORACLE warn ca_x0_rmsd unavailable: {type(e).__name__}: {str(e)[:100]}")
        return torch.full((B,), float("nan"))


def run_example_terms(model, nb: dict, n_recycle: int, amp: bool, device) -> tuple[dict, dict]:
    """One forward + the training loss terms, per example. Returns ({term: [B] cpu tensor}, nn_out)."""
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        out = model.call_nn(nb, n_recycle=n_recycle)
        L = model.fm.compute_loss(batch=nb, nn_out=out)
    t = {"bb_ca": L["bb_ca"].detach().float().cpu(), "local_latents": L["local_latents"].detach().float().cpu()}
    t["struct_sum"] = t["bb_ca"] + t["local_latents"]
    ce = L.get("sequence_prediction_ce_justlog")
    t["seq_ce"] = ce.detach().float().cpu() if ce is not None else torch.full_like(t["bb_ca"], float("nan"))
    t["ca_x0_rmsd"] = ca_x0_rmsd(model, nb, out)
    t.update(ring_terms(nb, out))
    return t, out


@torch.no_grad()
def evaluate(model, loader, args, device, levels, modes, ring_only=frozenset()) -> dict:
    """The oracle loop. -> dict(tab=Table, store=EvalStore (all-MASK ring head stats), units, ids, load=..., checks=...)."""
    from triadj_ext import p0_probe as P0

    amp = device.type == "cuda"
    L = len(levels)
    levels_t = (torch.tensor([a for a, _ in levels], dtype=torch.float32, device=device), torch.tensor([b for _, b in levels], dtype=torch.float32, device=device))
    tab = Table(L)
    store = P0.EvalStore(("M1",), L)
    logw = torch.log(P0.class_weights("triadj")).to(device)
    ids_all_seen: set = set()
    units, ids, raw_by_id = [], [], {}
    checks = {"crn_violations": 0, "no_donor_rows": 0, "donor_same_cluster_rows": 0, "no_observed_pairs": 0, "batches": 0}
    t0 = time.time()
    rng_cpu = torch.get_rng_state()
    for bidx, raw in enumerate(loader):
        if args.max_batches and bidx >= args.max_batches:
            break
        full, meta = P0.unpack(raw, device)
        bsz = full["padding_mask"].shape[0] if "padding_mask" in full else 0
        store.note_loaded(meta, bsz)
        if bsz == 0:
            continue
        ids_b = list(meta["example_ids"]) if meta["example_ids"] else [f"ex{len(ids) + k}" for k in range(bsz)]
        clu_b = [sequence_cluster(c, ids_b[k]) for k, c in enumerate(meta["clusters"])] if meta["clusters"] is not None else list(ids_b)   # SEQUENCE clusters
        for k, e in enumerate(ids_b):
            raw_by_id.setdefault(e, str(meta["clusters"][k]) if meta["clusters"] is not None else e)
        keep = [k for k, e in enumerate(ids_b) if e not in ids_all_seen]
        if len(keep) < bsz:
            store.n_dup += bsz - len(keep)
            if not keep:
                continue
            full, ids_b, clu_b, bsz = P0.index_batch(full, keep, bsz), [ids_b[k] for k in keep], [clu_b[k] for k in keep], len(keep)
        y, m = P0.label_masks(full)
        has = m.flatten(1).any(1)  # a row with no observed pair has no map to reveal (its "full" condition would silently equal all-MASK)
        if not bool(has.all()):
            checks["no_observed_pairs"] += int((~has).sum())
            store.n_no_obs += int((~has).sum())
            keep = [k for k in range(bsz) if bool(has[k])]
            if not keep:
                continue
            full, ids_b, clu_b, bsz = P0.index_batch(full, keep, bsz), [ids_b[k] for k in keep], [clu_b[k] for k in keep], len(keep)
            y, m = P0.label_masks(full)
        perm, bad_donor = donor_permutation(clu_b)  # cyclic neighbours = donors of mode 3: different sequence clusters wherever possible
        if perm != list(range(bsz)):
            full, ids_b, clu_b = P0.index_batch(full, perm, bsz), [ids_b[k] for k in perm], [clu_b[k] for k in perm]
            y, m = P0.label_masks(full)
        bad_t = torch.zeros(bsz, dtype=torch.bool)
        if bsz > 1:
            bad_t[bad_donor] = True
            checks["donor_same_cluster_rows"] += len(bad_donor)
        ids_all_seen.update(ids_b)
        ex, yy = m.nonzero()[:, 0], y[m]
        counts = torch.bincount(ex * N_CLS + yy, minlength=bsz * N_CLS).reshape(bsz, N_CLS)
        L_ex = (m | m.transpose(1, 2)).any(-1).sum(-1)
        store.add_examples(ids_b, clu_b, counts)
        ids += ids_b
        units += clu_b
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            torch.manual_seed(args.seed + 1000 * bidx)
            clean = P0.clean_batch(model, full)  # ONE AE posterior sample per example, shared by every level / mode
            hrd = P0.hard_mask(full, clean["x_1"]["bb_ca"])[m]
        for l in range(L):
            lvl = torch.full((bsz,), l, dtype=torch.long, device=device)
            ref_xt = None
            for mode in ([1] if (l in ring_only and 1 in modes) else (modes if l not in ring_only else [])):
                if mode == 3 and bsz < 2:  # the shuffled control needs a donor row in the batch
                    checks["no_donor_rows"] += bsz
                    for term in ALL_TERMS:
                        tab.add(l, mode, term, np.full(bsz, np.nan))
                    continue
                c2 = dict(clean, ring_val_mode=torch.full((bsz,), mode, dtype=torch.long, device=device))
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    nb = P0.noisy_batch(model, c2, lvl, levels_t, seed=args.seed + 1000 * bidx + 1 + l)  # same seed for every mode = common random numbers
                xt = nb["x_t"]["bb_ca"]
                if ref_xt is None:
                    ref_xt = xt
                elif not torch.equal(xt, ref_xt):
                    checks["crn_violations"] += 1
                terms, out = run_example_terms(model, nb, args.n_recycle, amp, device)
                for term in ALL_TERMS:
                    v = terms[term].clone() if mode == 3 else terms[term]
                    if mode == 3:
                        v[bad_t] = float("nan")  # donor in the receiver's own sequence cluster: not a control, excluded jointly
                    tab.add(l, mode, term, v)
                if mode == 1 and out.get("interaction") is not None:  # all hidden: the P0 AUC/AP histogram statistics on the observed pairs
                    store.add(l, "M1", P0.batch_stats(out["interaction"]["logits"][m].float() - logw, ex, yy, bsz, L_ex, hrd))
        checks["batches"] += 1
        if bidx % 5 == 0:
            log(f"ORACLE progress batch={bidx} examples={len(ids)} el_s={time.time() - t0:.0f}")
    torch.set_rng_state(rng_cpu)
    return {"tab": tab, "store": store, "units": units, "raw_units": [raw_by_id[i] for i in ids], "ids": ids, "checks": checks, "seconds": time.time() - t0}


def summarise_run(res: dict, levels, args, meta: dict) -> dict:
    """Table + store -> the JSON of the module docstring (cells, means, ring head table, load accounting)."""
    from triadj_ext import p0_probe as P0

    tab = res["tab"].finalize()
    unit_idx, n_units = index_units(res["units"])
    W = boot_weights(unit_idx, args.n_boot, args.seed + 7)
    store = res["store"]
    ring_summ = None
    if len(res["ids"]) and np.isfinite(tab["ring_ce"][:, 1]).any():
        ring_summ = P0.summarise(store, n_boot=args.n_boot, seed=args.seed + 11, device="cpu")
    cells = stream_cells(tab, levels, unit_idx, W)
    cells += ring_cells(tab, levels, unit_idx, W, ring_summ, parse_contrasts(args.ring_contrasts))
    ring_head = {}
    if ring_summ is not None:
        for l, (tca, _) in enumerate(levels):
            r = ring_summ[(l, "M1")]
            ring_head[f"{tca:g}"] = {k: stat(r[k]) for k in ("auc", "ap", "macro_ap", "topl", "auc_hard", "ap_hard")}
            ring_head[f"{tca:g}"]["ece"] = _num(r["ece"])
    load = store.load_report()
    load["evaluated"] = len(res["ids"])
    load["no_donor_rows"] = res["checks"]["no_donor_rows"]
    load["donor_same_cluster_rows"] = res["checks"]["donor_same_cluster_rows"]
    out = {"schema": SCHEMA, "tag": args.tag, "model": meta, "args": {k: v for k, v in vars(args).items() if k not in ("loader_factory",)},
           "levels": [list(x) for x in levels], "modes": {str(k): MODE_NAMES[k] for k in args.modes_list}, "load": load, "n_units": int(n_units),
           "n_units_raw": int(len(set(res.get("raw_units", res["units"])))), "ring_only_levels": sorted(res.get("ring_only", [])),
           "checks": res["checks"], "cells": cells, "means": table_means(tab, levels, unit_idx, W), "ring_head_all_mask": ring_head, "seconds": res["seconds"]}
    return out


def parse_levels(s: str):
    return tuple(tuple(float(x) for x in p.split(":")) for p in s.split(","))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default="", help="S2 training checkpoint (EMA state is used) or chk_*-ema.ckpt, visible in the container")
    p.add_argument("--bundle", default="", help="alternative to --ckpt: /w/rfd4_weights (released model = step-0 control) or an expanded base bundle")
    p.add_argument("--arm", required=True, choices=("A", "B", "C", "Z"), help="arm class to register (C = per-block heads)")
    p.add_argument("--tag", default="oracle")
    p.add_argument("--out", default=f"/tmp/{os.environ.get('USER', 'u')}/oracle_s2.json")
    p.add_argument("--n-val", type=int, default=384, help="held-out rows REQUESTED: ~27 %% of the val-manifest rows are non-protein units (unit_not_protein) and are "
                   "dropped in strict mode, so 384 gives ~280 evaluated; the first 256 of them are the rows of the in-loop S2 validation")
    p.add_argument("--min-evaluated", type=int, default=200, help="warn when fewer chains were evaluated")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=8, help="rows per forward (also the pool the shuffled donor is drawn from: row r shows row r-1's map, rows are ordered so that the donor is in another sequence cluster)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--levels", default=",".join(f"{a}:{b}" for a, b in DEFAULT_LEVELS), help="t_ca:t_lat,... (t=0 noise, 1 data)")
    p.add_argument("--ring-contrasts", default=DEFAULT_CONTRASTS, help="interactions|structure cells: lo:hi@t_lat,... = t_ca lo -> hi with t_lat held FIXED; the levels they need "
                   "and --levels lacks are added and run in mode 1 only")
    p.add_argument("--modes", default="0,1,2,3")
    p.add_argument("--n-recycle", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-batches", type=int, default=0, help="debug: stop after k loader batches")
    p.add_argument("--loader-factory", default="", help="path.py:func(args) -> (train, val, info); tests / loader-free smokes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    if bool(a.ckpt) == bool(a.bundle):
        p.error("give exactly one of --ckpt / --bundle")
    a.modes_list = [int(x) for x in a.modes.split(",")]
    return a


def main(argv=None) -> dict:
    args = parse_args(argv)
    t_start = time.time()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    levels, ring_only = resolve_levels(parse_levels(args.levels), parse_contrasts(args.ring_contrasts))
    model, meta = load_model(args, device)
    log("ORACLE init " + " ".join(f"{k}={v}" for k, v in meta.items()) + f" device={device} load_s={time.time() - t_start:.0f}")
    loader, info = build_loader(args)
    res = evaluate(model, loader, args, device, levels, args.modes_list, ring_only)
    res["ring_only"] = ring_only
    out = summarise_run(res, levels, args, meta)
    out["split"] = info
    ld = out["load"]
    log(f"ORACLE load requested={ld['requested']} evaluated={ld['evaluated']} failed_to_load={ld['failed_to_load']} no_observed_pairs={ld['no_observed_pairs']} "
        f"duplicates_skipped={ld['duplicates_skipped']} no_donor_rows={ld['no_donor_rows']} donor_same_cluster_rows={ld['donor_same_cluster_rows']} units={out['n_units']} units_raw={out['n_units_raw']} crn_violations={out['checks']['crn_violations']} "
        f"drops={json.dumps(ld['drop_reasons'], sort_keys=True).replace(' ', '')}")
    if ld["evaluated"] < args.min_evaluated:
        log(f"ORACLE WARN evaluated={ld['evaluated']} < {args.min_evaluated} chains: raise --n-val (strict mode drops rows that cannot be built)")
    if out["checks"]["crn_violations"]:
        log("ORACLE WARN common-random-number check FAILED (x_t differs between modes): the cells are not paired")
    if meta["ring_embed_absmax"] == 0.0:
        log("ORACLE note ring_embed is exactly zero (released model / step 0): every delta must be 0 up to numerics")
    for c in out["cells"]:
        log(cell_line(c))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"ORACLE done out={args.out} total_s={time.time() - t_start:.0f}")
    return out


if __name__ == "__main__":
    main()
