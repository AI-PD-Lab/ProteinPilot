"""Per-design geometry + ADJ-conditioning metrics (pure numpy, CPU). Used by the scorer and by the spec
builder (native reference values), so a design and its reference are measured by the SAME code.

Metric glossary (all CB = RFD1-reconstructed C-beta, see eval_common.reconstruct_cb):
  rg, rg_n04          radius of gyration of CA, and Rg / N^0.4
  lr_cb               long-range CB contacts per residue: CB-CB < 8 A and |i-j| > 12   (A4 `lr_pr`)
  split               min over cuts k in [.25N,.75N] of cross-cut CB<8 contacts (|i-j|>=6) per smaller-side residue
  half_sep_n2         |centroid(CA[:N//2]) - centroid(CA[N//2:])|   (A4 convention, spec-independent)
  half_sep            |centroid(real-CB-else-CA over the span of the first half of the reference H/E blocks) -
                      centroid(... second half)|, spans contiguous (loops between blocks included). The project's
                      ferredoxin metric generalised (reproduces [3,32] vs [37,63]); LEGACY rep atom on purpose.
  ends_n6, ends_min_d CB-CB pairs < 6 A / min distance between the FIRST and LAST H/E block (legacy `b1b4_n6`, same rep atom)
  recovery            fraction of REQUIRED contact block pairs (state 1) with min CB-CB < 6 A
  violation           fraction of EXPLICIT non-contact block pairs (state 0) with min CB-CB < 6 A
  extent_n6           mean over required pairs of #(CB-CB < 6 A) residue pairs  (interface SIZE; recovery cannot see it)
  extent_ratio        sum of those counts over the same sum in the native reference (1.0 = native-sized interfaces)
  pos_min_d / neg_min_d  mean min CB-CB over required / non-contact pairs
  helix, strand, coil SS fractions (biotite P-SEA on CA = the training label's SS); ss_acc = agreement with the
                      revealed H/E labels of the spec
  ca_breaks           consecutive CA-CA > 4.2 A
  ca_clash            CA-CA < 3.5 A with |i-j| >= 3
  heavy_clash         heavy-atom pairs < 2.2 A between residues |i-j| >= 2
"""
from __future__ import annotations

import math

import numpy as np

import eval_common as ec


def _centroid(x: np.ndarray) -> np.ndarray:
    return np.nanmean(x, axis=0)


def geometry_metrics(ca, cb, heavy=None, heavy_res=None, sse=None) -> dict:
    ca = np.asarray(ca, float)
    cb = np.asarray(cb, float)
    N = len(ca)
    out: dict = {"N": N}
    cen = _centroid(ca)
    rg = float(np.sqrt(np.nanmean(((ca - cen) ** 2).sum(1))))
    out["rg"] = rg
    out["rg_n04"] = rg / N ** 0.4
    D = ec.pairwise(cb)
    idx = np.arange(N)
    sep = np.abs(idx[:, None] - idx[None, :])
    iu = np.triu_indices(N, 1)
    d_u, s_u = D[iu], sep[iu]
    out["lr_cb"] = float(((d_u < ec.LR_CUT) & (s_u > ec.LR_SEP)).sum() / N)
    # min-cut "split": cumulative sums of the contact map C6 = (D<8)&(sep>=6)
    C6 = ((D < ec.LR_CUT) & (sep >= 6)).astype(np.int64)
    P = np.cumsum(np.cumsum(C6, axis=0), axis=1)
    ks = range(int(math.ceil(0.25 * N)), int(0.75 * N) + 1)
    vals = [(P[k - 1, N - 1] - P[k - 1, k - 1]) / min(k, N - k) for k in ks if 1 <= k < N]
    out["split"] = float(min(vals)) if vals else float("nan")
    out["half_sep_n2"] = float(np.linalg.norm(_centroid(ca[: N // 2]) - _centroid(ca[N // 2:])))
    out["half_sep_rg"] = out["half_sep_n2"] / rg if rg > 0 else float("nan")
    dca = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    out["ca_breaks"] = int((dca > ec.BREAK_CUT).sum())
    DA = ec.pairwise(ca)
    out["ca_clash"] = int((DA[iu][s_u >= ec.CA_CLASH_SEP] < ec.CA_CLASH_CUT).sum())
    if heavy is not None and len(heavy):
        from scipy.spatial import cKDTree

        pairs = cKDTree(np.asarray(heavy, float)).query_pairs(ec.HEAVY_CLASH_CUT, output_type="ndarray")
        if len(pairs):
            hr = np.asarray(heavy_res)
            out["heavy_clash"] = int((np.abs(hr[pairs[:, 0]] - hr[pairs[:, 1]]) >= 2).sum())
        else:
            out["heavy_clash"] = 0
    else:
        out["heavy_clash"] = float("nan")
    if sse is None:
        sse = ec.sse_labels(ca)
    sse = np.asarray(sse)
    out["helix"] = float((sse == ec.SS_HELIX).mean())
    out["strand"] = float((sse == ec.SS_STRAND).mean())
    out["coil"] = float((sse == ec.SS_LOOP).mean())
    return out


def _pair_list(pairs):
    return [(int(a), int(b)) for a, b in (pairs or [])]


def half_spans(target: dict):
    """((a0,a1),(b0,b1)) residue spans [start, stop) of the two halves, or None.

    Default rule = the first / second half of the reference H/E blocks, each half taken as the CONTIGUOUS residue
    span from its first block's start to its last block's stop (loops in between included). For the synthetic
    ferredoxin B1 A1 B2 | B3 A2 B4 this reproduces the project's spans [3,32] and [37,63] (score_ferredoxin.py).
    A target may override with explicit `half1_span` / `half2_span` (legacy manifest)."""
    if target.get("half1_span") and target.get("half2_span"):
        return tuple(target["half1_span"]), tuple(target["half2_span"])
    blocks = target["blocks"]
    h1, h2 = target.get("half1_blocks"), target.get("half2_blocks")
    if not h1 or not h2:
        return None
    return (blocks[h1[0]]["start"], blocks[h1[-1]]["stop"]), (blocks[h2[0]]["start"], blocks[h2[-1]]["stop"])


def conditioned_metrics(cb, sse, target: dict, arm: dict, cb_leg=None) -> dict:
    """ADJ/SS-conditioning metrics of one design against its reference target + arm record (manifest dicts).

    `cb` = RFD1-reconstructed CB (label convention) for contacts; `cb_leg` = real CB else CA, used ONLY for
    half_sep and ends_n6 so that they stay numerically comparable with the N002/N003 legacy tables."""
    cb = np.asarray(cb, float)
    cb_leg = cb if cb_leg is None else np.asarray(cb_leg, float)
    blocks = target["blocks"]
    out: dict = {}
    spans = half_spans(target)
    if spans:
        (a0, a1), (b0, b1) = spans
        out["half_sep"] = float(np.linalg.norm(_centroid(cb_leg[a0:a1]) - _centroid(cb_leg[b0:b1])))
    else:
        out["half_sep"] = float("nan")
    # first-vs-last H/E block ("B1-B4" of the ferredoxin): CB pairs < 6 A (legacy b1b4_n6)
    eb = target.get("ends_blocks") or ([0, len(blocks) - 1] if len(blocks) >= 2 else None)
    if eb:
        ia, ib = ec.block_idx(blocks, eb[0]), ec.block_idx(blocks, eb[1])
        out["ends_n6"] = float(ec.block_n_below(cb_leg, ia, ib))
        out["ends_min_d"] = ec.block_min_dist(cb_leg, ia, ib)
    else:
        out["ends_n6"] = out["ends_min_d"] = float("nan")
    pos, neg = _pair_list(arm.get("pos_pairs")), _pair_list(arm.get("neg_pairs"))
    native_n6 = target.get("pair_n6", {})
    if pos:
        d = [ec.block_min_dist(cb, ec.block_idx(blocks, i), ec.block_idx(blocks, j)) for i, j in pos]
        n6 = [ec.block_n_below(cb, ec.block_idx(blocks, i), ec.block_idx(blocks, j)) for i, j in pos]
        out["recovery"] = float(np.mean([x < ec.CONTACT_CUT for x in d]))
        out["pos_min_d"] = float(np.nanmean(d))
        out["extent_n6"] = float(np.mean(n6))
        nat = [native_n6.get(f"{i}-{j}") for i, j in pos]
        out["extent_ratio"] = float(sum(n6) / sum(nat)) if all(v is not None for v in nat) and sum(nat) > 0 else float("nan")
    else:
        out.update(recovery=float("nan"), pos_min_d=float("nan"), extent_n6=float("nan"), extent_ratio=float("nan"))
    if neg:
        d = [ec.block_min_dist(cb, ec.block_idx(blocks, i), ec.block_idx(blocks, j)) for i, j in neg]
        out["violation"] = float(np.mean([x < ec.CONTACT_CUT for x in d]))
        out["neg_min_d"] = float(np.nanmean(d))
    else:
        out["violation"] = float("nan")
        out["neg_min_d"] = float("nan")
    # --- SS agreement on revealed helix/strand tokens
    ss = arm.get("ss")
    if ss is not None and len(ss) == len(sse):
        ss = np.asarray(ss)
        m = np.isin(ss, (ec.SS_HELIX, ec.SS_STRAND))
        out["ss_acc"] = float((np.asarray(sse)[m] == ss[m]).mean()) if m.any() else float("nan")
    else:
        out["ss_acc"] = float("nan")
    return out


def design_row(chain: "ec.Chain", target: dict | None = None, arm: dict | None = None) -> dict:
    """All metrics of one design (Chain) as a flat dict; `len_ok` False when the length mismatches the spec."""
    sse = ec.sse_labels(chain.ca)
    row = geometry_metrics(chain.ca, chain.cb, chain.heavy, chain.heavy_res, sse)
    row["len_ok"] = True
    if target is not None and arm is not None:
        if len(chain) != target["n_res"]:
            row["len_ok"] = False
        else:
            row.update(conditioned_metrics(chain.cb, sse, target, arm, chain.cb_leg))
    return row


#: metrics that are aggregated per cell, in report order
CELL_METRICS = [
    "half_sep", "ends_n6", "ends_min_d", "recovery", "violation", "extent_n6", "extent_ratio", "pos_min_d", "neg_min_d", "ss_acc",
    "rg_n04", "rg", "lr_cb", "split", "half_sep_n2", "half_sep_rg", "helix", "strand", "coil",
    "ca_breaks", "ca_clash", "heavy_clash",
]
