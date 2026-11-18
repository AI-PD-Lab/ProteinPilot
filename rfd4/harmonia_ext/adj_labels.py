"""HARMONIA SS / block-adjacency labels for RFD4-Proteina (pure numpy, no RFD4 import).

Port of the RFD3 HARMONIA-v2 contract (foundry/src/foundry/data/ss_adj.py + FeaturizeSSADJ):

SS (RFD4's own C_SEC channel, SecondaryStructureSeqFeat): 0 NONE (= masked / unspecified), 1 HELIX, 2 SHEET, 3 LOOP.
ADJ (new pair input, token space [I, I]):
    0 NON_CONTACT  revealed, the two SS blocks do not touch (no reconstructed-CB pair < 6 A)
    1 CONTACT      revealed, >= 1 CB pair < 6 A (RFdiffusion1 rule, distinct H/E blocks only)
    2 MASKED       unspecified (loops, same block, ineligible tokens, hidden pairs); embeds to exactly 0
    3 HOT          a revealed contact with a LARGE interface (CB-pair count n6 in the top of the protein's contacts)
    4 COLD         a revealed non-contact that is CLEARLY apart (min CB distance >= COLD_MIN_DIST)
    5 PAD          padding (collate fill); embeds to exactly 0
HOT / COLD are defined by the data (not by an auxiliary loss, which left them inert on RFD3): HOT = "contact plus as much interface as
possible", COLD = "non-contact, enforced more strongly".
"""
from __future__ import annotations

import numpy as np

SS_NONE, SS_HELIX, SS_SHEET, SS_LOOP = 0, 1, 2, 3
ADJ_NON_CONTACT, ADJ_CONTACT, ADJ_MASKED, ADJ_HOT, ADJ_COLD, ADJ_PAD = 0, 1, 2, 3, 4, 5
N_ADJ_ROWS = 6
ZERO_STATES = (ADJ_MASKED, ADJ_PAD)

CONTACT_CUTOFF = 6.0          # RFD1: reconstructed CB, strict < 6 A
COLD_MIN_DIST = 12.0          # COLD = revealed non-contact with min CB distance >= 12 A
HOT_MIN_N6 = 6                # HOT needs at least 6 CB pairs < 6 A ...
HOT_QUANTILE = 0.75           # ... and n6 >= the 75th percentile of the SHOWN contact block pairs
MIN_BLOCK_LEN = {SS_HELIX: 4, SS_SHEET: 2}   # shorter H/E runs are not blocks (no ADJ), their SS label is kept

# compactness gate (R2): SS/ADJ are only revealed on rows whose protein chains are compact domains, measured exactly like the panel's
# universal split detector (RFD3 harness eval/design_metrics.geometry_metrics: thresholds = natural compact-domain p5/p5/p95)
GATE_MIN_CHAIN = 40
T_SPLIT, T_LR, T_RG = 0.67, 0.92, 2.15
LR_CUT, LR_SEP = 8.0, 12

# curriculum (HARMONIA K3 adapted: no interaction-site / remodel modes in a single-structure monomer/interface stream)
MODES = ("uncond", "ss_only", "adj_only", "full", "sparse")
MODE_P = (0.10, 0.15, 0.15, 0.35, 0.25)


def virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """RFdiffusion / ProteinMPNN ideal CB from backbone N, CA, C ([..., 3])."""
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def ss_blocks(ss: np.ndarray, chain: np.ndarray, resid: np.ndarray | None = None) -> np.ndarray:
    """Block id per residue/token (-1 = no block): maximal runs of HELIX or SHEET within one chain and without a residue-number gap
    (``resid`` consecutive), at least MIN_BLOCK_LEN long. Called on the FINAL tokens after the crop (blocks computed on
    the source chain encoded cropped-away SS through the same-block mask, and runs merged across chain breaks)."""
    n = len(ss)
    blk = np.full(n, -1, dtype=np.int64)
    brk = np.zeros(n, dtype=bool)
    if resid is not None and n > 1:
        brk[1:] = np.diff(np.asarray(resid, dtype=np.int64)) != 1
    nb, i = 0, 0
    while i < n:
        j = i + 1
        while j < n and ss[j] == ss[i] and chain[j] == chain[i] and not brk[j]:
            j += 1
        if ss[i] in MIN_BLOCK_LEN and (j - i) >= MIN_BLOCK_LEN[int(ss[i])]:
            blk[i:j] = nb
            nb += 1
        i = j
    return blk


def block_pair_stats(blk_tok: np.ndarray, cb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For the blocks present among the tokens: (block ids [B], min CB distance [B,B], n6 = CB pairs < 6 A [B,B]).
    ``blk_tok`` [I] block id per token (-1 none), ``cb`` [I,3] CB coordinates (NaN = unknown, ignored)."""
    ids = np.unique(blk_tok[blk_tok >= 0])
    B = len(ids)
    dmin = np.full((B, B), np.inf)
    n6 = np.zeros((B, B), dtype=np.int64)
    if B == 0:
        return ids, dmin, n6
    ok = np.isfinite(cb).all(-1)
    idx = [np.flatnonzero((blk_tok == b) & ok) for b in ids]
    for a in range(B):
        for b in range(a + 1, B):
            if idx[a].size == 0 or idx[b].size == 0:
                continue
            d = np.linalg.norm(cb[idx[a]][:, None, :] - cb[idx[b]][None, :, :], axis=-1)
            dmin[a, b] = dmin[b, a] = float(d.min())
            n6[a, b] = n6[b, a] = int((d < CONTACT_CUTOFF).sum())
    return ids, dmin, n6


def block_states(dmin: np.ndarray, n6: np.ndarray) -> np.ndarray:
    """[B,B] ground-truth block-pair state: CONTACT / NON_CONTACT (MASKED where unknown and on the diagonal)."""
    st = np.where(dmin < CONTACT_CUTOFF, ADJ_CONTACT, ADJ_NON_CONTACT).astype(np.int64)
    st[~np.isfinite(dmin)] = ADJ_MASKED
    np.fill_diagonal(st, ADJ_MASKED)
    return st


def reveal_block_pairs(gt: np.ndarray, dmin: np.ndarray, n6: np.ndarray, mode: str, rng: np.random.Generator,
                       p_hot: float = 0.5, p_cold: float = 0.5, max_promote: int = 3) -> np.ndarray:
    """[B,B] input state for one example: which block pairs are shown, with HOT / COLD promotion."""
    B = gt.shape[0]
    out = np.full((B, B), ADJ_MASKED, dtype=np.int64)
    if mode in ("uncond", "ss_only") or B < 2:
        return out
    iu, ju = np.triu_indices(B, 1)
    lab = gt[iu, ju]
    known = lab != ADJ_MASKED
    show = known.copy()
    if mode == "sparse":                                 # a random subset of the KNOWN pairs, label-independent (a
        show = known & (rng.random(len(lab)) < rng.uniform(0.1, 0.9))   # contact-biased reveal made "hidden" mean "probably non-contact")
    st = np.where(show, lab, ADJ_MASKED)
    # HOT: shown contacts with a large interface
    con = np.flatnonzero(show & (lab == ADJ_CONTACT))
    if con.size and rng.random() < p_hot:
        thr = max(HOT_MIN_N6, float(np.quantile(n6[iu[con], ju[con]], HOT_QUANTILE)))   # quantile over SHOWN contacts only
        cand = con[n6[iu[con], ju[con]] >= thr]
        if cand.size:
            k = int(rng.integers(1, max_promote + 1))
            st[rng.choice(cand, size=min(k, cand.size), replace=False)] = ADJ_HOT
    # COLD: shown non-contacts that are clearly apart
    non = np.flatnonzero(show & (lab == ADJ_NON_CONTACT))
    if non.size and rng.random() < p_cold:
        cand = non[dmin[iu[non], ju[non]] >= COLD_MIN_DIST]
        if cand.size:
            k = int(rng.integers(1, max_promote + 1))
            st[rng.choice(cand, size=min(k, cand.size), replace=False)] = ADJ_COLD
    out[iu, ju] = st
    out[ju, iu] = st
    return out


def tokens_from_blocks(blk_tok: np.ndarray, ids: np.ndarray, bstate: np.ndarray) -> np.ndarray:
    """Expand a [B,B] block-pair state to token pairs [I,I]; tokens outside blocks and same-block pairs are MASKED."""
    I = len(blk_tok)
    pos = np.full(I, -1, dtype=np.int64)
    if ids.size:
        lut = {int(b): k for k, b in enumerate(ids)}
        has = blk_tok >= 0
        pos[has] = [lut[int(b)] for b in blk_tok[has]]
    adj = np.full((I, I), ADJ_MASKED, dtype=np.int64)
    m = pos >= 0
    if m.any():
        sub = bstate[np.ix_(pos[m], pos[m])]
        adj[np.ix_(m, m)] = sub
    return adj


def reveal_ss(ss: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Residue-level C_SEC values for one example (0 = not shown). ``sparse``: each SS segment shown with p ~ U(0.3, 1)."""
    if mode in ("uncond", "adj_only"):
        return np.zeros_like(ss)
    if mode in ("ss_only", "full"):
        return ss.copy()
    out = np.zeros_like(ss)
    p = rng.uniform(0.3, 1.0)
    i, n = 0, len(ss)
    while i < n:
        j = i + 1
        while j < n and ss[j] == ss[i]:
            j += 1
        if rng.random() < p:
            out[i:j] = ss[i:j]
        i = j
    return out


def scorer_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Virtual CB with the RFD3 panel scorer's coefficients (eval_common.reconstruct_cb). Used ONLY by the compactness gate so that it
    measures exactly what the panel's split detector measures ; ADJ labels keep virtual_cb (RFD1 / R1 convention)."""
    b = ca - n
    cc = c - ca
    return -0.57910144 * np.cross(b, cc) + 0.5689693 * b - 0.5441217 * cc + ca


def compactness(ca: np.ndarray, cb: np.ndarray) -> dict:
    """rg_n04, lr_cb, split of one chain (CA, ideal CB [N,3], no NaN), same definitions as the RFD3 scorer's geometry_metrics."""
    n = len(ca)
    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean()))
    d = np.linalg.norm(cb[:, None] - cb[None], axis=-1)
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None])
    iu = np.triu_indices(n, 1)
    lr = float(((d[iu] < LR_CUT) & (sep[iu] > LR_SEP)).sum() / n)
    c6 = ((d < LR_CUT) & (sep >= 6)).astype(np.int64)
    P = np.cumsum(np.cumsum(c6, axis=0), axis=1)
    vals = [(P[k - 1, n - 1] - P[k - 1, k - 1]) / min(k, n - k) for k in range(int(np.ceil(0.25 * n)), int(0.75 * n) + 1) if 1 <= k < n]
    return {"rg_n04": rg / n ** 0.4, "lr_cb": lr, "split": float(min(vals)) if vals else float("nan")}


def is_compact(m: dict) -> bool:
    return not (m["split"] < T_SPLIT or m["lr_cb"] < T_LR or m["rg_n04"] > T_RG)


def dense_zero(adj: np.ndarray, elig: np.ndarray) -> np.ndarray:
    """RFD3 'dense-zero' dialect on a revealed map: every MASKED pair between eligible tokens (loops, same block, diagonal) becomes
    NON_CONTACT. Trains the model that zeros off the block pairs carry no geometry (the old benchmark / user-spec dialect)."""
    out = adj.copy()
    m = elig[:, None] & elig[None, :] & (out == ADJ_MASKED)
    out[m] = ADJ_NON_CONTACT
    return out


def sample_mode(rng: np.random.Generator, forced: str | None = None) -> str:
    return forced if forced is not None else str(rng.choice(MODES, p=MODE_P))
