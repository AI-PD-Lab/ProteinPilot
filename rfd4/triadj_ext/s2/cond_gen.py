"""TRIAD-R S2 CONTROLLABILITY: does the interaction stream steer REAL sampling from pure noise?

For N held-out single-chain examples of length 70-130 (strict ``RingDataset`` val split, the rows of ``oracle_s2``; one design per example and stream condition, at the example's OWN length) the
real sampler (``fm.full_simulation``: 60 steps, recycle schedule ``thinking_medium``, latent TSR k=1.5 = the reference sampler of N003/S2) is run from pure noise with the SAME noise seed in every
condition (common random numbers; the label reveal is drawn inside ``fork_rng``: ``TRIADR_RING_ISOLATE_RNG=1``) under these stream conditions:

    masked     the stream stays all-MASK (control)                                  TRIADR_RING_SAMPLER=masked
    oracle     the example's TRUE RING map is shown from step 0, never regenerated  TRIADR_RING_SAMPLER=oracle     (``ring_state_gt`` = true map)
    shuffled   the true map of a DIFFERENT chain: another sequence cluster, the closest length AMONG DONORS AT LEAST AS LONG as the receiver, cropped to the receiver's length (so every receiver pair
               lies inside the donor's chain: the same support as the oracle map; a receiver without such a donor is left out of the shuffled cells and counted)
    generated  labels sampled by the model during sampling (``--gen-temps``: TRIADR_RING_TEMP, 0 = argmax; default 0.5)    TRIADR_RING_SAMPLER=generated

The design and the native chain have the same residue count and are residue-index aligned (the i-th token of the de-novo batch is the i-th token of the native chain, which is the index the RING map
is defined on). Per design, against the native CA trace (resolved residues only): ``rmsd`` (Kabsch, Angstrom), ``tm`` (TM-score normalised by the NATIVE length = ALL native tokens, unresolved residues contribute 0, d0 from that length; fixed residue
alignment, the fragment + iterative search of the TMscore program over the resolved residues), ``lddt`` (CA lDDT, superposition-free), ``cf1`` / ``cf1_lr`` (F1 of the CA-CA <= 8 A contact sets, |i-j| >= 3 / >= 12), and ``sat_true`` (share of the native map's
positive pairs, state < 22, whose design CA-CA <= 10 A); for the conditions that show a map also ``sat_shown`` (the same against the map that was SHOWN: the donor's for ``shuffled``, the generated one for
``generated``). The native chain itself gives the reference level of ``sat``. Cells (95 % CIs from a cluster bootstrap, units = SEQUENCE clusters as in oracle_s2): per condition means, and paired
differences oriented so that POSITIVE = better:  oracle - masked ("the true map steers"), oracle - shuffled ("the true map beats a wrong one"), shuffled - masked, generated - masked, generated - shuffled.
Expected if the stream is a real interface: oracle >> masked ~ shuffled on tm / rmsd.

SPARSE maps (the realistic use; conditions ``oracle_frac<k>`` k in {5,10,25,50}, ``hotspot<n>`` n in {3,8,20}, ``hotspot<n>_neg``, ``edited``; ``--conditions ...,sparse`` adds all eleven; sampler mode ``pinned``): ONLY
the revealed pairs are shown, from step 0 to the end, with their true labels; every other pair stays MASK for the whole run (never revealed, never generated, never NO_INTERACTION). ``oracle_frac<k>``: a random
k % of the chain's true positive pairs (states 0..21; one seed-fixed permutation per chain, so 5 % c 10 % c 25 % c 50 %). ``hotspot<n>``: the n true positive pairs with the largest |i-j| (>= 12; ties by class
strength: disulfide, metal, salt bridge, ..., hbond, ..., hydrophobic, vdw). ``hotspot<n>_neg``: the same plus 5n random pairs shown as NO_INTERACTION where the native pair is a true non-contact (|i-j| >= 12,
native CA-CA > 12 A, both resolved, not a labelled contact). ``edited``: the full true map (oracle display) with the single strongest long-range contact (strongest class among |i-j| >= 12 pairs whose native
CA-CA <= 8 A, then the largest separation) set to NO_INTERACTION. Note: a positives-only sparse map is a reveal pattern the training distribution (uniformly random reveal over ALL pairs, ~95 % NO_INTERACTION) never produced.
Extra terms (CA-CA <= 8 A): ``sat8_rev`` the SHOWN contacts, ``sat8_non`` the true contacts that were not shown (all of them for ``masked``), ``sat8_removed`` the pair removed in ``edited`` (reported for EVERY condition),
``sat8_oth`` the other true contacts. ``--nrep K`` (with ``--rep-conds``) gives conditions K designs per chain with shared noise seeds (replicate r: seed + 100003 r in every condition); per-chain metrics are
replicate means and ``rep_spread`` reports the within-chain SD against the between-chain SD.

Limitation of the ``generated`` condition at ``TRIADR_RING_LAG >= 1``: the reveal schedule is frac = t_ca - lag, so every network prediction during sampling sees ZERO revealed pairs; the exported
final label map is completed by the last argmax step AFTER the last prediction and cannot steer it. (Default lag 0.1; this run does not change it.)

Offline re-scoring WITHOUT the sampler (the npz keeps every design CA trace): ``python -m triadj_ext.s2.cond_gen recompute --npz <run>.npz --out <run>_rescored.json`` (``recompute_metrics``):
RMSD / TM / lDDT / contact F1 are recomputed with the definitions above; ``sat_*`` only when the npz carries ``true_map`` (runs of this version on) -- older npz files give NaN for them.

Outputs: ONE json (``--out``, schema ``cond_gen/v2``: args, model, selection accounting, per-chain table, cells, means) and ONE npz (``--npz``): ``ids``, ``clusters``, ``n``, ``offsets``, ``native_ca``,
``native_mask`` and per condition ``ca_<cond>`` [sum n, 3] float32 Angstrom (design CA traces), ``donor_idx`` (-1 = no donor), ``true_map`` (int8, row-major n*n per chain concatenated: 0..22 states, 24 unobserved), ``shown_shuffled`` (the cropped donor maps, same layout, -1 without donor) and,
per generated condition, ``ring_<cond>`` (int8 label maps, row-major n*n per chain, concatenated). stdout lines start with ``CONDGEN``. Needs no scorer. Placement: 1 H100, <= 1 GPU-h (see cond_gen.sbatch); the pure parts are unit-tested on CPU (tests/test_cond_gen.py).

    python -m triadj_ext.s2.cond_gen --ckpt /w/triadr_s2/A/checkpoints/chk_..-ema.ckpt --arm A --tag A_s2000 --out /out/cond_gen.json --npz /out/cond_gen.npz --n-chains 96
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

from triadj_ext.s2.gates_report import sequence_cluster
from triadj_ext.s2.oracle_s2 import boot_weights, index_units, mean_stat, paired_stat, stat, verdict_of

SCHEMA = "cond_gen/v3"
NO_INT, N_CLS, UNOBS = 22, 23, 24
CONTACT_A, SAT_A, SAT8_A = 8.0, 10.0, 8.0
LONG_SEP, NEG_MIN_CA = 12, 12.0
TYPE_NAMES = ("hbond", "salt_bridge", "ionic", "hydrophobic", "pi_stack", "pi_cation", "disulfide", "vdw", "metal", "halogen", "pi_hbond")  # ring_data.RING_INTERACTION_TYPES: state = type + 11 * inter
COLLAPSE_PRIORITY = ("disulfide", "metal", "salt_bridge", "ionic", "pi_cation", "pi_stack", "pi_hbond", "hbond", "halogen", "hydrophobic", "vdw")  # strongest first (ring_data.COLLAPSE_PRIORITY)
TYPE_RANK = np.array([COLLAPSE_PRIORITY.index(t) for t in TYPE_NAMES], dtype=np.int64)  # rank of a type index; lower = stronger
SPARSE_CONDS = ("oracle_frac5", "oracle_frac10", "oracle_frac25", "oracle_frac50", "hotspot3", "hotspot8", "hotspot20", "hotspot3_neg", "hotspot8_neg", "hotspot20_neg", "edited")
HIGHER_BETTER = {"tm": True, "lddt": True, "cf1": True, "cf1_lr": True, "sat_true": True, "rmsd": False,
                 # sparse-map terms (CA-CA <= 8 A): satisfaction of the REVEALED contacts / of the native contacts that were NOT revealed / of the single contact removed in `edited` / of all other true contacts
                 "sat8_rev": True, "sat8_non": True, "sat8_removed": True, "sat8_oth": True}
METRICS = tuple(HIGHER_BETTER)
MONOMER_CIF = "/opt/rfd4/data/benchmarks/monomer/monomer-short.cif.zst"
SCHEMA_PROTEINA = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"


def log(*a):
    print("CONDGEN", *a, flush=True)


# ------------------------------------------------------------------------------------------------------------------ geometry (numpy)
def _centered(x: np.ndarray, m: np.ndarray):
    c = x[m].mean(0)
    return x - c, c


def kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray | None = None):
    """Rotation R and translation t minimising sum |R P_i + t - Q_i|^2 over the rows (P, Q: [k,3]); proper rotation (det +1)."""
    w = np.ones(len(P)) if w is None else w
    pc, qc = (P * w[:, None]).sum(0) / w.sum(), (Q * w[:, None]).sum(0) / w.sum()
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def rmsd(P: np.ndarray, Q: np.ndarray, mask: np.ndarray) -> float:
    """CA RMSD after the optimal superposition over the ``mask`` residues (design P onto native Q)."""
    p, q = P[mask].astype(np.float64), Q[mask].astype(np.float64)
    if len(p) < 3:
        return float("nan")
    R, t = kabsch(p, q)
    return float(np.sqrt((((p @ R.T + t) - q) ** 2).sum(1).mean()))


def tm_d0(L: int) -> float:
    return 0.5 if L <= 21 else max(0.5, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


def tm_score(P: np.ndarray, Q: np.ndarray, mask: np.ndarray, l_norm: int | None = None) -> float:
    """TM-score of design P against native Q (both [n,3] Angstrom, same residue indexing; ``mask`` = resolved native residues), normalised by the NATIVE chain length
    ``l_norm`` (default: all n native tokens, so unresolved residues count 0 and a perfect fit of 80 resolved residues of 100 scores 0.8; d0 comes from the same length).
    The superposition is maximised with the TMscore-program search over the resolved residues: seeds from the whole set and from contiguous fragments of length Lr, Lr/2, Lr/4, ... (>= 4),
    each refined by up to 20 rounds of re-superposing on the residues closer than a growing cutoff. Fixed residue-index alignment (no gaps)."""
    p, q = P[mask].astype(np.float64), Q[mask].astype(np.float64)
    L = len(p)  # resolved residues: the ones that can be superposed
    if L < 4:
        return float("nan")
    Ln = int(len(Q) if l_norm is None else l_norm)
    d0 = tm_d0(Ln)
    d0s = min(max(d0, 4.5), 8.0)  # search cutoff of TMscore
    best = 0.0

    def tm_of(R, t):
        d = np.sqrt((((p @ R.T + t) - q) ** 2).sum(1))
        return float((1.0 / (1.0 + (d / d0) ** 2)).sum() / Ln), d

    lens = []
    lf = L
    while lf >= 4:
        lens.append(lf)
        if lf == 4:
            break
        lf = max(4, lf // 2)
    for lf in lens:
        step = max(1, lf // 2) if lf < L else 1
        for s0 in range(0, L - lf + 1, step):
            idx = np.arange(s0, s0 + lf)
            R, t = kabsch(p[idx], q[idx])
            for _ in range(20):
                tm, d = tm_of(R, t)
                best = max(best, tm)
                cut = d0s
                sel = np.flatnonzero(d <= cut)
                while len(sel) < 3 and cut < 50:
                    cut += 0.5
                    sel = np.flatnonzero(d <= cut)
                if len(sel) < 3:
                    break
                R2, t2 = kabsch(p[sel], q[sel])
                if np.allclose(R2, R, atol=1e-7) and np.allclose(t2, t, atol=1e-6):
                    break
                R, t = R2, t2
            tm, _ = tm_of(R, t)
            best = max(best, tm)
    return best


def contact_set(ca: np.ndarray, mask: np.ndarray, cutoff: float = CONTACT_A, min_sep: int = 3) -> np.ndarray:
    d = np.sqrt(((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
    n = len(ca)
    ii, jj = np.indices((n, n))
    return (d <= cutoff) & (jj - ii >= min_sep) & mask[:, None] & mask[None, :]


def contact_f1(design: np.ndarray, native: np.ndarray, mask: np.ndarray, min_sep: int = 3, cutoff: float = CONTACT_A) -> float:
    """F1 between the CA-CA <= ``cutoff`` contact sets (i < j, j - i >= min_sep, both residues resolved). NaN without native contacts."""
    a, b = contact_set(design, mask, cutoff, min_sep), contact_set(native, mask, cutoff, min_sep)
    nb, na = int(b.sum()), int(a.sum())
    if nb == 0:
        return float("nan")
    tp = int((a & b).sum())
    if tp == 0:
        return 0.0
    pr, rc = tp / na, tp / nb
    return 2 * pr * rc / (pr + rc)


def lddt_ca(design: np.ndarray, native: np.ndarray, mask: np.ndarray, inc: float = 15.0) -> float:
    """CA lDDT (superposition-free): for native pairs within ``inc`` A (i != j, both resolved) the share of thresholds {0.5, 1, 2, 4} A met by |d_design - d_native|."""
    dn = np.sqrt(((native[:, None] - native[None]) ** 2).sum(-1))
    dd = np.sqrt(((design[:, None] - design[None]) ** 2).sum(-1))
    m = (dn <= inc) & mask[:, None] & mask[None, :] & ~np.eye(len(native), dtype=bool)
    if not m.any():
        return float("nan")
    diff = np.abs(dd - dn)[m]
    return float(np.mean([(diff < t).mean() for t in (0.5, 1.0, 2.0, 4.0)]))


def pair_satisfaction(design: np.ndarray, pairs: np.ndarray, cutoff: float = SAT_A) -> float:
    """Share of the (i, j) rows of ``pairs`` [k,2] whose design CA-CA distance is <= cutoff; NaN for no pairs."""
    if len(pairs) == 0:
        return float("nan")
    d = np.sqrt(((design[pairs[:, 0]] - design[pairs[:, 1]]) ** 2).sum(-1))
    return float((d <= cutoff).mean())


def positive_pairs(gt: np.ndarray, n: int | None = None) -> np.ndarray:
    """Upper-triangle (i<j) pairs whose state is a positive interaction class 0..21 -> [k,2] (the map's labelled contacts)."""
    g = np.asarray(gt)[: n, : n] if n is not None else np.asarray(gt)
    ii, jj = np.where(np.triu((g >= 0) & (g < NO_INT), 1))
    return np.stack([ii, jj], 1)


def _pair_set(pairs: np.ndarray) -> set:
    return {(int(a), int(b)) for a, b in np.asarray(pairs).reshape(-1, 2)}


def _rows(pairs: set) -> np.ndarray:
    return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def design_metrics(design: np.ndarray, native: np.ndarray, mask: np.ndarray, true_pairs: np.ndarray, shown_pairs: np.ndarray | None = None, removed: tuple | None = None) -> dict:
    """Per-design terms. ``shown_pairs`` = the positive pairs of the map that was SHOWN (None: nothing shown / not applicable); ``removed`` = the chain's (i, j, ...) pair of the ``edited`` condition.
    sat_true / sat_shown use the 10 A cutoff (all positives / the shown map); the sat8_* terms use CA-CA <= 8 A: ``sat8_rev`` = the REVEALED contacts, ``sat8_non`` = the true contacts that were NOT revealed
    (all true contacts for a masked design), ``sat8_removed`` = the one pair removed in ``edited`` and ``sat8_oth`` = the other true contacts (NaN without a removed pair)."""
    out = {"rmsd": rmsd(design, native, mask), "tm": tm_score(design, native, mask), "lddt": lddt_ca(design, native, mask),
           "cf1": contact_f1(design, native, mask, 3), "cf1_lr": contact_f1(design, native, mask, 12), "sat_true": pair_satisfaction(design, true_pairs)}
    if shown_pairs is not None:
        out["sat_shown"] = pair_satisfaction(design, shown_pairs)
    tset = _pair_set(true_pairs)
    rset = _pair_set(shown_pairs) if shown_pairs is not None else set()
    out["sat8_rev"] = pair_satisfaction(design, _rows(rset), SAT8_A) if rset else float("nan")  # the shown contacts (true ones for oracle / sparse / edited; the donor's or the generated ones otherwise)
    out["sat8_non"] = pair_satisfaction(design, _rows(tset - rset), SAT8_A) if (tset - rset) else float("nan")
    if removed is not None:
        rp = (int(removed[0]), int(removed[1]))
        out["sat8_removed"] = pair_satisfaction(design, np.array([rp]), SAT8_A)
        rest = tset - {rp}
        out["sat8_oth"] = pair_satisfaction(design, _rows(rest), SAT8_A) if rest else float("nan")
    return out


# ------------------------------------------------------------------------------------------------------------------ donors
def choose_donors(lengths, clusters, usable=None) -> list:
    """Per chain the donor chain of the SHUFFLED condition: another sequence cluster, AT LEAST as long as the receiver (so cropping the donor map to the receiver covers every receiver pair: the
    same support as the oracle map), and among those the closest length (ties: the lowest index). -1 when no such chain exists (the receiver is then left out of the shuffled cells).
    ``usable`` [k] bool: chains that may serve as donor (default all)."""
    k = len(lengths)
    usable = [True] * k if usable is None else list(usable)
    out = []
    for i in range(k):
        cands = [j for j in range(k) if j != i and usable[j] and clusters[j] != clusters[i] and lengths[j] >= lengths[i]]
        out.append(min(cands, key=lambda j: (lengths[j] - lengths[i], j)) if cands else -1)
    return out


def donor_map(donor_gt: np.ndarray, n_recv: int) -> np.ndarray:
    """The donor's label map cropped to the receiver: pair (i, j) takes the donor's (i, j) for i, j < min(n_recv, n_donor); anything beyond the donor (a donor SHORTER than the receiver: not
    produced by ``choose_donors``) would be UNOBS (24) and shown as NO_INTERACTION by the sampler's oracle mode."""
    out = np.full((n_recv, n_recv), UNOBS, dtype=np.int16)
    m = min(n_recv, donor_gt.shape[0])
    out[:m, :m] = donor_gt[:m, :m]
    return out


def shuffled_support(chains: list, donors: list) -> dict:
    """Matched-support accounting of the shuffled condition: chains with / without a donor, the share of receiver pairs that lie inside the donor chain (1.0 by construction), the length gap,
    and the observed positive-pair counts of the receiver's own map vs the cropped donor map."""
    have = [i for i, d in enumerate(donors) if d >= 0]
    gaps, covered, pos_recv, pos_don = [], [], [], []
    for i in have:
        n, nd = chains[i]["n"], chains[donors[i]]["n"]
        gaps.append(nd - n)
        covered.append(min(n, nd) * (min(n, nd) - 1) / (n * (n - 1)))
        pos_recv.append(len(positive_pairs(chains[i]["gt"], n)))
        pos_don.append(len(positive_pairs(donor_map(chains[donors[i]]["gt"], n), n)))
    return {"n_chains": len(chains), "n_with_donor": len(have), "n_skipped_no_donor": len(chains) - len(have), "mean_length_gap": float(np.mean(gaps)) if gaps else float("nan"),
            "max_length_gap": int(max(gaps)) if gaps else 0, "mean_covered_pair_fraction": float(np.mean(covered)) if covered else float("nan"),
            "mean_positive_pairs_receiver": float(np.mean(pos_recv)) if pos_recv else float("nan"), "mean_positive_pairs_donor_cropped": float(np.mean(pos_don)) if pos_don else float("nan")}


# ------------------------------------------------------------------------------------------------------------------ statistics
# ------------------------------------------------------------------------------------------------------------------ sparse maps
def chain_rng(seed: int, chain: int, tag: int) -> np.random.Generator:
    """Seed-fixed per chain: the same subset for every step and replicate of a run."""
    return np.random.default_rng([int(seed), int(chain), int(tag)])


def positive_list(gt: np.ndarray, n: int | None = None):
    """Observed positive pairs (state 0..21) of a map, i < j -> (i [k], j [k], state [k]) in row-major order."""
    g = np.asarray(gt)[: n, : n] if n is not None else np.asarray(gt)
    ii, jj = np.where(np.triu((g >= 0) & (g < NO_INT), 1))
    return ii, jj, g[ii, jj].astype(np.int64)


def type_rank(state) -> np.ndarray:
    """Strength rank of an interaction state (0 = strongest: disulfide, metal, salt bridge, ...); intra and inter classes of a type rank alike."""
    return TYPE_RANK[np.asarray(state, dtype=np.int64) % 11]


def map_from_pairs(n: int, ii, jj, labels) -> np.ndarray:
    """Symmetric int16 map with the given labels on (ii, jj) and 24 (UNOBS / not revealed) everywhere else."""
    m = np.full((n, n), UNOBS, dtype=np.int16)
    m[ii, jj] = labels
    m[jj, ii] = labels
    return m


def frac_order(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random permutation of the chain's observed positive pairs; the k% subsets are its prefixes, so 5 % c 10 % c 25 % c 50 % (nested)."""
    return rng.permutation(len(positive_list(gt)[0]))


def frac_map(gt: np.ndarray, k_pct: float, order: np.ndarray) -> np.ndarray | None:
    """``oracle_frac<k>``: the first ceil(k % of the positives) pairs of ``order`` with their TRUE labels; every other pair is not revealed (stays MASK in the sampler). None without positives."""
    ii, jj, st = positive_list(gt)
    if len(ii) == 0:
        return None
    m = max(1, int(math.ceil(len(ii) * k_pct / 100.0 - 1e-9)))
    sel = order[:m]
    return map_from_pairs(gt.shape[0], ii[sel], jj[sel], st[sel])


def hotspot_pairs(gt: np.ndarray, n_pairs: int, min_sep: int = LONG_SEP) -> np.ndarray:
    """The ``n_pairs`` true positive pairs with the largest sequence separation |i-j| (>= min_sep); ties by class strength (disulfide/metal/salt bridge ... hbond ... hydrophobic, ``TYPE_RANK``), then (i, j)."""
    ii, jj, st = positive_list(gt)
    keep = (jj - ii) >= min_sep
    ii, jj, st = ii[keep], jj[keep], st[keep]
    order = sorted(range(len(ii)), key=lambda k: (-(jj[k] - ii[k]), int(type_rank(st[k])), int(ii[k]), int(jj[k])))[:n_pairs]
    return np.array([[ii[k], jj[k], st[k]] for k in order], dtype=np.int64).reshape(-1, 3)


def neg_order(ca: np.ndarray, mask: np.ndarray, rng: np.random.Generator, min_sep: int = LONG_SEP, min_ca: float = NEG_MIN_CA, gt: np.ndarray | None = None) -> np.ndarray:
    """Candidate NEGATIVE pairs in a random (seed-fixed) order: |i-j| >= min_sep, both residues resolved, the native CA-CA distance > min_ca AND (when ``gt`` is given) not a labelled true contact of the
    map (long side chains can touch at CA-CA > 12 A) -> [k,2]."""
    d = np.sqrt(((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1))
    ok = (d > min_ca) & mask[:, None] & mask[None, :]
    if gt is not None:
        g = np.asarray(gt)
        ok &= ~((g >= 0) & (g < NO_INT))
    ii, jj = np.where(np.triu(ok, min_sep))
    pairs = np.stack([ii, jj], 1)
    return pairs[rng.permutation(len(pairs))]


def hotspot_map(gt: np.ndarray, n_pairs: int, neg: np.ndarray | None = None, neg_ratio: int = 5) -> np.ndarray | None:
    """``hotspot<n>``: only the hotspot pairs revealed (true labels). ``hotspot<n>_neg`` additionally reveals ``neg_ratio * n`` pairs of ``neg`` (random native non-contacts) as NO_INTERACTION (22). None without a hotspot."""
    hp = hotspot_pairs(gt, n_pairs)
    if len(hp) == 0:
        return None
    m = map_from_pairs(gt.shape[0], hp[:, 0], hp[:, 1], hp[:, 2])
    if neg is not None:
        sel = neg[: neg_ratio * len(hp)]
        m[sel[:, 0], sel[:, 1]] = NO_INT
        m[sel[:, 1], sel[:, 0]] = NO_INT
    return m


def strongest_longrange(gt: np.ndarray, ca: np.ndarray, mask: np.ndarray, min_sep: int = LONG_SEP, max_ca: float = SAT8_A):
    """The single strongest long-range true contact: among positive pairs with |i-j| >= min_sep whose NATIVE CA-CA <= ``max_ca`` (resolved both; fallback: any long-range positive pair), the one of the
    strongest class (``TYPE_RANK``), then the largest separation, then (i, j). -> (i, j, state, native CA-CA distance, native_within_cutoff) or None."""
    ii, jj, st = positive_list(gt)
    keep = (jj - ii) >= min_sep
    ii, jj, st = ii[keep], jj[keep], st[keep]
    if len(ii) == 0:
        return None
    d = np.sqrt(((ca[ii] - ca[jj]) ** 2).sum(-1))
    ok = mask[ii] & mask[jj] & (d <= max_ca)
    idx = np.flatnonzero(ok) if ok.any() else np.arange(len(ii))
    k = min(idx, key=lambda k: (int(type_rank(st[k])), -int(jj[k] - ii[k]), int(ii[k]), int(jj[k])))
    return int(ii[k]), int(jj[k]), int(st[k]), float(d[k]), bool(ok[k])


def edited_map(gt: np.ndarray, pair) -> np.ndarray | None:
    """``edited``: the true map with ONE pair (the strongest long-range contact) set to NO_INTERACTION in both triangles; everything else true."""
    if pair is None:
        return None
    m = np.asarray(gt, dtype=np.int16).copy()
    m[pair[0], pair[1]] = NO_INT
    m[pair[1], pair[0]] = NO_INT
    return m


COND_PATTERNS = (("generated", re.compile(r"^generated(?:_t([0-9.]+))?$")), ("frac", re.compile(r"^oracle_frac(\d+)$")), ("hotspot", re.compile(r"^hotspot(\d+)(_neg)?$")),
                 ("masked", re.compile(r"^masked$")), ("oracle", re.compile(r"^oracle$")), ("shuffled", re.compile(r"^shuffled$")), ("edited", re.compile(r"^edited$")))
SAMPLER_OF_KIND = {"masked": "masked", "oracle": "oracle", "shuffled": "oracle", "edited": "oracle", "generated": "generated", "frac": "pinned", "hotspot": "pinned"}


def parse_condition(name: str) -> dict:
    """'hotspot8_neg' -> {'kind': 'hotspot', 'k': 8, 'neg': True, 'sampler': 'pinned', 'cfg': None}; ``<cond>@cfg<w>`` = the same condition with stream classifier-free guidance weight w
    (``TRIADR_STREAM_CFG``; 'base' = the name without the suffix); raises ValueError for an unknown name."""
    mc = re.match(r"^(.+)@cfg([0-9.]+)$", name)
    if mc:
        spec = parse_condition(mc.group(1))
        if spec["kind"] == "masked":
            raise ValueError("a masked stream has nothing to guide: masked@cfg<w> is not a condition")
        return {**spec, "name": name, "cfg": float(mc.group(2)), "base": mc.group(1)}
    for kind, rx in COND_PATTERNS:
        m = rx.match(name)
        if m:
            g = m.groups()
            return {"kind": kind, "name": name, "sampler": SAMPLER_OF_KIND[kind], "k": (float(g[0]) if kind == "frac" else int(g[0]) if kind == "hotspot" else None),
                    "neg": bool(kind == "hotspot" and g[1]), "temp": (float(g[0]) if kind == "generated" and g[0] else None), "cfg": None, "base": name}
    raise ValueError(f"unknown condition {name!r}")


def build_sparse_maps(chains: list, conds: list, seed: int) -> tuple:
    """Per condition the map to SHOW per chain (None = the chain cannot take this condition) for oracle_frac<k>, hotspot<n>[_neg] and edited, plus the per-chain removed pair of ``edited`` (also used as the
    reference pair of sat8_removed in every condition) and the reveal counts. -> (maps {cond: [map|None]}, removed [pair|None], counts {cond: [(n_pos, n_neg)|None]})."""
    E = len(chains)
    removed = [strongest_longrange(c["gt"], c["ca"], c["mask"]) for c in chains]
    orders = [frac_order(c["gt"], chain_rng(seed, i, 0)) for i, c in enumerate(chains)]
    negs = [neg_order(c["ca"], c["mask"], chain_rng(seed, i, 1), gt=c["gt"]) for i, c in enumerate(chains)]
    maps, counts = {}, {}
    for cond in conds:
        spec = parse_condition(cond)
        if spec["kind"] == "frac":
            maps[cond] = [frac_map(c["gt"], spec["k"], orders[i]) for i, c in enumerate(chains)]
        elif spec["kind"] == "hotspot":
            maps[cond] = [hotspot_map(c["gt"], spec["k"], negs[i] if spec["neg"] else None) for i, c in enumerate(chains)]
        elif spec["kind"] == "edited":
            maps[cond] = [edited_map(c["gt"], removed[i][:2] if removed[i] else None) for i, c in enumerate(chains)]
        else:
            continue
        counts[cond] = [None if m is None else (len(positive_list(m)[0]), int((np.triu(m == NO_INT, 1)).sum())) for m in maps[cond]]
    return maps, removed, counts


# ------------------------------------------------------------------------------------------------------------------ statistics
def cells_of(tab: dict, conds: list, unit_idx: np.ndarray, W: np.ndarray) -> list:
    """tab[metric][cond] = [E] float arrays (NaN = missing). Paired differences oriented so that POSITIVE = better, over the chains where both conditions are finite."""
    pairs = [("oracle", "masked"), ("oracle", "shuffled"), ("shuffled", "masked")] + [(g, "masked") for g in conds if g.startswith("generated")] \
        + [(g, "shuffled") for g in conds if g.startswith("generated")] + [(g, "oracle") for g in conds if g.startswith("generated")]
    for c in conds:  # stream-CFG variants: vs their w = 1 base (what the guidance adds) and vs masked
        sp = parse_condition(c)
        if sp["cfg"] is not None:
            pairs += [(c, sp["base"]), (c, "masked")]
    for c in conds:  # sparse conditions: vs the masked control, vs the full true map ("how much of the oracle gain is kept"), the negative-constraint variant vs its positive-only twin, edited vs oracle
        kind = parse_condition(c)["kind"]
        if kind in ("frac", "hotspot"):
            pairs += [(c, "masked"), (c, "oracle")]
            if c.endswith("_neg"):
                pairs.append((c, c[: -len("_neg")]))
        elif kind == "edited":
            pairs += [("edited", "oracle"), ("edited", "masked")]
    cells = []
    for m in METRICS:
        if m not in tab:
            continue
        for a, b in pairs:
            if a not in tab[m] or b not in tab[m]:
                continue
            x, y = tab[m][a], tab[m][b]
            if not HIGHER_BETTER[m]:
                x, y = -x, -y  # lower is better: a better by (y - x) -> positive
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            delta, rel, n = paired_stat(x, y, unit_idx, W, ok)
            cell = {"term": m, "a": a, "b": b, "delta": delta, "n": n, "n_units": int(len(np.unique(unit_idx[ok]))), "better_if": "higher" if HIGHER_BETTER[m] else "lower"}
            if (a, b) == ("oracle", "masked") and "shuffled" in tab[m]:
                z = tab[m]["shuffled"] if HIGHER_BETTER[m] else -tab[m]["shuffled"]
                ok3 = ok & np.isfinite(z)
                if ok3.any():
                    cell["delta"], _, cell["n"] = paired_stat(x, y, unit_idx, W, ok3)
                    margin, _, _ = paired_stat(x, z, unit_idx, W, ok3)  # oracle beats shuffled?
                    cell["margin_vs_shuffled"] = margin
                    cell["verdict"] = verdict_of(cell["delta"], margin)
            cells.append(cell)
    return cells


def cell_line(c: dict) -> str:
    ci = c["delta"]["ci95"]
    s = (f"CONDGEN cell term={c['term']} {c['a']}-{c['b']} (+=better, {c['better_if']} is better) delta={c['delta']['value']:.4g} "
         f"ci=[{ci[0]:.4g},{ci[1]:.4g}] n={c['n']} units={c['n_units']}" if ci else
         f"CONDGEN cell term={c['term']} {c['a']}-{c['b']} delta={c['delta']['value']} n={c['n']}")
    return s + (f" margin_vs_shuffled={c['margin_vs_shuffled']['value']:.4g} verdict={c['verdict']}" if "margin_vs_shuffled" in c and c["margin_vs_shuffled"]["value"] is not None else "")


def means_of(tab: dict, conds: list, unit_idx: np.ndarray, W: np.ndarray) -> dict:
    out = {}
    for m, per in tab.items():
        out[m] = {}
        for c in conds:
            if c in per and np.isfinite(per[c]).any():
                s = mean_stat(per[c], unit_idx, W)
                out[m][c] = [s["value"]] + (s["ci95"] or [None, None])
    return out


# ------------------------------------------------------------------------------------------------------------------ selection
def chain_facts(full: dict, k: int, lmin: int, lmax: int, min_pos: int, min_resolved: float, allow_multichain: bool):
    """Is row ``k`` of an unpacked val batch a usable single chain? -> (ok, reason, facts). ``full``: plain dict of tensors (cpu)."""
    pm = full["padding_mask"][k].bool()
    n = int(pm.sum())
    if not (lmin <= n <= lmax):
        return False, "length", None
    ch = full["chain_iid"][k][pm]
    if not allow_multichain and int(torch.unique(ch).numel()) != 1:
        return False, "multichain", None
    gt = full["ring_state_gt"][k][:n, :n].long()
    obs = full["ring_obs"][k][:n, :n].bool()
    obs = torch.triu(obs, 1)
    obs = obs | obs.T
    gt = torch.triu(gt.clamp(0, UNOBS), 1)
    gt = gt + gt.T
    gt = torch.where(obs, gt, torch.full_like(gt, UNOBS))
    npos = int(((gt < NO_INT) & obs).sum()) // 2
    if npos < min_pos:
        return False, "few_positive_pairs", None
    ca = full["coords"][k][:n, 1, :].float().numpy()
    res = (full["atom37_mask"][k][:n, 1].bool() & (full["is_resolved_mask"][k][:n].bool() if "is_resolved_mask" in full else True)).numpy()
    if res.mean() < min_resolved:
        return False, "unresolved", None
    return True, "ok", {"n": n, "gt": gt.numpy().astype(np.int16), "ca": ca, "mask": res, "n_pos_pairs": npos}


# ------------------------------------------------------------------------------------------------------------------ model / sampler
def load_model(args, device):
    """Arm check FIRST (``model_s2.guard_arm_for_args``: a Z request on an A checkpoint, C on A, ... raises before anything is built), then the same loading as the oracle matrix
    (EMA checkpoint, arm class registered first, or a bundle)."""
    from triadj_ext.model_s2 import guard_arm_for_args
    from triadj_ext.s2 import oracle_s2

    guard = guard_arm_for_args(args.arm, ckpt=args.ckpt or "", bundle=args.bundle or "")
    model, meta = oracle_s2.load_model(args, device)
    meta = {**meta, "arm_check": guard["verified"], "arm_declared": guard["declared"]}
    return model, meta


class DeNovoBatches:
    """De-novo (fully diffused) inference batches of ANY length through the real inference pipeline: the benchmark monomer CIF (one expandable-segment residue) with its
    ``annotation_expseg{min,max}_1_residue`` set to the requested length. At L=100 the collated batch equals the one ``generate`` builds from ``monomer-short.cif.zst``."""

    def __init__(self, cif: str = MONOMER_CIF):
        from rfproteina.datapipes.pipeline import build_design_validation_pipeline
        from rfproteina.inference_utils.datasets import DesignProblemDataset

        self.pipe = build_design_validation_pipeline(metrics_tags=None)
        self.ds = DesignProblemDataset(cif, transform=None)
        self.base = self.ds._load_from_path(self.ds.file_paths[0])

    def item(self, name: str, length):
        """One de-novo item. ``length`` an int = ONE chain of that many residues (the monomer path, unchanged); a sequence of ints (L1, L2, ...) = a COMPLEX of that many chains A, B, ... (token order: every residue of
        chain A, then chain B ...; the sentinel of each chain is a separate expandable segment, so the real pipeline gives each chain its own ``chain_iid`` and no covalent path between them)."""
        from atomworks.ml.transforms.base import TransformedDict

        data = TransformedDict(dict(self.base))
        arr = self.base["atom_array"].copy()
        if isinstance(length, (tuple, list)):
            lens = [int(x) for x in length]
            if len(lens) < 1 or any(x < 1 for x in lens):
                raise ValueError(f"chain lengths must be positive integers, got {length!r}")
            if len(lens) > 1:
                from atomworks.ml.utils.expandable_segment import insert_expandable_segment

                for k in range(1, len(lens)):
                    arr = insert_expandable_segment(arr, seg_min=lens[k], seg_max=lens[k], new_chain=True, new_chain_id=chr(ord("A") + k), new_chain_type="POLYPEPTIDE(L)")
                dt = arr.get_annotation("annotation_expsegmin_1_residue").dtype
                for c in ("annotation_expsegmin_1_residue", "annotation_expsegmax_1_residue"):
                    arr.set_annotation(c, np.array(lens, dtype=dt))
            else:
                length = lens[0]
        if not isinstance(length, (tuple, list)):
            for c in ("annotation_expsegmin_1_residue", "annotation_expsegmax_1_residue"):
                arr.set_annotation(c, np.full(arr.array_length(), int(length), dtype=arr.get_annotation(c).dtype))
        data["atom_array"] = arr
        data["example_id"] = name
        data["task"] = "denovo"
        return self.pipe(data)

    def batch(self, names: list, lengths: list, keep_atom_array: bool = False):
        """Collated batch (a PaddedBatchedDataSample). ``lengths`` entries are ints (one chain) or (L1, L2, ...) tuples (a complex). ``keep_atom_array=True`` keeps the input atom arrays: needed by the all-atom decode
        of ``VanillaStrategy.generate`` (design_with_contacts)."""
        from rfproteina.datapipes.collate import collate_batch

        b = collate_batch([self.item(nm, L) for nm, L in zip(names, lengths)], schema=SCHEMA_PROTEINA, fixed_dimensions=None, keep_atom_array=keep_atom_array)
        got = b["padding_mask"].sum(1).tolist()
        want = [int(sum(L)) if isinstance(L, (tuple, list)) else int(L) for L in lengths]
        if got != want:
            raise RuntimeError(f"de-novo batch lengths {got} != requested {want}")
        return b


def compose_inference_cfg(configs_dir: str, overrides: list):
    """The REAL inference config group ``generation`` (sampler settings, recycle schedule) with Hydra overrides, composed without running anything."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath(configs_dir), version_base="1.3"):
        cfg = compose(config_name="inference", overrides=["inputs=x", *overrides])
    return cfg.generation


def stream_batch(base_batch, device, gts: list | None):
    """Plain dict of tensors on ``device`` for one condition: the de-novo batch plus (for oracle/shuffled) ``ring_state_gt`` [B,N,N] int16 with the maps ``gts`` (per row [n,n] int16, 24 = unobserved)."""
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in base_batch.items()}
    B, N = b["padding_mask"].shape
    if gts is not None:
        gt = torch.full((B, N, N), UNOBS, dtype=torch.int16)
        for r, g in enumerate(gts):
            gt[r, : g.shape[0], : g.shape[1]] = torch.as_tensor(np.asarray(g, dtype=np.int16))
        b["ring_state_gt"] = gt.to(device)
        b["ring_obs"] = (gt < N_CLS).to(device)
    return b


@contextlib.contextmanager
def env_set(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


COND_ENV = {  # condition kind -> TRIADR_* environment of the sampler (generated conditions add TRIADR_RING_TEMP)
    "masked": {"TRIADR_RING_SAMPLER": "masked"},
    "oracle": {"TRIADR_RING_SAMPLER": "oracle"},
    "shuffled": {"TRIADR_RING_SAMPLER": "oracle"},  # the donor's true map is shown exactly like an oracle map
    "edited": {"TRIADR_RING_SAMPLER": "oracle"},    # the true map with one contact removed: shown like an oracle map
    "generated": {"TRIADR_RING_SAMPLER": "generated"},
    "frac": {"TRIADR_RING_SAMPLER": "pinned"},      # sparse: ONLY the revealed pairs are shown, everything else stays MASK for the whole run
    "hotspot": {"TRIADR_RING_SAMPLER": "pinned"},
}
KNOB_VARS = ("TRIADR_RING_TEMP", "TRIADR_RING_MIN_PROB", "TRIADR_RING_LAG", "TRIADR_RING_BIAS", "TRIADR_RING_ONLYNO", "TRIADR_RING_ISOLATE_RNG", "TRIADR_STREAM_CFG")


def run_sampler(model, batch: dict, inf_cfg, device, seed: int, amp: bool) -> np.ndarray:
    """One ``full_simulation`` on a de-novo batch (the body of ``VanillaStrategy.generate`` without decoding) -> CA traces [B,N,3] Angstrom (numpy)."""
    model.inf_cfg = inf_cfg
    b = model.preprocess_batch_for_generation(batch)
    b = model.add_clean_samples(b)
    B, N = b["padding_mask"].shape
    nsteps = inf_cfg.args.nsteps
    depths = model.recycle_depths(nsteps, full_nsteps=nsteps)
    torch.manual_seed(seed)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        x, _info = model.fm.full_simulation(batch=b, predict_for_sampling=model.predict_for_sampling, recycle_depths=depths, nsteps=nsteps, nsamples=B, n=N,
                                            sampling_model_args=inf_cfg.model, device=device, save_trajectory_every=0, dump_noisy_trajectories=False,
                                            guidance_w=inf_cfg.args.get("guidance_w", 1.0), guidance_model=None)
    return (x["bb_ca"].float() * 10.0).cpu().numpy()  # nm -> Angstrom


def select_chains(loader, args) -> tuple:
    """Scan the strict val loader; keep the first ``n_chains`` usable single chains of length lmin..lmax. -> (chains, accounting)."""
    from triadj_ext import p0_probe as P0

    chains, seen = [], set()
    acct = {"requested": 0, "failed_to_load": 0, "drop_reasons": {}, "scanned": 0, "duplicates": 0, "length": 0, "multichain": 0, "few_positive_pairs": 0, "unresolved": 0, "ok": 0}
    for raw in loader:
        full, meta = P0.unpack(raw, torch.device("cpu"))
        bsz = full["padding_mask"].shape[0] if "padding_mask" in full else 0
        acct["requested"] += int(meta.get("n_requested") or bsz)
        acct["failed_to_load"] += int(meta.get("n_failed") or 0)
        for r, c in (meta.get("drops") or {}).items():
            acct["drop_reasons"][r] = acct["drop_reasons"].get(r, 0) + int(c)
        ids = list(meta["example_ids"]) if meta["example_ids"] else [f"ex{len(seen) + k}" for k in range(bsz)]
        clu = meta["clusters"] if meta["clusters"] is not None else ids
        for k in range(bsz):
            acct["scanned"] += 1
            if ids[k] in seen:
                acct["duplicates"] += 1
                continue
            seen.add(ids[k])
            ok, why, f = chain_facts(full, k, args.lmin, args.lmax, args.min_pos_pairs, args.min_resolved, args.allow_multichain)
            acct[why] += 1
            if ok:
                f.update(id=ids[k], cluster=sequence_cluster(clu[k], ids[k]), raw_cluster=str(clu[k]))
                chains.append(f)
            if len(chains) >= args.n_chains:
                return chains, acct
    return chains, acct


def write_npz(path: str, chains: list, designs: dict, donors: list, rings: dict, extra_reps: list | None = None, shown: dict | None = None, removed: list | None = None,
              extra_rings: list | None = None) -> None:
    """``designs[cond][i]`` = replicate 0; ``extra_reps[r-1][cond][i]`` = replicate r (saved as ``rep<r>_<cond>``); ``shown[cond][i]`` = the label map that was shown ([n,n] int16, >= 23 = nothing) for the
    sparse / edited conditions (saved as ``shown_<cond>``, int8 n*n per chain, -1 = nothing); ``removed[i]`` = (i, j, state, native CA-CA, native within 8 A) of the strongest long-range contact or None."""
    n = np.array([c["n"] for c in chains])
    offs = np.concatenate([[0], np.cumsum(n)])
    arrays = {"ids": np.array([c["id"] for c in chains]), "clusters": np.array([c["cluster"] for c in chains]), "n": n, "offsets": offs, "donor_idx": np.array(donors),
              "native_ca": np.concatenate([c["ca"] for c in chains]).astype(np.float32), "native_mask": np.concatenate([c["mask"] for c in chains])}
    if all("gt" in c for c in chains):  # the maps behind sat_true / sat_shown (older files lack them: recompute_metrics then leaves sat_* NaN)
        arrays["true_map"] = np.concatenate([np.asarray(c["gt"], dtype=np.int8).reshape(-1) for c in chains])
        arrays["shown_shuffled"] = np.concatenate([np.full(c["n"] ** 2, -1, np.int8) if donors[i] < 0 else donor_map(chains[donors[i]]["gt"], c["n"]).astype(np.int8).reshape(-1)
                                                  for i, c in enumerate(chains)])
    if removed is not None:
        arrays["removed_pair"] = np.array([[-1, -1, -1, np.nan, 0] if r is None else [r[0], r[1], r[2], r[3], float(r[4])] for r in removed], dtype=np.float64)
    for cond, per in (shown or {}).items():
        arrays[f"shown_{cond}"] = np.concatenate([np.full(c["n"] ** 2, -1, np.int8) if per[i] is None else np.where(np.asarray(per[i]) >= N_CLS, -1, per[i]).astype(np.int8).reshape(-1)
                                                  for i, c in enumerate(chains)])
    for cond, per in designs.items():
        arrays[f"ca_{cond}"] = np.concatenate([np.full((c["n"], 3), np.nan, np.float32) if per[i] is None else per[i].astype(np.float32) for i, c in enumerate(chains)])
    for r, d in enumerate(extra_reps or [], start=1):
        for cond, per in d.items():
            if any(x is not None for x in per):
                arrays[f"rep{r}_{cond}"] = np.concatenate([np.full((c["n"], 3), np.nan, np.float32) if per[i] is None else per[i].astype(np.float32) for i, c in enumerate(chains)])
    for cond, per in rings.items():
        arrays[f"ring_{cond}"] = np.concatenate([np.full(c["n"] ** 2, -1, np.int8) if per[i] is None else per[i].astype(np.int8).reshape(-1) for i, c in enumerate(chains)])
    for r, d in enumerate(extra_rings or [], start=1):  # the GENERATED map of every replicate (the label draws differ per replicate): ring_rep<r>_<cond>
        for cond, per in d.items():
            if any(x is not None for x in per):
                arrays[f"ring_rep{r}_{cond}"] = np.concatenate([np.full(c["n"] ** 2, -1, np.int8) if per[i] is None else per[i].astype(np.int8).reshape(-1) for i, c in enumerate(chains)])
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)


def score_all(natives: list, masks: list, designs: dict, true_pairs: list, shown_pairs: dict, conds: list, removed: list | None = None) -> dict:
    """tab[metric][cond] = float [E] (NaN where a design is missing). ``designs[cond][i]`` [n_i,3] or None; ``shown_pairs[cond][i]`` [k,2] or None; ``removed[i]`` the chain's edited-pair tuple or None."""
    E = len(natives)
    tab = {m: {c: np.full(E, np.nan) for c in conds} for m in METRICS + ("sat_shown",)}
    for i in range(E):
        for cond in conds:
            d = designs[cond][i]
            if d is None or not np.isfinite(d).all():
                continue
            res = design_metrics(d, natives[i], masks[i], true_pairs[i], shown_pairs[cond][i], removed[i] if removed is not None else None)
            for m, v in res.items():
                tab[m][cond][i] = v
    return tab


def score_reps(natives: list, masks: list, designs_r: list, true_pairs: list, shown_pairs_r: list, conds: list, removed: list | None = None) -> tuple:
    """Per-chain terms averaged over the replicates (rep 0 only where a condition has one) and the replicate spread: for conditions with >= 2 replicates per chain, the mean over chains of the
    within-chain SD (ddof 1) next to the SD across chain means. -> (tab [metric][cond] -> [E], spread {metric: {cond: {...}}})."""
    tabs = [score_all(natives, masks, designs_r[r], true_pairs, shown_pairs_r[r], conds, removed) for r in range(len(designs_r))]
    tab, spread = {}, {}
    for m in tabs[0]:
        tab[m], spread[m] = {}, {}
        for c in conds:
            stack = np.stack([t[m][c] for t in tabs])  # [R, E]
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tab[m][c] = np.nanmean(stack, 0)
                    nr = np.isfinite(stack).sum(0)
                    if len(tabs) > 1 and (nr >= 2).any():
                        sd = np.nanstd(stack, 0, ddof=1)
                        spread[m][c] = {"within_chain_sd": float(np.nanmean(np.where(nr >= 2, sd, np.nan))), "between_chain_sd": float(np.nanstd(tab[m][c], ddof=1)),
                                        "n_chains": int((nr >= 2).sum()), "n_reps": int(nr.max())}
        spread[m] = {c: v for c, v in spread[m].items()}
    return tab, spread


def score_run(natives: list, masks: list, designs_r: list, true_pairs: list, shown_maps_r: list, removed: list | None, conds: list) -> tuple:
    """The scoring of a run: ``shown_maps_r[r][cond][i]`` = the label map shown to replicate r (per replicate: generated labels differ) -> per-chain terms averaged over replicates + spread.
    Used by ``run`` and by the tests that compare a run with ``recompute_metrics``."""
    shown_pairs_r = [{c: [_shown_pairs_of_map(m) for m in shown_maps_r[r][c]] for c in conds} for r in range(len(designs_r))]
    return score_reps(natives, masks, designs_r, true_pairs, shown_pairs_r, conds, removed)


def summarise_tables(tab: dict, conds: list, clusters: list, n_boot: int, seed: int) -> dict:
    unit_idx, n_units = index_units(clusters)
    W = boot_weights(unit_idx, n_boot, seed)
    return {"n_units": int(n_units), "cells": cells_of({m: tab[m] for m in METRICS}, conds, unit_idx, W), "means": means_of(tab, conds, unit_idx, W)}


def _shown_pairs_of_map(m) -> np.ndarray | None:
    return None if m is None else positive_pairs(np.where((np.asarray(m) >= 0) & (np.asarray(m) < N_CLS), m, UNOBS))


def recompute_metrics(npz_path: str, n_boot: int = 1000, seed: int = 1234, out: str | None = None) -> dict:
    """Re-score the saved CA traces of a cond_gen run WITHOUT the sampler, with the definitions of this module (TM-score normalised by the native length, ...).
    Reads ``native_ca`` / ``native_mask`` / ``ca_<cond>`` (+ replicates ``rep<r>_<cond>``) / ``clusters`` / ``offsets`` and, for the sat terms when present, ``true_map`` / ``shown_shuffled`` / ``shown_<cond>`` /
    ``ring_<cond>`` / ``removed_pair``; older files lack some of them and give NaN for the terms that need them."""
    import re as _re

    z = np.load(npz_path, allow_pickle=False)
    offs, ids, clusters = z["offsets"], [str(x) for x in z["ids"]], [str(x) for x in z["clusters"]]
    E = len(ids)
    conds = [k[3:] for k in z.files if k.startswith("ca_")]
    sl = [slice(int(offs[i]), int(offs[i + 1])) for i in range(E)]
    nat, msk = [z["native_ca"][s_].astype(np.float64) for s_ in sl], [z["native_mask"][s_].astype(bool) for s_ in sl]
    reps = sorted({int(m.group(1)) for k in z.files for m in [_re.match(r"^rep(\d+)_", k)] if m})
    designs_r = [{c: [z[f"ca_{c}"][s_].astype(np.float64) for s_ in sl] for c in conds}]
    for r in reps:
        designs_r.append({c: ([z[f"rep{r}_{c}"][s_].astype(np.float64) for s_ in sl] if f"rep{r}_{c}" in z.files else [None] * E) for c in conds})
    n = [int(x) for x in z["n"]]
    has_maps = "true_map" in z.files
    starts = np.concatenate([[0], np.cumsum([m_ * m_ for m_ in n])]).astype(np.int64)
    true_pairs = [positive_pairs(z["true_map"][starts[i]: starts[i + 1]].reshape(n[i], n[i]).astype(np.int16), n[i]) if has_maps else np.zeros((0, 2), int) for i in range(E)]
    removed = None
    if "removed_pair" in z.files:
        rp = z["removed_pair"]
        removed = [None if rp[i, 0] < 0 else (int(rp[i, 0]), int(rp[i, 1]), int(rp[i, 2]), float(rp[i, 3]), bool(rp[i, 4])) for i in range(E)]
    shown_r = [{c: [None] * E for c in conds} for _ in designs_r]
    if has_maps:
        for r, dr in enumerate(designs_r):
            for c in conds:
                for i in range(E):
                    if dr[c][i] is None or not np.isfinite(dr[c][i]).all():
                        continue
                    seg = slice(int(starts[i]), int(starts[i + 1]))
                    if c == "oracle":
                        shown_r[r][c][i] = true_pairs[i]
                    elif c == "shuffled" and "shown_shuffled" in z.files:
                        shown_r[r][c][i] = _shown_pairs_of_map(z["shown_shuffled"][seg].reshape(n[i], n[i]).astype(np.int16))
                    elif f"shown_{c}" in z.files:
                        shown_r[r][c][i] = _shown_pairs_of_map(z[f"shown_{c}"][seg].reshape(n[i], n[i]).astype(np.int16))
                    elif c.startswith("generated"):
                        key = f"ring_{c}" if r == 0 else f"ring_rep{r}_{c}"  # the generated map differs per replicate: rescoring needs THAT replicate's map
                        if key not in z.files:
                            raise ValueError(f"recompute: replicate {r} of condition {c!r} has a design but the npz has no {key!r}: its generated map was not saved (run written before "
                                             f"per-replicate maps); refusing to reuse another replicate's map (sat_shown / sat8_rev would silently change)")
                        shown_r[r][c][i] = _shown_pairs_of_map(z[key][seg].reshape(n[i], n[i]).astype(np.int16))
    tab, spread = score_reps(nat, msk, designs_r, true_pairs, shown_r, conds, removed)
    summ = summarise_tables(tab, conds, clusters, n_boot, seed + 7)
    res = {"schema": SCHEMA + "-rescored", "npz": os.path.abspath(npz_path), "n_chains": E, "conditions": conds, "n_replicates": len(designs_r), "sat_available": bool(has_maps), "ids": ids,
           "per_chain": {m: {c: [None if not np.isfinite(x) else float(x) for x in tab[m][c]] for c in conds} for m in tab}, "rep_spread": spread, **summ}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return res


def cond_names(args) -> list:
    """Condition list of a run: ``generated`` -> one condition per label temperature, ``sparse`` -> SPARSE_CONDS; every non-masked condition is repeated as ``<cond>@cfg<w>`` for each stream-CFG weight
    w != 1 of ``--cfg-w`` (optionally only for ``--cfg-conds``); duplicates dropped, order kept."""
    base = []
    for c in args.conditions_list:
        base += [f"generated_t{t:g}" for t in args.gen_temps_list] if c == "generated" else list(SPARSE_CONDS) if c == "sparse" else [c]
    base = list(dict.fromkeys(base))
    ws = [w for w in getattr(args, "cfg_w_list", [1.0])]
    only = getattr(args, "cfg_conds_list", [])
    out = []
    for c in base:
        out.append(c)
        if parse_condition(c)["kind"] != "masked" and (not only or c in only):  # stream CFG: every stream condition once more per w != 1 (masked has nothing to guide)
            out += [f"{c}@cfg{w:g}" for w in ws if w != 1.0]
    return list(dict.fromkeys(out))


def run(args) -> dict:
    t_start = time.time()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
    amp = device.type == "cuda"
    model, meta = load_model(args, device)
    log("init " + " ".join(f"{k}={v}" for k, v in meta.items()) + f" device={device} load_s={time.time() - t_start:.0f}")
    from triadj_ext.s2 import oracle_s2

    loader, info = oracle_s2.build_loader(args)
    chains, acct = select_chains(loader, args)
    log(f"selected {len(chains)}/{args.n_chains} chains of length {args.lmin}-{args.lmax}; accounting={json.dumps(acct, sort_keys=True).replace(' ', '')}")
    if len(chains) < 2:
        raise SystemExit("CONDGEN error: fewer than 2 usable chains (raise --n-val / widen --lmin --lmax)")
    if len(chains) < args.n_chains:
        log(f"WARN only {len(chains)} of the requested {args.n_chains} chains were found in {acct['scanned']} scanned rows: raise --n-val (the cells use what was found)")
    lengths, clusters = [c["n"] for c in chains], [c["cluster"] for c in chains]
    donors = choose_donors(lengths, clusters)
    support = shuffled_support(chains, donors)
    log("shuffled donors (other sequence cluster, length >= receiver, cropped): " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in support.items()))
    conds = cond_names(args)
    specs = {c: parse_condition(c) for c in conds}
    sparse_maps, removed, counts = build_sparse_maps(chains, conds, args.seed)
    for c in conds:
        if c in sparse_maps:
            have = [x for x in counts[c] if x is not None]
            log(f"sparse condition {c}: {len(have)}/{len(chains)} chains eligible; mean revealed positives={np.mean([x[0] for x in have]) if have else float('nan'):.1f} mean revealed NO_INTERACTION={np.mean([x[1] for x in have]) if have else float('nan'):.1f}")
    log(f"removed contact (edited / sat8_removed): {sum(r is not None for r in removed)}/{len(chains)} chains have a long-range true contact; within 8 A natively: {sum(bool(r[4]) for r in removed if r is not None)}")
    nrep_of = {c: (args.nrep if (not args.rep_conds_list or c in args.rep_conds_list) else 1) for c in conds}
    n_cfg = sum(1 for c in conds if specs[c]["cfg"] is not None)
    if n_cfg:
        log(f"stream CFG: {n_cfg} conditions with w != 1 run 2 denoiser passes per step (the pass with the all-MASK stream is skipped while nothing is shown): "
            f"~{(len(conds) + n_cfg) / len(conds):.2f}x the denoiser cost of the same conditions without guidance")
    n_rep = max(nrep_of.values())
    gen_over = [f"generation.model.local_latents.simulation_step_params.tsr_k={args.tsr_k}", *args.gen_overrides.split()]
    inf_cfg = compose_inference_cfg(args.configs_dir, gen_over)
    nsteps = int(inf_cfg.args.nsteps)
    log(f"sampler: nsteps={nsteps} tsr_k_latents={inf_cfg.model.local_latents.simulation_step_params.tsr_k} recycle_schedule={'yes' if inf_cfg.get('n_recycle_schedule') else 'const ' + str(inf_cfg.get('n_recycle'))} "
        f"conditions={conds} replicates={ {c: k for c, k in nrep_of.items() if k > 1} or 1 }")
    src = DeNovoBatches(args.denovo_cif)
    from triadj_ext import ring_flow as RF

    E = len(chains)
    order = sorted(range(E), key=lambda i: (lengths[i], i))  # micro-batches of similar length
    designs_r = [{c: [None] * E for c in conds} for _ in range(n_rep)]
    shown_maps_r = [{c: [None] * E for c in conds} for _ in range(n_rep)]  # the label map that was shown (generated: per replicate)
    rings_r = [{c: [None] * E for c in conds if specs[c]["kind"] == "generated"} for _ in range(n_rep)]  # the generated map of EVERY replicate
    for bi in range(0, len(order), args.micro_batch):
        rows = order[bi: bi + args.micro_batch]
        base = src.batch([f"cg{i}" for i in rows], [lengths[i] for i in rows])
        for cond in conds:
            spec = specs[cond]
            kind = spec["kind"]
            env = dict(COND_ENV[kind])
            env.update({k: None for k in KNOB_VARS})
            env["TRIADR_RING_ISOLATE_RNG"] = "1"
            if kind == "generated":
                env["TRIADR_RING_TEMP"] = str(spec["temp"])
            if spec["cfg"] is not None:
                env["TRIADR_STREAM_CFG"] = repr(spec["cfg"])
            gts, usable = None, [True] * len(rows)
            if kind == "oracle":
                gts = [chains[i]["gt"] for i in rows]
            elif kind == "shuffled":
                usable = [donors[i] >= 0 for i in rows]
                gts = [donor_map(chains[donors[i]]["gt"], lengths[i]) if donors[i] >= 0 else chains[i]["gt"] for i in rows]  # donor-less rows are computed but discarded
            elif kind in ("frac", "hotspot", "edited"):
                usable = [sparse_maps[cond][i] is not None for i in rows]
                fill = (lambda i: chains[i]["gt"]) if COND_ENV[kind]["TRIADR_RING_SAMPLER"] == "oracle" else (lambda i: np.full((lengths[i], lengths[i]), UNOBS, np.int16))
                gts = [sparse_maps[cond][i] if sparse_maps[cond][i] is not None else fill(i) for i in rows]  # unusable rows are computed but discarded
            if not any(usable):
                continue
            for r in range(nrep_of[cond]):
                col = None
                if kind == "generated":
                    col = []
                    RF.add_ring_sink(col.append)
                try:
                    with env_set(**env):
                        ca = run_sampler(model, stream_batch(base, device, gts), inf_cfg, device, args.seed + bi + 100003 * r, amp)
                finally:
                    if col is not None:
                        RF.remove_ring_sink(col.append)
                for k, i in enumerate(rows):
                    if not usable[k]:
                        continue
                    designs_r[r][cond][i] = ca[k, : lengths[i]]
                    if kind in ("oracle", "shuffled", "frac", "hotspot", "edited"):
                        shown_maps_r[r][cond][i] = gts[k]
                    elif kind == "generated" and col:
                        rs = col[-1]["ring_state"][k].numpy()[: lengths[i], : lengths[i]]
                        shown_maps_r[r][cond][i] = np.where(rs < N_CLS, rs, UNOBS)
                        rings_r[r][cond][i] = rs
        log(f"progress batch={bi // args.micro_batch + 1}/{math.ceil(len(order) / args.micro_batch)} chains_done={min(bi + args.micro_batch, len(order))} el_s={time.time() - t_start:.0f}")
    # ---- metrics
    natives, masks = [c["ca"] for c in chains], [c["mask"] for c in chains]
    true_pairs = [positive_pairs(c["gt"], c["n"]) for c in chains]
    native_sat = np.array([pair_satisfaction(c["ca"], tp) for c, tp in zip(chains, true_pairs)])
    tab, spread = score_run(natives, masks, designs_r, true_pairs, shown_maps_r, removed, conds)
    summ = summarise_tables(tab, conds, clusters, args.n_boot, args.seed + 7)
    cells, means, n_units = summ["cells"], summ["means"], summ["n_units"]
    removed_info = [None if r is None else {"pair": [r[0], r[1]], "state": r[2], "type": TYPE_NAMES[r[2] % 11], "native_ca": round(r[3], 2), "native_within_8A": r[4]} for r in removed]
    out = {"schema": SCHEMA, "tag": args.tag, "model": meta, "args": {k: v for k, v in vars(args).items() if k != "loader_factory"}, "split": info, "selection": acct,
           "n_chains": E, "n_units": int(n_units), "conditions": conds, "nsteps": nsteps, "shuffled_support": support, "native_sat_true": float(np.nanmean(native_sat)),
           "n_replicates": {c: nrep_of[c] for c in conds}, "rep_spread": spread, "reveal_counts": {c: counts[c] for c in counts},
           "chains": [{"id": c["id"], "cluster": c["cluster"], "n": c["n"], "n_pos_pairs": c["n_pos_pairs"], "donor": (chains[donors[i]]["id"] if donors[i] >= 0 else None), "removed": removed_info[i]}
                      for i, c in enumerate(chains)],
           "per_chain": {m: {c: [None if not np.isfinite(x) else float(x) for x in tab[m][c]] for c in conds} for m in tab}, "cells": cells, "means": means,
           "seconds": time.time() - t_start}
    for c in cells:
        print(cell_line(c), flush=True)
    for m in METRICS:
        log(f"means {m}: " + " ".join(f"{c}={means[m][c][0]:.4g}" for c in conds if c in means.get(m, {})))
    for m in ("tm", "sat8_rev"):
        sp = {c: v for c, v in spread.get(m, {}).items()}
        if sp:
            log(f"replicate spread {m}: " + " ".join(f"{c}: within-chain SD {v['within_chain_sd']:.4f} (between-chain SD {v['between_chain_sd']:.4f}, {v['n_chains']} chains x {v['n_reps']} reps)" for c, v in sp.items()))
    log(f"native_sat_true (reference level of sat_true: native chain vs its own map) = {out['native_sat_true']:.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_npz(args.npz, chains, designs_r[0], donors, rings_r[0], extra_reps=designs_r[1:], extra_rings=rings_r[1:],
              shown={c: sparse_maps[c] for c in sparse_maps}, removed=removed)
    log(f"done out={args.out} npz={args.npz} total_s={time.time() - t_start:.0f}")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default="", help="S2 checkpoint (chk_*-ema.ckpt or a training last.ckpt: its SimpleEMA weights are used)")
    p.add_argument("--bundle", default="", help="alternative: a bundle directory (released model / expanded base bundle = step-0 control)")
    p.add_argument("--arm", required=True, choices=("A", "B", "C", "Z"))
    p.add_argument("--tag", default="cond_gen")
    p.add_argument("--run-id", default="", help="unique id of this run (the sbatch passes the SLURM job id): lets a watchdog address THIS run's processes and no other")
    p.add_argument("--out", default=f"/tmp/{os.environ.get('USER', 'u')}/cond_gen.json")
    p.add_argument("--npz", default=f"/tmp/{os.environ.get('USER', 'u')}/cond_gen.npz")
    p.add_argument("--n-chains", type=int, default=96)
    p.add_argument("--lmin", type=int, default=70)
    p.add_argument("--lmax", type=int, default=130)
    p.add_argument("--min-pos-pairs", type=int, default=10, help="minimum number of observed POSITIVE (state < 22) pairs of the true map")
    p.add_argument("--min-resolved", type=float, default=0.8, help="minimum share of native residues with a resolved CA")
    p.add_argument("--allow-multichain", action="store_true", help="plumbing only: do not require a single chain")
    p.add_argument("--n-val", type=int, default=2500, help="val rows REQUESTED from the strict RingDataset (only ~1 in 10 passes the length 70-130 / single-chain / map filters: 96 of 958 scanned rows in the first GPU run); the scan stops as soon as --n-chains chains are found")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=8, help="designs per sampler call (chains sorted by length)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--conditions", default="masked,oracle,shuffled,generated",
                   help="comma list of masked | oracle | shuffled | generated | oracle_frac<k> (k %% of the true positive pairs revealed, nested per chain) | hotspot<n> | hotspot<n>_neg | edited | sparse "
                        "(= " + ",".join(SPARSE_CONDS) + ")")
    p.add_argument("--gen-temps", default="0.5", help="label temperatures of the generated condition (TRIADR_RING_TEMP; 0 = argmax at every step), comma list: one extra sampler run each")
    p.add_argument("--nrep", type=int, default=1, help="designs per chain and condition (replicates share the maps; replicate r uses noise seed + 100003*r in EVERY condition: shared-noise rule kept); "
                   "the per-chain metrics are replicate means and the spread is reported")
    p.add_argument("--rep-conds", default="", help="comma list: with --nrep > 1 only these conditions get the extra replicates (default: all). Typical: hotspot3,hotspot20,masked,oracle")
    p.add_argument("--cfg-w", default=os.environ.get("CFG_W", "1"), help="stream classifier-free guidance weights, comma list (env CFG_W; default 1 = off): every non-masked condition is run once more per w != 1 as "
                   "<cond>@cfg<w> with the SAME noise; pred = pred_masked + w (pred_cond - pred_masked) at every sampler step = 2x denoiser passes per step for those runs (TRIADR_STREAM_CFG)")
    p.add_argument("--cfg-conds", default=os.environ.get("CFG_CONDS", ""), help="comma list: restrict --cfg-w to these base conditions (default: all non-masked)")
    p.add_argument("--tsr-k", type=float, default=1.5, help="latent TSR k (reference sampler)")
    p.add_argument("--gen-overrides", default="", help="extra Hydra overrides of the composed inference config, space separated (plumbing: 'generation.args.nsteps=20 generation.n_recycle_schedule=null generation.n_recycle=1')")
    p.add_argument("--configs-dir", default=os.path.join(os.environ.get("PROJECT_ROOT", "/fork"), "configs"))
    p.add_argument("--denovo-cif", default=MONOMER_CIF)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--loader-factory", default="", help="path.py:func(args) -> (train, val, info): tests / loader-free plumbing")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    if bool(a.ckpt) == bool(a.bundle):
        p.error("give exactly one of --ckpt / --bundle")
    a.conditions_list = [c for c in a.conditions.split(",") if c]
    bad = []
    for c in a.conditions_list:
        if c == "sparse":
            continue
        try:
            parse_condition(c)
        except ValueError:
            bad.append(c)
    if bad:
        p.error(f"unknown conditions {bad}; choose from masked, oracle, shuffled, generated, oracle_frac<k>, hotspot<n>, hotspot<n>_neg, edited, sparse")
    a.gen_temps_list = [float(t) for t in a.gen_temps.split(",") if t != ""]
    a.rep_conds_list = [c for c in a.rep_conds.split(",") if c]
    a.cfg_w_list = [float(w) for w in a.cfg_w.split(",") if w != ""] or [1.0]
    a.cfg_conds_list = [c for c in a.cfg_conds.split(",") if c]
    if any(not math.isfinite(w) or w < 0 for w in a.cfg_w_list):
        p.error("--cfg-w weights must be finite numbers >= 0")
    if a.nrep < 1:
        p.error("--nrep must be >= 1")
    a.min_evaluated = 0
    return a


def main(argv=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "recompute":  # python -m triadj_ext.s2.cond_gen recompute --npz X.npz --out Y.json [--n-boot 1000]
        ap = argparse.ArgumentParser(prog="cond_gen recompute", description=recompute_metrics.__doc__.split("\n")[0])
        ap.add_argument("--npz", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--n-boot", type=int, default=1000)
        ap.add_argument("--seed", type=int, default=1234)
        a = ap.parse_args(argv[1:])
        res = recompute_metrics(a.npz, a.n_boot, a.seed, a.out)
        for c in res["cells"]:
            print(cell_line(c), flush=True)
        for m in METRICS:
            log(f"means {m}: " + " ".join(f"{c}={res['means'][m][c][0]:.4g}" for c in res["conditions"] if c in res["means"][m]))
        log(f"rescored {res['n_chains']} chains of {a.npz} -> {a.out} (sat terms {'recomputed' if res['sat_available'] else 'NaN: the npz has no true_map'})")
        return res
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
