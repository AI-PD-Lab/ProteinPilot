"""TM-score and RMSD for two CA traces with a FIXED residue correspondence (pure numpy, CPU), plus an optional
wrapper around foldseek's exhaustive TM-align for the sequence-order-independent score.

Why fixed correspondence is enough for the guardrail: a fold-conditioned design has exactly the length and the
block order of its reference, so residue i of the design corresponds to residue i of the reference. Index-aligned
TM-score <= TM-align TM-score for the same pair, so `tm_index >= 0.5` implies the TM-align criterion
(TM >= 0.5 = same fold) while the converse is not guaranteed; use `foldseek_tm` for the optimal-alignment value.

TM-score (Zhang & Skolnick 2004), normalised by the length of the reference:
    TM = max_superposition (1/L) sum_i 1 / (1 + (d_i/d0)^2),   d0 = 1.24 (L-15)^(1/3) - 1.8  (>= 0.5)
The maximum is searched like the TMscore program: Kabsch superpositions seeded from fragments of length L, L/2,
L/4, ... (>= 4), each refined by iteratively re-superimposing the residues within a distance cutoff.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

import numpy as np

FOLDSEEK_DEFAULT = "/path/to/work/tools/foldseek/foldseek/bin/foldseek"


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rotation R and translation t minimising |(P @ R + t) - Q| (rows = points). Returns (R, t)."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return R, qc - pc @ R


def rmsd_superposed(P: np.ndarray, Q: np.ndarray) -> float:
    """CA RMSD after the optimal rigid superposition (same residue order)."""
    R, t = kabsch(P, Q)
    return float(np.sqrt((((P @ R + t) - Q) ** 2).sum(1).mean()))


def d0_of(L: int) -> float:
    return max(0.5, 1.24 * (max(L, 16) - 15) ** (1.0 / 3.0) - 1.8)


def tm_score(P: np.ndarray, Q: np.ndarray, L_ref: int | None = None) -> float:
    """TM-score of mobile P against reference Q (same length, residue i <-> i), normalised by len(Q) (or L_ref)."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    n = len(Q)
    if len(P) != n:
        raise ValueError(f"fixed-correspondence TM-score needs equal lengths ({len(P)} vs {n})")
    L = L_ref or n
    d0 = d0_of(L)
    d0_search = min(max(d0, 4.5), 8.0)
    best = 0.0

    def score(R, t):
        d2 = (((P @ R + t) - Q) ** 2).sum(1)
        return float((1.0 / (1.0 + d2 / d0 ** 2)).sum() / L), d2

    lens = sorted({n} | {max(4, n // 2 ** k) for k in range(1, 6) if n // 2 ** k >= 4}, reverse=True)
    for lf in lens:
        step = max(1, lf // 2) if lf < n else n
        for s in range(0, n - lf + 1, step):
            idx = np.arange(s, s + lf)
            if len(idx) < 3:
                continue
            R, t = kabsch(P[idx], Q[idx])
            sc, d2 = score(R, t)
            best = max(best, sc)
            cut = d0_search
            for _ in range(20):
                sel = np.nonzero(d2 < cut ** 2)[0]
                while len(sel) < 3 and cut < 1e3:
                    cut += 0.5
                    sel = np.nonzero(d2 < cut ** 2)[0]
                if len(sel) < 3:
                    break
                R2, t2 = kabsch(P[sel], Q[sel])
                sc2, d2 = score(R2, t2)
                best = max(best, sc2)
                if np.array_equal(sel, np.nonzero(d2 < cut ** 2)[0]):
                    break
    return best


def foldseek_tm(pdb_a: str, pdb_b: str, foldseek: str = FOLDSEEK_DEFAULT) -> float | None:
    """alntmscore (TM-align, sequence-order independent, exhaustive) of two PDB files via foldseek; None if it fails."""
    if not os.path.exists(foldseek):
        return None
    with tempfile.TemporaryDirectory(prefix="fs_tm_") as td:
        out = os.path.join(td, "o.m8")
        r = subprocess.run([foldseek, "easy-search", pdb_a, pdb_b, out, os.path.join(td, "tmp"), "--alignment-type", "1",
                            "--exhaustive-search", "--tmscore-threshold", "0.0", "-e", "100000", "--format-output",
                            "query,target,alntmscore,qcov,tcov", "--threads", "1", "-v", "1"],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            return None
        vals = [float(l.split("\t")[2]) for l in open(out) if l.strip() and re.match(r"^\S+\t\S+\t[0-9.]+", l)]
        return max(vals) if vals else None
