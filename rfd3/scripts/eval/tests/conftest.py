"""Shared fixtures for the WP-EVAL CPU tests: synthetic structures (no PDB mirror, no GPU, no rfd3 import)."""
import gzip
import io
import os
import sys

import numpy as np
import pytest

EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import eval_common as ec  # noqa: E402

# fixed local offsets: N and C relative to CA; every residue gets the same offsets, so the reconstructed CB is CA + V
N_OFF = np.array([-1.2, 0.9, 0.0])
C_OFF = np.array([1.2, 0.9, 0.3])
CB_V = ec.reconstruct_cb(N_OFF[None], np.zeros((1, 3)), C_OFF[None])[0]


def atoms_from_ca(ca, chain_id="A", res_start=1, res_names=None, with_cb=True, cb_real_offset=None):
    """biotite AtomArray with N, CA, C, O, (CB) per residue from CA coordinates (fixed local frame, see above)."""
    import biotite.structure as struc

    ca = np.asarray(ca, float)
    n = len(ca)
    names = ["N", "CA", "C", "O"] + (["CB"] if with_cb else [])
    arr = struc.AtomArray(n * len(names))
    cbv = CB_V if cb_real_offset is None else np.asarray(cb_real_offset, float)
    k = 0
    for i in range(n):
        for nm in names:
            off = {"N": N_OFF, "CA": np.zeros(3), "C": C_OFF, "O": C_OFF + np.array([0.6, 0.9, 0.0]), "CB": cbv}[nm]
            arr.coord[k] = ca[i] + off
            arr.atom_name[k] = nm
            arr.element[k] = nm[0]
            arr.res_name[k] = (res_names[i] if res_names else "ALA")
            arr.res_id[k] = res_start + i
            arr.chain_id[k] = chain_id
            arr.hetero[k] = False
            k += 1
    return arr


def two_strands_ca(d: float, n1: int = 10, loop: int = 2):
    """Two antiparallel straight 'strands' (CA spacing 3.8 A) d apart, joined by a short loop: block A = [0,n1), block B = after loop."""
    s1 = np.array([[3.8 * i, 0.0, 0.0] for i in range(n1)])
    s2 = np.array([[3.8 * (n1 - 1 - k), d, 0.0] for k in range(n1)])
    lp = np.array([s1[-1] + (s2[0] - s1[-1]) * (j + 1) / (loop + 1) for j in range(loop)])
    return np.concatenate([s1, lp, s2]), (0, n1), (n1 + loop, 2 * n1 + loop)


def helix_bundle_ca(n_helix=4, helix_len=14, loop_len=5, spacing=9.5):
    """CA trace of an up-down helix bundle (ideal alpha helices on a square lattice, interpolated loops)."""
    pts = []
    pos = [(0, 0), (spacing, 0), (spacing, spacing), (0, spacing)]
    for h in range(n_helix):
        cx, cy = pos[h % 4]
        up = h % 2 == 0
        hel = []
        for i in range(helix_len):
            th = np.radians(100 * i)
            hel.append([cx + 2.3 * np.cos(th), cy + 2.3 * np.sin(th), (1.5 * i) if up else (1.5 * (helix_len - 1 - i))])
        hel = np.array(hel)
        if pts:
            prev = pts[-1]
            for j in range(loop_len):
                pts.append(prev + (hel[0] - prev) * (j + 1) / (loop_len + 1))
        pts.extend(list(hel))
    return np.array(pts)


def to_cif_gz_bytes(atoms) -> bytes:
    import biotite.structure.io.pdbx as pdbx

    f = pdbx.CIFFile()
    pdbx.set_structure(f, atoms)
    buf = io.StringIO()
    f.write(buf)
    return gzip.compress(buf.getvalue().encode())


@pytest.fixture
def strands():
    return two_strands_ca


@pytest.fixture
def make_atoms():
    return atoms_from_ca
