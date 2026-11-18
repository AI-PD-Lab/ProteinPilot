"""Shared helpers of the HARMONIA-v2 guardrail harness (WP-EVAL, N003). CPU only.

Everything here is numpy/biotite; nothing imports the rfd3/foundry package (its __init__ needs
beartype/environs, which the CPU envs lack). The TRAINING label function is loaded straight from its
file (`load_label_module`), so the spec builder and the training pipeline share one implementation.

Conventions (identical to the training labels, foundry/data/ss_adj.py):
  * SS labels 0 helix / 1 strand / 2 loop / 3 masked; ADJ states 0 non-contact / 1 contact / 2 masked
    (hot/cold states 3/4 are never emitted by this harness except in hotspot_eval.py's optional arms).
  * CB = the RFD1 reconstruction from N/CA/C (`reconstruct_cb`), the SAME point the label function uses
    for every residue, glycine included. A block pair is "in contact" when min CB-CB < CONTACT_CUT (6 A).
  * Block = maximal run of one SS label on one chain; only helix/strand blocks carry ADJ in the v2
    dialect (loops are masked). Block ids are 0-based over the H/E blocks in sequence order.

Run python scripts here with `-I` if you like: every entry script puts its own directory on sys.path.
"""
from __future__ import annotations

import gzip
import importlib.util
import io
import json
import math
import os
import re
import sys
import tarfile
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DEFAULT = os.path.abspath(os.path.join(HERE, "..", ".."))
WORKSPACE = "/path/to/work_rfd3"
V2_ROOT = WORKSPACE + "/v2"
EVAL_ROOT = V2_ROOT + "/eval"
SPECS_TAR_DEFAULT = EVAL_ROOT + "/specs_v2.tar.gz"
PDB_MIRROR_TAR = WORKSPACE + "/pdb_mirror.tar"
PDB_MIRROR_INDEX = WORKSPACE + "/pdb_mirror_tar_index.parquet"
STOCK_CKPT = "/path/to/work/rfd3/rfd3_latest.ckpt"

SS_HELIX, SS_STRAND, SS_LOOP, SS_MASKED = 0, 1, 2, 3
SS_LETTER = {0: "H", 1: "E", 2: "L", 3: "-"}
ADJ_NO, ADJ_YES, ADJ_MASK, ADJ_HOT, ADJ_COLD = 0, 1, 2, 3, 4
CONTACT_CUT = 6.0          # block-pair contact, CB-CB (label rule)
LR_CUT, LR_SEP = 8.0, 12   # long-range contact density: CB-CB < 8 A and |i-j| > 12 (A4 convention)
BREAK_CUT = 4.2            # CA-CA chain break
CA_CLASH_CUT, CA_CLASH_SEP = 3.5, 3   # CA-CA < 3.5 A, |i-j| >= 3 (A5 4.3.3 / A4)
HEAVY_CLASH_CUT = 2.2      # heavy atoms of residues |i-j| >= 2 (a real overlap, not an H-bond)
SPEC_ROOT_TOKEN = "@SPEC_ROOT@"

CFG_DIR_RE = re.compile(r"cfg(?P<cfg>[0-9]+(?:\.[0-9]+)?)$")
DESIGN_RE = re.compile(r"^(?P<key>.+)_(?P<batch>\d+)_model_(?P<model>\d+)\.(?:cif\.gz|cif|pdb|pdb\.gz)$")


# --------------------------------------------------------------------------- label module
_LABEL_MODULES: dict[str, object] = {}


def load_module_from_path(path: str, name: str):
    """Import a single .py file without importing its package (no __init__ side effects)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_label_module(repo: str | None = None):
    """`foundry.data.ss_adj` from <repo>/foundry/src/foundry/data/ss_adj.py (the training label function)."""
    repo = os.path.abspath(repo or os.environ.get("HARMONIA_REPO") or REPO_DEFAULT)
    if repo not in _LABEL_MODULES:
        path = os.path.join(repo, "foundry", "src", "foundry", "data", "ss_adj.py")
        _LABEL_MODULES[repo] = load_module_from_path(path, "harmonia_ss_adj_labels")
    return _LABEL_MODULES[repo]


# --------------------------------------------------------------------------- geometry
def reconstruct_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """RFdiffusion1 C-beta reconstruction (same constants as foundry.data.ss_adj._reconstruct_cb)."""
    b = ca - n
    cv = c - ca
    a = np.cross(b, cv)
    return (-0.57910144 * a + 0.5689693 * b - 0.5441217 * cv + ca).astype(np.float64)


def pairwise(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    b = a if b is None else b
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def block_min_dist(cb: np.ndarray, ia, ib) -> float:
    """min CB-CB distance between two residue index sets (NaN-safe; nan if undefined)."""
    d = pairwise(cb[np.asarray(ia)], cb[np.asarray(ib)])
    d = d[np.isfinite(d)]
    return float(d.min()) if d.size else float("nan")


def block_n_below(cb: np.ndarray, ia, ib, cut: float = CONTACT_CUT) -> int:
    d = pairwise(cb[np.asarray(ia)], cb[np.asarray(ib)])
    return int((d[np.isfinite(d)] < cut).sum())


def blocks_from_ss(ss) -> list[tuple[int, int, int]]:
    """Maximal runs of equal SS label -> [(label, start, stop)] (stop exclusive), all labels."""
    ss = np.asarray(ss).astype(int)
    out, i = [], 0
    while i < len(ss):
        j = i + 1
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        out.append((int(ss[i]), i, j))
        i = j
    return out


def he_blocks_from_ss(ss) -> list[dict]:
    """Helix/strand blocks only, ids 0..nb-1 in sequence order: {id, type, start, stop}."""
    out = []
    for lab, a, b in blocks_from_ss(ss):
        if lab in (SS_HELIX, SS_STRAND):
            out.append({"id": len(out), "type": SS_LETTER[lab], "start": a, "stop": b})
    return out


def covalent_neighbours(blocks: list[dict]) -> set[tuple[int, int]]:
    """H/E block pairs (i, i+1) with NO residue between them (direct sequence neighbours)."""
    return {(i, i + 1) for i in range(len(blocks) - 1) if blocks[i]["stop"] == blocks[i + 1]["start"]}


def block_idx(blocks: list[dict], i: int) -> np.ndarray:
    return np.arange(blocks[i]["start"], blocks[i]["stop"])


# --------------------------------------------------------------------------- statistics
def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, stat=np.mean):
    """(stat, lo, hi) of the finite values of x; bootstrap percentile CI. NaN triple if empty."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    s = float(stat(x))
    if x.size == 1:
        return s, s, s
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1)
    return s, float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def diff_ci(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Bootstrap CI of mean(a) - mean(b) (independent samples)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    ba = a[rng.randint(0, a.size, size=(n_boot, a.size))].mean(1)
    bb = b[rng.randint(0, b.size, size=(n_boot, b.size))].mean(1)
    d = ba - bb
    return float(a.mean() - b.mean()), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


# --------------------------------------------------------------------------- structure IO
@dataclass
class Chain:
    """Protein residues of one chain; arrays are per residue (NaN where an atom is missing)."""
    chain_id: str
    res_ids: np.ndarray
    res_names: list
    n: np.ndarray
    ca: np.ndarray
    c: np.ndarray
    cb: np.ndarray           # RFD1-reconstructed from N/CA/C (the label convention)
    cb_leg: np.ndarray       # LEGACY representative atom: real CB, else CA (score_ferredoxin.py / N002 convention)
    heavy: np.ndarray        # [M,3] all heavy atoms of the chain
    heavy_res: np.ndarray    # [M] residue index (0-based within the chain) of each heavy atom
    heavy_name: np.ndarray   # [M] atom names

    def __len__(self):
        return len(self.ca)


def read_structure(data: bytes | str, name: str):
    """biotite AtomArray (model 1) from bytes or a path; supports .cif(.gz) and .pdb(.gz)."""
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io.pdb as pdb

    if isinstance(data, str):
        with open(data, "rb") as fh:
            data = fh.read()
    low = name.lower()
    if low.endswith(".gz"):
        data = gzip.decompress(data)
        low = low[:-3]
    text = data.decode("utf-8", errors="replace")
    if low.endswith(".pdb"):
        return pdb.PDBFile.read(io.StringIO(text)).get_structure(model=1)
    return pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(text)), model=1)


def protein_chains(atoms, min_len: int = 1) -> list[Chain]:
    """Split an AtomArray into protein Chains (amino acids only, no H), in order of appearance."""
    import biotite.structure as struc

    atoms = atoms[struc.filter_amino_acids(atoms) & (atoms.element != "H")]
    chains = []
    seen = []
    for cid in atoms.chain_id:
        if cid not in seen:
            seen.append(cid)
    for cid in seen:
        ch = atoms[atoms.chain_id == cid]
        starts = list(struc.get_residue_starts(ch)) + [ch.array_length()]
        L = len(starts) - 1
        if L < min_len:
            continue
        N = np.full((L, 3), np.nan)
        CA = np.full((L, 3), np.nan)
        C = np.full((L, 3), np.nan)
        CBL = np.full((L, 3), np.nan)
        names, rids, hres = [], [], []
        for i in range(L):
            s, e = starts[i], starts[i + 1]
            nm = ch.atom_name[s:e]
            for arr, atom in ((N, "N"), (CA, "CA"), (C, "C")):
                k = np.nonzero(nm == atom)[0]
                if k.size:
                    arr[i] = ch.coord[s + k[0]]
            k = np.nonzero(nm == "CB")[0]
            CBL[i] = ch.coord[s + k[0]] if k.size else CA[i]
            names.append(str(ch.res_name[s]))
            rids.append(int(ch.res_id[s]))
            hres.extend([i] * (e - s))
        chains.append(Chain(
            chain_id=str(cid), res_ids=np.asarray(rids), res_names=names, n=N, ca=CA, c=C,
            cb=reconstruct_cb(N, CA, C), cb_leg=CBL, heavy=ch.coord.astype(np.float64),
            heavy_res=np.asarray(hres), heavy_name=np.asarray(ch.atom_name)))
    return chains


def sse_labels(ca: np.ndarray) -> np.ndarray:
    """Per-residue SS from CA only with biotite P-SEA (annotate_sse = the training label's SS), as 0/1/2."""
    import biotite.structure as struc

    n = len(ca)
    arr = struc.AtomArray(n)
    arr.coord = np.nan_to_num(ca).astype(np.float32)
    arr.atom_name[:] = "CA"
    arr.res_name[:] = "ALA"
    arr.res_id = np.arange(1, n + 1)
    arr.chain_id[:] = "A"
    arr.element[:] = "C"
    arr.hetero[:] = False
    sse = struc.annotate_sse(arr)
    out = np.full(n, SS_LOOP, dtype=int)
    out[sse == "a"] = SS_HELIX
    out[sse == "b"] = SS_STRAND
    return out


# --------------------------------------------------------------------------- tar helpers
def safe_extract(tar_path: str, dest: str):
    """Extract a tar(.gz) into dest, refusing absolute/parent-escaping members."""
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            p = os.path.normpath(m.name)
            if p.startswith("..") or os.path.isabs(p) or m.issym() or m.islnk():
                raise ValueError(f"refusing unsafe tar member {m.name!r}")
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # python < 3.12 without the filter argument
            tf.extractall(dest)


def read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def write_json(path: str, obj, indent: int | None = 1):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def fmt(x, nd: int = 2) -> str:
    """Compact number formatting for tables ('-' for missing)."""
    if x is None:
        return "-"
    try:
        if not math.isfinite(float(x)):
            return "-"
    except (TypeError, ValueError):
        return str(x)
    return f"{float(x):.{nd}f}"
