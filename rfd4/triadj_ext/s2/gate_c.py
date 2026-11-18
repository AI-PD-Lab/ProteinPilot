#!/usr/bin/env python3
"""TRIAD-R S2 GATE (c): do the interaction labels the model GENERATES agree with the interactions actually PRESENT in its decoded all-atom designs?

Needs numpy; the natural reference additionally needs pyarrow + lmdb. All three are in the login-node `triad` env:
  PYTHONDONTWRITEBYTECODE=1 /path/to/miniforge3/envs/triad/bin/python -I triadj_ext/s2/gate_c.py \
      --run A2000_masked_s0=<tj_s2_designs_..._masked_..tar.gz> --run A2000_generated_s0=<designs tar>,<tj_s2_ring_..npz> --run base_s0=<stock tar> --json OUT.json
  then (stdlib, any python3)   gates_report.py ... --gate-c OUT.json

What it does (CPU only, RING is a CPU tool; ~0.9 s per design, parallel over designs, per-design files node-local in /tmp/$USER, ONE output json)
 1. RING (the static binary third_party/ring-4.0/out/bin/ring, `RING v4.0-31-g8c5f7cd` = the `ring_version` of every record of the training lmdb) is run on every design PDB with
    `--all_edges --prefer_label_asym_id --no_colors`, the flags of the lmdb pipeline. Its `<pdb>_ringEdges` file is a TAB table (NodeId1 Interaction NodeId2 ...);
    `TRIAD/triad/inference/ring_compute.py` expects `*_ringEdges.json` and would silently fall through to its biotite fallback with this binary: it is not used.
 2. Edge -> residue pair -> state is the lmdb/ring_data conversion, with every constant (the 11 types, COLLAPSE_PRIORITY = strongest type per pair, RING_RECORD_TO_TYPE,
    NO_INTERACTION = 22, intra states 0..10) read from `triadj_ext/ring_data.py` by AST (no import: ring_data imports rfproteina at import time). The only part that is NOT in
    ring_data is raw RING edge -> lmdb record key, which the lmdb pipeline did before ring_data sees the records. It was recovered from DATA (the lmdb records of natural chains
    vs RING run on their CIFs; `--validate-converter N` reproduces the check), NOT from the pipeline code:
      HBOND:*->hbonds  IONIC->salt_bridges  PIPISTACK->pi_stacking  PICATION->pi_cation  SSBOND->disulfides  METAL_ION->metal_ion  HALOGEN->halogen  PIHBOND->pi_hbond
      VDW:SC_SC between two residues of {ALA VAL ILE LEU MET PHE TRP PRO CYS} -> hydrophobic (247/247 edges of 8 chains, 0 mismatches), every other VDW:* -> vdw
    (`ionic_supplemented` / `pi_cation_supplemented` entries are absent from natural monomers: 0 of ~25k entries; they cannot arise from RING output and are not produced.)
    Residues are indexed by their order in the design PDB (= the design's token order; L=100 single chain); labels are symmetric [L, L]: 0..10 intra types, 22 none, 24 diagonal.
 3. Per run: RING-measured positives per residue (PRIMARY: nearly length independent) and density = positive pairs / C(L,2) (length dependent), per design and per class, bootstrap 95 % CI over
    designs, next to the NATURAL reference, which is LENGTH MATCHED: a design of length L is compared with the val-manifest monomer chains (protein only, no non-standard residues, labels
    straight from the lmdb) of length L(1 -/+ 0.2) (L=100: 80..120); the natural mean is the average of the window means weighted by the share of designs of each length; CI by bootstrap over
    designs and over SEQUENCE clusters. The natural length distribution (loaded / matched / designs) is in the json (`natural.length_distribution`) and on a `GATEC natural_lengths` line.
    (The natural chains of 80..120 are NOT centred on 100: 54 of 141 are 80-89, 44 are 110-119; only the window means / positives per residue are matched, not the full length histogram.)
 4. Runs given with an npz (generated labels, `generate_entry.load_ring_states`): IDENTITY join design_NNNN -> example id (design_map.tsv) -> npz id, BOUND TO THE SAME RUN
    (example ids like monomer-short_rep0 repeat across seeds): file-name stems `tj_s2_designs_<stem>.tar.gz` / `tj_s2_ring_<stem>.npz` must be equal (tag, mode, seed, length, job id), the npz
    metadata seed / sampler must agree and the npz must hold the archive's number of maps. NO index fallback. A design without a map, a missing / empty design_map.tsv, repeated ids (npz,
    design_map rows, pdb members), a foreign / unverifiable npz make the run INVALID(join) (no agreement computed; `--allow-unverified-names` is for exploration only). Agreement of the generated final map with the
    RING map of the SAME design. Strata: `gap3` = pairs with |i-j| >= 3 (PRIMARY: RING reports no pair closer than 3 in sequence, so |i-j| < 3 can never be a true positive; those pairs are
    counted as generated false positives in `all`), `lr` = |i-j| >= 8 (long range; helix i,i+3/4 patterns cannot carry it), `all` = every pair.
      binary  P / R / F1 of "any interaction" (generated states 0..21 incl. the inter states 11..21 = class state%11, vs RING states 0..10), pooled over designs (micro), CI by bootstrap of designs
      class   per-class P / R / F1 (class-matched on the collapsed label), support, AP of the distance score per class (pooled, no CI)
      23-state level  `f1_state23` = micro F1 of the EXACT state (generated state == RING state, 0..21; an inter state never matches), the pooled TYPE confusion (`type_confusion`: rows generated,
        cols RING) over pairs that are positive in BOTH maps, `type_agree` = identical type among those position-matched positives, `type_majority_baseline` = share of the commonest RING type among
        them (always predict that type), `dtype_vs_majority` = type_agree - baseline with a CI, `type_kappa`. Binary agreement alone is NOT a 23-state result.
      distance-only baseline (CA-CA on the decoded design): AUC and AP of -d (per design, macro mean), P/R/F1 of d <= 8 A, and `distK` = the K closest pairs with K = the design's own number of
        generated positives (same budget as the label map: the like-for-like comparison). The generated map is a hard 0/1 predictor: its AUC is balanced accuracy (`auc_hard`).
      shuffled control: the generated map of design i against the RING map of design j != i of the same length (cyclic donors, mean over up to 8 shifts); dF1_shuf = F1(true) - F1(shuffled)
        with a paired bootstrap. A helix-only map scores well against ANY helical design; only the difference shows agreement is specific.
      in-contact lift: among pairs with CA-CA <= 8 A, P(RING positive | generated positive) - P(RING positive | generated none): does the label add to a distance threshold?
    Provisional verdict (no absolute floor was pre-registered: read the F1 itself): PASS = dF1_shuf CI > 0 AND F1(gen) - max(F1 distK, F1 dist8) CI > 0 AND dtype_vs_majority CI > 0, all in BOTH
    `gap3` and `lr`; WEAK = binary specificity (dF1_shuf CI > 0) in `gap3` only; else FAIL. (gap3 alone is not enough: helix i,i+3/4 contacts are generic and the plain 8 A rule already scores
    F1 ~0.58 there; a map whose types are no better than "always hbond" is degenerate.) Precedence over those: INVALID(join) > INCOMPLETE (also unless >= --min-informative (30) designs carry positive evidence, i.e. RING-measured AND generated positives, in BOTH strata, and
    unless every verdict statistic keeps >= 95 % of its bootstrap draws (draws without typed evidence are counted, not hidden); `n_informative` is printed; evaluated designs < --min-frac (0.9) of the ARCHIVED
    designs or < --min-designs (30): RING failures, length mismatches and --max-designs count against it; the line shows the counts) > INVALID_STATES (impossible states in the final monomer map:
    inter-chain 11..21, MASK, PAD off the diagonal, above --max-invalid-frac (0) of the generated positives; counts printed). `verdict_if_valid` keeps the rule's own answer.
 5. ONE json (schema gate_c/v1) + one-line `GATEC ...` lines on stdout. `--validate-converter N` (optional) re-checks the edge->label conversion on N natural chains against the lmdb.
Never writes under /home except the one --json path you give (refuses a temp root under /home).
"""
from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures as cf
import datetime
import getpass
import gzip
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RING_DATA_PY = REPO / "triadj_ext" / "ring_data.py"
RING_BIN_DEFAULT = "/path/to/home/Training_Triad/TRIAD/third_party/ring-4.0/out/bin/ring"
RING_FLAGS = ("--all_edges", "--prefer_label_asym_id", "--no_colors")
RING_VERSION = "RING v4.0-31-g8c5f7cd"
MANIFEST_VAL = "/path/to/work/manifests/pdb_afdb_val_clustersplit.parquet"
LMDB_PATH = "/path/to/work/ring_pipeline/ring_interactions.lmdb"
TAR_INDEX = "/path/to/work_rfd3/pdb_mirror_tar_index.parquet"
TAR_PATH = "/path/to/work_rfd3/pdb_mirror.tar"

MIN_SEP = 3            # RING reports no residue pair closer than 3 in sequence (natural lmdb: 0.5 % at sep 1-2, numbering irregularities only)
LONG_SEP = 8
DIST_CUT = 8.0         # A, CA-CA contact predictor
N_CLASS = 11
MIN_CLASS_SUPPORT = 20  # pooled RING positives for a per-class number to count as "meaningful"
HYDROPHOBIC_RES = frozenset(("ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "CYS"))   # recovered from the lmdb, see module doc
STD_AA = frozenset(("ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"))
#: raw RING interaction name (before ':') -> lmdb record key (ring_data.RING_RECORD_TO_TYPE then gives the type)
RING_EDGE_TO_RECORD = {"HBOND": "hbonds", "IONIC": "salt_bridges", "PIPISTACK": "pi_stacking", "PICATION": "pi_cation", "SSBOND": "disulfides", "VDW": "vdw",
                       "METAL_ION": "metal_ion", "HALOGEN": "halogen", "PIHBOND": "pi_hbond"}


# ------------------------------------------------------------------------------------------------------------- constants from ring_data (AST, no import)
def load_ring_constants(path: Path = RING_DATA_PY) -> dict:
    """The label constants of ``triadj_ext/ring_data.py`` (single source of truth) evaluated from its source: ring_data imports rfproteina / torch at import time."""
    want = {"NO_INTERACTION", "RING_MASK", "RING_PAD", "RING_INTERACTION_TYPES", "COLLAPSE_PRIORITY", "RING_RECORD_TO_TYPE", "RESNAME_ALIASES"}
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in want:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    missing = want - set(out)
    if missing:
        raise RuntimeError(f"ring_data constants not found in {path}: {sorted(missing)}")
    out["TYPES"] = tuple(out["RING_INTERACTION_TYPES"])
    out["N_TYPES"] = len(out["TYPES"])
    out["TYPE_RANK"] = np.array([out["COLLAPSE_PRIORITY"].index(t) for t in out["TYPES"]], dtype=np.int64)
    return out


C = load_ring_constants()
NONE, DIAG = C["NO_INTERACTION"], C["RING_PAD"]
TYPES = C["TYPES"]
assert N_CLASS == C["N_TYPES"] and NONE == 22


# ------------------------------------------------------------------------------------------------------------- RING output -> pair labels
def edge_type_index(interaction: str, res1: str, res2: str):
    """Raw RING edge (``HBOND:MC_MC``, ``VDW:SC_SC`` ...) -> index into RING_INTERACTION_TYPES, or None (unknown name). See the module docstring for the VDW split."""
    name, _, sub = interaction.partition(":")
    key = RING_EDGE_TO_RECORD.get(name)
    if key is None:
        return None
    if key == "vdw" and sub == "SC_SC" and res1 in HYDROPHOBIC_RES and res2 in HYDROPHOBIC_RES:
        key = "hydrophobic"
    return TYPES.index(C["RING_RECORD_TO_TYPE"][key])


def collapse_pairs(i: np.ndarray, j: np.ndarray, t: np.ndarray):
    """(i, j, type index) entries -> unique pairs (i < j) with the STRONGEST type (ring_data.COLLAPSE_PRIORITY via TYPE_RANK), as in ``build_ring_labels``."""
    i, j, t = (np.asarray(x, dtype=np.int64) for x in (i, j, t))
    if i.size == 0:
        return i, j, t
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    keep = lo != hi
    lo, hi, t = lo[keep], hi[keep], t[keep]
    if lo.size == 0:
        return lo, hi, t
    key = lo * np.int64(1 << 31) + hi
    order = np.lexsort((C["TYPE_RANK"][t], key))
    ks = key[order]
    first = np.ones(ks.size, dtype=bool)
    first[1:] = ks[1:] != ks[:-1]
    sel = order[first]
    return lo[sel], hi[sel], t[sel]


def parse_pdb(text: str) -> dict:
    """Protein residues of a PDB text in file order: keys (chain, resSeq, insertion code or '_'), names, CA coordinates [L, 3] (NaN when absent)."""
    index, names, ca = {}, [], []
    for ln in text.splitlines():
        if not ln.startswith("ATOM"):
            continue
        key = (ln[21], int(ln[22:26]), ln[26].strip() or "_")
        r = index.get(key)
        if r is None:
            r = index[key] = len(names)
            names.append(ln[17:20].strip())
            ca.append([np.nan] * 3)
        if ln[12:16].strip() == "CA":
            ca[r] = [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
    return {"index": index, "names": names, "ca": np.asarray(ca, dtype=np.float64).reshape(-1, 3)}


def parse_ring_edges(text: str) -> list:
    """RING 4 `_ringEdges` TSV -> [(node1, interaction, node2)] with node = (chain, resnum, icode, resname); the header line is skipped."""
    out = []
    for ln in text.splitlines():
        f = ln.split("\t")
        if len(f) < 3 or f[0] == "NodeId1":
            continue
        n = []
        for s in (f[0], f[2]):
            p = s.split(":")
            n.append((p[0], int(p[1]), p[2], p[3]))
        out.append((n[0], f[1], n[1]))
    return out


def edges_to_state(edges: list, pdb: dict):
    """-> (state int8 [L, L]: 0..10 type, 22 none, 24 diagonal; stats). Edges whose residues are not in the PDB or not standard amino acids are counted, never guessed."""
    L = len(pdb["names"])
    ii, jj, tt = [], [], []
    st = collections.Counter()
    for n1, it, n2 in edges:
        a, b = pdb["index"].get(n1[:3]), pdb["index"].get(n2[:3])
        if a is None or b is None:
            st["edge_residue_not_in_pdb"] += 1
            continue
        r1, r2 = pdb["names"][a], pdb["names"][b]
        if r1 not in STD_AA or r2 not in STD_AA:
            st["edge_nonstandard_residue"] += 1
            continue
        k = edge_type_index(it, r1, r2)
        if k is None:
            st["edge_unmapped_" + it.split(":")[0]] += 1
            continue
        st["edge_" + it.split(":")[0]] += 1
        ii.append(a)
        jj.append(b)
        tt.append(k)
    pi, pj, pt = collapse_pairs(np.array(ii), np.array(jj), np.array(tt))
    s = np.full((L, L), NONE, dtype=np.int8)
    s[pi, pj] = pt
    s[pj, pi] = pt
    s[np.arange(L), np.arange(L)] = DIAG
    st["pairs"] = int(pi.size)
    return s, dict(st)


def run_ring(pdb_path: str, out_dir: str, ring_bin: str = RING_BIN_DEFAULT, timeout: float = 300.0) -> str:
    """Run RING on one structure file; returns the text of `<file>_ringEdges`. Raises RuntimeError on a non-zero exit or a missing edges file."""
    os.makedirs(out_dir, exist_ok=True)
    cp = subprocess.run([ring_bin, "-i", pdb_path, "--out_dir", out_dir, *RING_FLAGS], capture_output=True, text=True, timeout=timeout)
    if cp.returncode != 0:
        raise RuntimeError(f"ring exit {cp.returncode}: {(cp.stderr or cp.stdout)[-300:]}")
    hits = [f for f in os.listdir(out_dir) if f.endswith("_ringEdges")]
    if not hits:
        raise RuntimeError("ring wrote no *_ringEdges file")
    return Path(out_dir, hits[0]).read_text()


def ring_version(ring_bin: str = RING_BIN_DEFAULT) -> str:
    """`ring --version` (last line); the lmdb records carry ``ring_version`` = RING_VERSION."""
    try:
        cp = subprocess.run([ring_bin, "--version"], capture_output=True, text=True, timeout=30)
        return (cp.stdout.strip().splitlines() or [""])[-1]
    except Exception as e:   # noqa: BLE001
        return f"unknown ({type(e).__name__})"


def _ring_job(job: tuple) -> dict:
    """Worker (module level: picklable). job = (name, pdb_path, work_dir, ring_bin, timeout)."""
    name, pdb_path, work, ring_bin, timeout = job
    try:
        pdb = parse_pdb(Path(pdb_path).read_text())
        if not pdb["names"]:
            raise RuntimeError("no ATOM records")
        t0 = time.time()
        text = run_ring(pdb_path, os.path.join(work, name), ring_bin, timeout)
        state, st = edges_to_state(parse_ring_edges(text), pdb)
        shutil.rmtree(os.path.join(work, name), ignore_errors=True)
        return {"name": name, "ok": True, "state": state, "ca": pdb["ca"], "seq": pdb["names"], "stats": st, "sec": time.time() - t0}
    except Exception as e:  # noqa: BLE001 - reported per design, the run goes on
        return {"name": name, "ok": False, "err": f"{type(e).__name__}: {e}"[:300]}


# ------------------------------------------------------------------------------------------------------------- inputs: tar, npz
def read_design_tar_ex(path: str, max_designs: int = 0) -> dict:
    """{'designs': [(name, pdb text)] in design order (first max_designs), 'dmap': {design: example id}, 'total': archived designs, 'dups': [...]}. Members `pdbs/design_NNNN.pdb`,
    `pdbs/design_map.tsv` (eval_ckpt.sbatch). Repeated design members / design_map rows are NEVER overwritten silently: they are listed in `dups` (the join then refuses the run)."""
    pdbs, dmap, dups = {}, {}, []
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:
            b = os.path.basename(m.name)
            if not m.isfile():
                continue
            if re.fullmatch(r"design_\d+\.pdb", b):
                if b[:-4] in pdbs:
                    dups.append(f"pdb member {b}")
                pdbs[b[:-4]] = tf.extractfile(m).read().decode()
            elif b == "design_map.tsv":
                for ln in tf.extractfile(m).read().decode().splitlines():
                    f = ln.split("\t")
                    if len(f) >= 2:
                        if f[0] in dmap:
                            dups.append(f"design_map row {f[0]}")
                        dmap[f[0]] = f[1]
    names = sorted(pdbs)
    total = len(names)
    if max_designs and max_designs < len(names):
        names = names[:max_designs]
    return {"designs": [(n, pdbs[n]) for n in names], "dmap": dmap, "total": total, "dups": dups}


def read_design_tar(path: str, max_designs: int = 0, return_total: bool = False):
    """-> (designs [(name, pdb text)], design_map {design: example id}[, total archived designs]); see ``read_design_tar_ex`` (duplicates)."""
    r = read_design_tar_ex(path, max_designs)
    return (r["designs"], r["dmap"], r["total"]) if return_total else (r["designs"], r["dmap"])


def load_ring_states(path: str) -> dict:
    """Same layout as ``generate_entry.load_ring_states`` (not imported: that module pulls in the model stack) -> {'meta', 'ids', 'ring': {id: int8 [n, n]}}."""
    z = np.load(path, allow_pickle=False)
    ids = [str(i) for i in z["ids"]]
    return {"meta": json.loads(str(z["meta"])), "ids": ids, "ring": {i: z[f"ring_{k}"] for k, i in enumerate(ids)}}


def run_name_from_tar(path: str) -> str:
    b = os.path.basename(path)
    b = re.sub(r"\.tar\.gz$", "", b)
    b = re.sub(r"^tj_s2_designs_", "", b)
    return re.sub(r"_\d+$", "", b)


# ------------------------------------------------------------------------------------------------------------- statistics
def avg_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, inv, cnt = np.unique(xs, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    avg = (cum - cnt + 1 + cum) / 2.0
    r = np.empty(x.size, dtype=np.float64)
    r[order] = avg[inv]
    return r


def rank_auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction; NaN when one class is empty."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = avg_ranks(np.asarray(score, dtype=np.float64))
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    """AP = sum_k (R_k - R_{k-1}) P_k over the distinct thresholds (the scikit-learn definition); NaN without positives."""
    y = np.asarray(y, dtype=bool)
    npos = int(y.sum())
    if npos == 0:
        return float("nan")
    s = np.asarray(score, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    s, yy = s[order], y[order]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), s.size - 1]       # last index of every tie group
    tp = np.cumsum(yy)[last]
    fp = (last + 1) - tp
    prec, rec = tp / (tp + fp), tp / npos
    return float(np.sum((rec - np.r_[0.0, rec[:-1]]) * prec))


def prf(tp, fp, fn):
    tp, fp, fn = float(tp), float(fp), float(fn)
    p = tp / (tp + fp) if tp + fp > 0 else float("nan")
    r = tp / (tp + fn) if tp + fn > 0 else float("nan")
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else float("nan")
    return p, r, f


def conf_binary(c: np.ndarray):
    """12x12 joint confusion (rows generated class, cols RING class; index 0 = none, 1..11 = type 0..10) -> binary tp, fp, fn, tn."""
    return c[1:, 1:].sum(), c[1:, 0].sum(), c[0, 1:].sum(), c[0, 0]


def conf_f1(c: np.ndarray) -> float:
    tp, fp, fn, _ = conf_binary(c)
    return prf(tp, fp, fn)[2]


def ci95(v) -> list:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [float("nan"), float("nan")]


def bootstrap(fn, n: int, B: int, seed: int) -> np.ndarray:
    """Design bootstrap: ``fn(idx)`` (idx = resampled design indices) -> 1-D float vector; returns [B, k]."""
    rng = np.random.default_rng(seed)
    return np.array([np.atleast_1d(np.asarray(fn(rng.integers(0, n, n)), dtype=np.float64)) for _ in range(B)])


def mean_ci(x, B: int, seed: int, groups=None) -> dict:
    """Mean of x with a 95 % bootstrap CI over designs (``groups`` = cluster id per element: clusters are the resampling unit and the ratio of sums is taken)."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "n": 0}
    if groups is None:
        xo = x[ok]
        bs = bootstrap(lambda idx: xo[idx].mean(), xo.size, B, seed)[:, 0]
        return {"mean": float(xo.mean()), "ci95": ci95(bs), "n": int(xo.size)}
    g = np.asarray(groups)[ok]
    uniq, inv = np.unique(g, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=x[ok]), np.bincount(inv).astype(np.float64)
    bs = bootstrap(lambda idx: sums[idx].sum() / cnts[idx].sum(), uniq.size, B, seed)[:, 0]
    return {"mean": float(x[ok].mean()), "ci95": ci95(bs), "n": int(ok.sum()), "n_clusters": int(uniq.size)}


# ------------------------------------------------------------------------------------------------------------- natural reference
def pairs_from_record(rec: dict, unit: str) -> dict:
    """An lmdb pn_unit record -> {'ij': (i, j, type) collapsed arrays over residue numbers, 'stats'}: intra entries of ``unit`` between standard residues,
    record keys / types from ring_data.RING_RECORD_TO_TYPE (so `ionic` is skipped: it duplicates `salt_bridges`), strongest type per pair."""
    ii, jj, tt = [], [], []
    st = collections.Counter()
    for key, tname in C["RING_RECORD_TO_TYPE"].items():
        for e in (rec.get(key) or {}).get("full") or []:
            r1 = C["RESNAME_ALIASES"].get(str(e["resname_1"]).strip().upper(), str(e["resname_1"]).strip().upper())
            r2 = C["RESNAME_ALIASES"].get(str(e["resname_2"]).strip().upper(), str(e["resname_2"]).strip().upper())
            if str(e["chain_1"]) != unit or str(e["chain_2"]) != unit:
                st["entry_other_chain"] += 1
                continue
            if r1 not in STD_AA or r2 not in STD_AA:
                st["entry_nonstandard_residue"] += 1
                continue
            ii.append(int(e["resnum_1"]))
            jj.append(int(e["resnum_2"]))
            tt.append(TYPES.index(tname))
    pi, pj, pt = collapse_pairs(np.array(ii), np.array(jj), np.array(tt))
    st["pairs"] = int(pi.size)
    return {"i": pi, "j": pj, "t": pt, "stats": dict(st)}


def class_counts(t: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(t, dtype=np.int64), minlength=N_CLASS)[:N_CLASS]


def natural_reference(manifest: str = MANIFEST_VAL, lmdb_path: str = LMDB_PATH, lo: int = 80, hi: int = 120, max_chains: int = 0) -> dict:
    """RING-label density of natural val-split monomer chains (labels straight from the lmdb; no RING run). Rows: is_monomer, protein only, ``lo <= n_residues <= hi``,
    a record in the lmdb, no non-standard residues. Returns per-chain arrays (for the bootstrap) under 'rows'."""
    import lmdb
    import pyarrow.parquet as pq

    cols = ["pdb_id", "pn_unit_iid", "n_residues", "ring_key", "is_monomer", "n_prot", "n_nuc", "n_ligand", "cluster"]
    t = pq.read_table(manifest, columns=cols).to_pydict()
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, subdir=True, max_readers=64)
    rows, seen, skip = [], set(), collections.Counter()
    with env.begin() as txn:
        for k in range(len(t["pdb_id"])):
            if not t["is_monomer"][k] or t["pn_unit_iid"][k] is None or t["ring_key"][k] is None:
                continue
            n = t["n_residues"][k]
            if n is None or not lo <= n <= hi or t["n_prot"][k] != 1 or (t["n_nuc"][k] or 0) != 0 or (t["n_ligand"][k] or 0) != 0:
                continue
            key = t["ring_key"][k]
            if key in seen:
                continue
            seen.add(key)
            raw = txn.get(key.encode())
            if raw is None:
                skip["no_record"] += 1
                continue
            rec = pickle.loads(raw)
            if (rec.get("metadata") or {}).get("has_nonstandard_residues"):
                skip["nonstandard_residues"] += 1
                continue
            p = pairs_from_record(rec, str(t["pn_unit_iid"][k]))
            cl = t["cluster"][k]
            rows.append({"id": key, "L": int(n), "cluster": str(cl).split("+", 1)[0] if cl else key, "n_pos": int(p["i"].size), "cls": class_counts(p["t"]), "stats": p["stats"]})
            if max_chains and len(rows) >= max_chains:
                break
    return {"rows": rows, "skipped": dict(skip), "manifest": manifest, "lmdb": lmdb_path, "length_range": [lo, hi]}


def density_summary(n_pos, cls, L, B: int, seed: int, groups=None) -> dict:
    """Density = positive pairs / C(L,2) per design; per class likewise; positives per residue. ``groups`` = cluster ids (natural) else designs are the unit."""
    n_pos, L = np.asarray(n_pos, dtype=np.float64), np.asarray(L, dtype=np.float64)
    cls = np.asarray(cls, dtype=np.float64).reshape(len(n_pos), N_CLASS)
    npairs = L * (L - 1) / 2
    out = {"n": int(len(n_pos)), "L_mean": float(L.mean()) if len(L) else float("nan"), "density": mean_ci(n_pos / npairs, B, seed, groups),
           "pos_per_res": mean_ci(n_pos / L, B, seed + 1, groups), "classes": {}}
    for c in range(N_CLASS):
        out["classes"][TYPES[c]] = {**mean_ci(cls[:, c] / npairs, B, seed + 2 + c, groups), "pairs_total": int(cls[:, c].sum())}
    return out


# ------------------------------------------------------------------------------------------------------------- generated vs measured
def gen_class(state: np.ndarray) -> np.ndarray:
    """Generated state -> class index 0 (none / mask / pad) or 1..11 (type 0..10; the inter states 11..21 fold onto their type: a monomer has no inter pairs)."""
    s = state.astype(np.int64)
    return np.where((s >= 0) & (s <= 21), s % N_CLASS + 1, 0)


def ring_class(state: np.ndarray) -> np.ndarray:
    s = state.astype(np.int64)
    return np.where((s >= 0) & (s < N_CLASS), s + 1, 0)


def gen_type_counts(state: np.ndarray) -> np.ndarray:
    """Positive pairs (i < j) of a generated map per type 0..10 (inter states folded onto their type)."""
    v = state[np.triu_indices(state.shape[0], 1)].astype(np.int64)
    return np.bincount(v[(v >= 0) & (v <= 21)] % N_CLASS, minlength=N_CLASS)[:N_CLASS]


def joint_conf(gc: np.ndarray, rc: np.ndarray) -> np.ndarray:
    return np.bincount(gc * 12 + rc, minlength=144).reshape(12, 12).astype(np.float64)


def design_agreement(gen: np.ndarray, ring: np.ndarray, ca: np.ndarray, donors: list, min_sep: int, long_sep: int, cut: float) -> dict:
    """Everything one design contributes: joint confusions (true, shuffled mean over ``donors`` = RING maps of other designs of the same length), distance baseline."""
    L = gen.shape[0]
    iu, ju = np.triu_indices(L, 1)
    sep = ju - iu
    g, r = gen_class(gen)[iu, ju], ring_class(ring)[iu, ju]
    d = np.linalg.norm(ca[iu] - ca[ju], axis=1)
    d = np.where(np.isfinite(d), d, np.inf)
    gs, rs = gen[iu, ju].astype(np.int64), ring[iu, ju].astype(np.int64)
    bad = ((gs >= N_CLASS) & (gs <= 21)) | (gs == C["RING_MASK"]) | (gs == C["RING_PAD"]) | (gs < 0) | (gs > 24)      # impossible for a FINAL monomer map
    out = {"n_mask": int((gs == C["RING_MASK"]).sum()), "n_gen_inter_state": int(((gs >= N_CLASS) & (gs <= 21)).sum()), "n_invalid": int(bad.sum()),
           "n_gen_pos_all": int(((gs >= 0) & (gs <= 21)).sum()), "n_pairs_all": int(gs.size)}
    for name, lo in (("all", 1), ("gap3", min_sep), ("lr", long_sep)):
        m = sep >= lo
        gm, rm, dm = g[m], r[m], d[m]
        gsm, rsm = gs[m], rs[m]
        exact = np.array([float(((gsm == rsm) & (rsm >= 0) & (rsm < N_CLASS)).sum()), float(((gsm >= 0) & (gsm <= 21)).sum()), float(((rsm >= 0) & (rsm < N_CLASS)).sum())])
        y = rm > 0
        k = int((gm > 0).sum())
        top = np.zeros(m.sum(), dtype=bool)
        if k:
            top[np.argsort(dm, kind="mergesort")[:k]] = True
        pk, pc = top.astype(np.int64), (dm <= cut).astype(np.int64)
        inc = dm <= cut                                   # pairs the geometry already calls a contact: does the label still discriminate?
        wc = np.bincount((gm[inc] > 0).astype(np.int64) * 2 + (rm[inc] > 0).astype(np.int64), minlength=4).astype(np.float64).reshape(2, 2)   # [gen 0/1, ring 0/1]
        # donor shuffles (same pairs, RING map of another design)
        cs = np.zeros((12, 12))
        for rd in donors:
            cs += joint_conf(gm, ring_class(rd)[iu, ju][m])
        if donors:
            cs /= len(donors)
        out[name] = {"conf": joint_conf(gm, rm), "conf_shuf": cs if donors else None, "n_donors": len(donors),
                     "conf_distK": joint_conf(pk, rm), "conf_d8": joint_conf(pc, rm), "within_contact": wc, "exact": exact,
                     "auc_dist": rank_auc(-dm, y), "ap_dist": average_precision(-dm, y), "n_pairs": int(m.sum()), "n_ring_pos": int(y.sum()), "n_gen_pos": k,
                     "ap_dist_cls": [average_precision(-dm, rm == c + 1) for c in range(N_CLASS)] if name == "gap3" else None, "cls_ring": np.bincount(rm, minlength=12)[1:].tolist()}
    return out


def agreement_summary(per: list, B: int, seed: int) -> dict:
    """Pooled agreement (micro) with design-bootstrap CIs, per stratum. ``per`` = list of design_agreement outputs (>= 1)."""
    n = len(per)
    res = {"n_designs": n}
    for s in ("gap3", "lr", "all"):
        C_t = np.stack([p[s]["conf"] for p in per])
        has_sh = all(p[s]["conf_shuf"] is not None for p in per)
        C_s = np.stack([p[s]["conf_shuf"] for p in per]) if has_sh else None
        C_k = np.stack([p[s]["conf_distK"] for p in per])
        C_8 = np.stack([p[s]["conf_d8"] for p in per])
        auc_d = np.array([p[s]["auc_dist"] for p in per])
        ap_d = np.array([p[s]["ap_dist"] for p in per])
        W_c = np.stack([p[s]["within_contact"] for p in per])
        E_t = np.stack([p[s]["exact"] for p in per])

        def stats(idx, C_t=C_t, C_s=C_s, C_k=C_k, C_8=C_8, auc_d=auc_d, ap_d=ap_d, has_sh=has_sh, W_c=W_c, E_t=E_t):
            ct, ck, c8 = C_t[idx].sum(0), C_k[idx].sum(0), C_8[idx].sum(0)
            p, r, f = prf(*conf_binary(ct)[:3])
            tp, fp, fn, tn = conf_binary(ct)
            auc_h = 0.5 * (tp / (tp + fn) + tn / (tn + fp)) if (tp + fn) > 0 and (tn + fp) > 0 else float("nan")
            fk, f8 = conf_f1(ck), conf_f1(c8)
            fs = conf_f1(C_s[idx].sum(0)) if has_sh else float("nan")
            tpc = np.trace(ct[1:, 1:])
            tag = tpc / ct[1:, 1:].sum() if ct[1:, 1:].sum() > 0 else float("nan")
            wc = W_c[idx].sum(0)
            r1 = wc[1, 1] / wc[1].sum() if wc[1].sum() > 0 else float("nan")      # P(RING positive | generated positive, CA-CA <= cut)
            r0 = wc[0, 1] / wc[0].sum() if wc[0].sum() > 0 else float("nan")      # P(RING positive | generated none,     CA-CA <= cut)
            e = E_t[idx].sum(0)                                                   # exact 23-state level: generated state == RING state (type AND intra), micro F1 over the 22 positive states
            f1s = 2 * e[0] / (e[1] + e[2]) if e[1] + e[2] > 0 else float("nan")
            mt = ct[1:, 1:]
            tot = mt.sum()
            if tot > 0:                                                           # type agreement among POSITION-matched positives vs the majority-class baseline (always predict the commonest RING class)
                maj = mt.sum(0).max() / tot
                pe = (mt.sum(0) * mt.sum(1)).sum() / tot ** 2
                kap = (tag - pe) / (1 - pe) if pe < 1 else float("nan")
            else:
                maj = kap = float("nan")
            return [p, r, f, auc_h, fk, f8, fs, f - fs, f - fk, f - max(fk, f8), np.nanmean(auc_d[idx]) if np.isfinite(auc_d[idx]).any() else float("nan"),
                    np.nanmean(ap_d[idx]) if np.isfinite(ap_d[idx]).any() else float("nan"), tag, r1, r0, r1 - r0, f1s, maj, tag - maj, kap]

        names = ["precision", "recall", "f1", "auc_hard", "f1_distK", "f1_dist8", "f1_shuffled", "dF1_shuf", "dF1_distK", "dF1_dist_best", "auc_dist", "ap_dist", "type_agree",
                 "p_ring_given_gen_in_contact", "p_ring_given_nogen_in_contact", "lift_in_contact", "f1_state23", "type_majority_baseline", "dtype_vs_majority", "type_kappa"]
        pt = stats(np.arange(n))
        bs = bootstrap(stats, n, B, seed)
        res[s] = {k: {"value": float(v), "ci95": ci95(bs[:, i]), "dropped_frac": float(np.mean(~np.isfinite(bs[:, i])))} for i, (k, v) in enumerate(zip(names, pt))}
        res[s]["n_informative"] = int(sum(p[s]["n_ring_pos"] > 0 and p[s]["n_gen_pos"] > 0 for p in per))      # designs with RING-measured AND generated positives in this stratum
        res[s]["n_pairs_per_design"] = float(np.mean([p[s]["n_pairs"] for p in per]))
        res[s]["ring_pos_per_design"] = float(np.mean([p[s]["n_ring_pos"] for p in per]))
        res[s]["gen_pos_per_design"] = float(np.mean([p[s]["n_gen_pos"] for p in per]))
        res[s]["n_donors_mean"] = float(np.mean([p[s]["n_donors"] for p in per]))
        res[s]["conf_pooled"] = C_t.sum(0).astype(int).tolist()
        mt = C_t.sum(0)[1:, 1:]                       # rows generated type, cols RING type, POSITION-matched positives only
        res[s]["type_confusion"] = {"rows_generated_cols_ring": list(TYPES), "matrix": mt.astype(int).tolist(), "n_matched": int(mt.sum()),
                                    "by_ring_class": {TYPES[c]: {"n": int(mt[:, c].sum()), "correct": int(mt[c, c]), "acc": float(mt[c, c] / mt[:, c].sum()) if mt[:, c].sum() > 0 else float("nan")} for c in range(N_CLASS) if mt[:, c].sum() > 0}}
        if s == "gap3":
            res[s]["classes"] = {}
            ap_cls = np.array([p[s]["ap_dist_cls"] for p in per], dtype=np.float64)
            for c in range(N_CLASS):
                def cstat(idx, c=c, C_t=C_t):
                    ct = C_t[idx].sum(0)
                    tp = ct[c + 1, c + 1]
                    return list(prf(tp, ct[c + 1, :].sum() - tp, ct[:, c + 1].sum() - tp))
                v, bsc = cstat(np.arange(n)), bootstrap(cstat, n, B, seed + 7 + c)
                ct = C_t.sum(0)
                sup, npred = float(ct[:, c + 1].sum()), float(ct[c + 1, :].sum())
                res[s]["classes"][TYPES[c]] = {"support_ring": int(sup), "n_generated": int(npred), "meaningful": bool(sup >= MIN_CLASS_SUPPORT),
                                               "precision": {"value": v[0], "ci95": ci95(bsc[:, 0])}, "recall": {"value": v[1], "ci95": ci95(bsc[:, 1])},
                                               "f1": {"value": v[2], "ci95": ci95(bsc[:, 2])},
                                               "ap_dist": float(np.nanmean(ap_cls[:, c])) if np.isfinite(ap_cls[:, c]).any() else float("nan")}
    res["informative"] = {s: res[s]["n_informative"] for s in ("gap3", "lr", "all")}
    res["n_mask_pairs"] = int(sum(p["n_mask"] for p in per))
    res["n_gen_inter_state_pairs"] = int(sum(p["n_gen_inter_state"] for p in per))
    n_bad, n_gp = int(sum(p["n_invalid"] for p in per)), int(sum(p["n_gen_pos_all"] for p in per))
    res["states"] = {"invalid_pairs": n_bad, "inter_state_pairs": res["n_gen_inter_state_pairs"], "mask_pairs": res["n_mask_pairs"], "generated_positive_pairs": n_gp,
                     "invalid_frac": n_bad / max(1, n_gp), "designs_with_invalid": int(sum(p["n_invalid"] > 0 for p in per)),
                     "note": "invalid = inter-chain states 11..21 / MASK 23 / PAD 24 off the diagonal in a FINAL monomer map; invalid_frac = invalid pairs / generated positive pairs"}
    return res


VERDICT_KEYS = tuple((st, k) for st in ("gap3", "lr") for k in ("dF1_shuf", "dF1_dist_best", "dtype_vs_majority"))


def verdict(a: dict, comp: dict | None = None, max_invalid_frac: float = 0.0, min_informative: int = 0, max_dropped: float = 0.05) -> dict:
    """Provisional gate-(c) verdict, see the module docstring (no absolute F1 floor). Precedence: INCOMPLETE > INVALID_STATES > PASS / WEAK / FAIL (``verdict_if_valid`` = the rule applied
    ignoring completeness, evidence and invalid states). The gap3 stratum alone is NOT enough: helix i,i+3/4 patterns are generic (shuffled-design F1 0.3-0.45) and a CA-CA <= 8 A rule already
    scores F1 ~0.58 there, so the long-range stratum has to be passed as well; binary agreement alone does not make a 23-state map right: the TYPE agreement among position-matched
    positives has to beat the majority-class baseline (CI of type_agree - majority share > 0) in both strata. EVIDENCE: complete but empty designs (no RING positives or no generated positives)
    count toward completeness only; INCOMPLETE unless >= ``min_informative`` designs carry positive evidence in BOTH strata and no verdict statistic loses more than ``max_dropped`` of its
    bootstrap draws to missing (typed) evidence."""
    def pos(s, k):
        ci = a[s][k]["ci95"]
        return bool(np.isfinite(ci[0]) and ci[0] > 0)
    sp3, spl = pos("gap3", "dF1_shuf"), pos("lr", "dF1_shuf")
    b3, bl = pos("gap3", "dF1_dist_best"), pos("lr", "dF1_dist_best")
    t3, tl = pos("gap3", "dtype_vs_majority"), pos("lr", "dtype_vs_majority")
    base = "PASS" if (sp3 and spl and b3 and bl and t3 and tl) else ("WEAK" if sp3 else "FAIL")
    v, reasons = base, []
    st = a.get("states") or {}
    if st and st.get("invalid_frac", 0.0) > max_invalid_frac:
        v = "INVALID_STATES"
        reasons.append(f"{st['invalid_pairs']} impossible state pairs (inter-chain / MASK / PAD) in {st['designs_with_invalid']} designs = {st['invalid_frac']:.4f} of generated positives > tolerance {max_invalid_frac}")
    info = {s_: a[s_].get("n_informative") for s_ in ("gap3", "lr")}
    dropped = {f"{s_}.{k}": a[s_][k].get("dropped_frac", 0.0) for s_, k in VERDICT_KEYS}
    bad_drop = {k: round(d, 3) for k, d in dropped.items() if not d <= max_dropped}
    ev_reasons = []
    for s_, n_ in info.items():
        if n_ is not None and n_ < min_informative:
            ev_reasons.append(f"only {n_} informative designs in {s_} (RING-measured AND generated positives; need >= {min_informative})")
    if bad_drop:
        ev_reasons.append(f"bootstrap draws lacking evidence exceed {max_dropped:.0%}: {bad_drop}")
    if ev_reasons or (comp is not None and not comp["ok"]):
        v = "INCOMPLETE"
        reasons += ev_reasons
        if comp is not None and not comp["ok"]:
            reasons.append(f"evaluated {comp['n_evaluated']}/{comp['n_archived']} archived designs (need >= {comp['min_frac']:.0%} and >= {comp['min_designs']}): ring failures {comp['n_ring_failed']}, join missing {comp['n_join_missing']}, length mismatch {comp['n_length_mismatch']}")
    return {"verdict": v, "verdict_if_valid": base, "reasons": reasons, "n_informative_gap3": info["gap3"], "n_informative_lr": info["lr"], "max_dropped_draw_frac": float(max(dropped.values())),
            "dropped_draw_frac": dropped, "specific_gap3": sp3, "specific_lr": spl, "beats_distance_gap3": b3, "beats_distance_lr": bl, "beats_distance": bool(b3 and bl),
            "type_beats_majority_gap3": t3, "type_beats_majority_lr": tl, "adds_beyond_contact_gap3": pos("gap3", "lift_in_contact"),
            "rule": "PASS = [F1(true) - F1(shuffled design) CI>0] AND [F1(gen) - max(F1 distK, F1 dist8) CI>0] AND [type_agree - majority-class share CI>0], each in BOTH gap3 and lr; WEAK = specific in gap3 only "
                    "(binary agreement without the rest); FAIL otherwise; INVALID_STATES if impossible states exceed the tolerance; INCOMPLETE if evaluated designs < min fraction / min count of the archive, "
                    "or < min informative designs (RING-measured AND generated positives, both strata), or > 5% of bootstrap draws lack evidence; no absolute F1 floor"}


# ------------------------------------------------------------------------------------------------------------- one run
def tmp_root() -> str:
    root = os.environ.get("GATEC_TMP") or f"/tmp/{os.environ.get('USER') or getpass.getuser()}"
    if os.path.abspath(root).startswith("/home"):
        raise SystemExit(f"GATEC refuse: temp root {root} is under /home (inode quota); set GATEC_TMP to a node-local directory")
    os.makedirs(root, exist_ok=True)
    return root


RUN_STEM = re.compile(r"^(?P<tag>.+)_(?P<mode>masked|generated|stock)_s(?P<seed>\d+)_(?P<len>short|medium|long)_(?P<job>\d+)$")


def run_stem(path: str, kind: str):
    """`tj_s2_designs_<stem>.tar.gz` (kind 'designs') / `tj_s2_ring_<stem>.npz` (kind 'ring') -> (stem, parsed fields or None); stem None when the name does not follow eval_ckpt.sbatch."""
    pat = r"^tj_s2_designs_(.+)\.tar\.gz$" if kind == "designs" else r"^tj_s2_ring_(.+)\.npz$"
    m = re.match(pat, os.path.basename(path))
    if not m:
        return None, None
    f = RUN_STEM.match(m.group(1))
    return m.group(1), (f.groupdict() if f else None)


def join_generated(designs: list, dmap: dict, z: dict, tar: str | None = None, npz: str | None = None, n_total: int | None = None, tar_dups: list | None = None,
                   allow_unverified_names: bool = False) -> dict:
    """IDENTITY join design -> generated map, bound to the SAME RUN. Example ids (`monomer-short_rep0`) repeat across seeds, so the id join alone would accept another seed's maps. The npz must
    belong to the archive: eval_ckpt.sbatch names them `tj_s2_designs_<stem>.tar.gz` / `tj_s2_ring_<stem>.npz` with stem = <tag>_<mode>_s<seed>_<len>_<job>: the stems must be equal (same tag,
    mode, seed, length, job id), the npz metadata (seed, sampler) must agree with them and the npz must hold exactly the archive's number of maps. Then design name -> example id
    (design_map.tsv) -> npz id. Any failure -> INVALID(join): no index fallback, no silent overwrite of repeated ids (npz ids, design_map rows, pdb members), no design without a map."""
    names = [n for n, _ in designs]
    eid = {n: dmap.get(n) for n in names}
    reasons, binding = [], {}
    if tar and npz:
        st_t, f_t = run_stem(tar, "designs")
        st_n, f_n = run_stem(npz, "ring")
        binding.update({"tar_stem": st_t, "npz_stem": st_n})
        if st_t is None or st_n is None:
            if not allow_unverified_names:
                reasons.append(f"cannot bind the npz to the archive: file names do not follow tj_s2_designs_<stem>.tar.gz / tj_s2_ring_<stem>.npz ({os.path.basename(tar)}, {os.path.basename(npz)}); "
                               "--allow-unverified-names skips this check (exploration only)")
        elif st_t != st_n:
            reasons.append(f"npz and archive are from different runs: tar stem '{st_t}' != npz stem '{st_n}' (ids like monomer-short_rep0 repeat across seeds / jobs)")
        f = f_t or f_n
        meta = z.get("meta") or {}
        if f and st_t == st_n:
            binding.update({"seed": int(f["seed"]), "job": f["job"], "mode": f["mode"], "len": f["len"]})
            if meta.get("seed") is not None and int(meta["seed"]) != int(f["seed"]):
                reasons.append(f"npz metadata seed {meta['seed']} != seed {f['seed']} of the archive name")
            sm = meta.get("sampler")
            if sm is not None and f["mode"] in ("masked", "generated") and list(sm) != [f["mode"]]:
                reasons.append(f"npz metadata sampler {sm} != mode '{f['mode']}' of the archive name")
        if meta.get("n_designs") is not None and int(meta["n_designs"]) != len(z["ids"]):
            reasons.append(f"npz metadata n_designs {meta['n_designs']} != {len(z['ids'])} stored maps")
    if n_total is not None and len(z["ids"]) != n_total:
        reasons.append(f"npz holds {len(z['ids'])} maps but the archive has {n_total} designs")
    dup_npz = sorted(i for i, c in collections.Counter(z["ids"]).items() if c > 1)
    if dup_npz:
        reasons.append(f"npz ids repeated (a later map would overwrite an earlier one): {dup_npz[:3]}")
    if tar_dups:
        reasons.append(f"archive repeats members: {tar_dups[:3]}")
    missing = [n for n in names if eid[n] is None or eid[n] not in z["ring"]]
    cnt = collections.Counter(e for e in eid.values() if e is not None)
    dup = sorted(e for e, c in cnt.items() if c > 1)
    if not dmap:
        reasons.append("design_map.tsv missing or empty in the archive")
    if missing:
        reasons.append(f"{len(missing)} design(s) without a map in the npz (e.g. {missing[:3]}; npz ids e.g. {z['ids'][:3]})")
    if dup:
        reasons.append(f"example id(s) used by several designs: {dup[:3]}")
    ok = not reasons
    return {"ok": ok, "status": "design_map.tsv identity join, same run verified" if ok else "INVALID(join)", "reasons": reasons, "n_missing": len(missing), "n_duplicate_ids": len(dup) + len(dup_npz),
            "binding": binding, "eid": eid, "n_npz": len(z["ids"]), "n_npz_unused": len(set(z["ids"]) - set(eid.values()))}


def evaluate_run(name: str, tar: str, npz: str | None, a) -> dict:
    t0 = time.time()
    ex_ = read_design_tar_ex(tar, a.max_designs)
    designs, dmap, n_total = ex_["designs"], ex_["dmap"], ex_["total"]
    work = tempfile.mkdtemp(prefix="gatec_", dir=tmp_root())
    try:
        jobs = []
        for nm, text in designs:
            p = os.path.join(work, nm + ".pdb")
            Path(p).write_text(text)
            jobs.append((nm, p, work, a.ring, a.timeout))
        nw = max(1, min(a.workers, len(jobs)))
        with cf.ProcessPoolExecutor(nw) as ex:
            res = list(ex.map(_ring_job, jobs, chunksize=1))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ok = [r for r in res if r["ok"]]
    fail = {r["name"]: r["err"] for r in res if not r["ok"]}
    out = {"name": name, "tar": tar, "npz": npz, "mode": _mode(name, npz), "n_designs": len(designs), "n_archived": n_total, "n_ring_ok": len(ok), "ring_failures": fail,
           "ring_sec_per_design": float(np.mean([r["sec"] for r in ok])) if ok else None, "workers": nw}
    comp = {"n_archived": n_total, "n_considered": len(designs), "n_ring_ok": len(ok), "n_ring_failed": len(fail), "n_join_missing": 0, "n_length_mismatch": 0,
            "n_evaluated": len(ok), "min_frac": a.min_frac, "min_designs": a.min_designs}
    if not ok:
        out["error"] = "no design could be processed"
        comp["ok"] = False
        out["completeness"] = comp
        return out
    L = np.array([len(r["seq"]) for r in ok])
    npairs = L * (L - 1) / 2
    n_pos = np.array([int((np.triu(r["state"] < N_CLASS, 1)).sum()) for r in ok])
    cls = np.array([np.bincount(r["state"][np.triu_indices(len(r["seq"]), 1)].astype(np.int64).clip(0, 24), minlength=25)[:N_CLASS] for r in ok])
    out["density"] = density_summary(n_pos, cls, L, a.n_boot, a.seed)
    es = collections.Counter()
    for r in ok:
        es.update(r["stats"])
    out["edge_stats"] = dict(es)
    per_design = [{"design": r["name"], "L": int(len(r["seq"])), "ring_pairs": int(n_pos[i]), "ring_density": float(n_pos[i] / npairs[i]), "ring_pos_per_res": float(n_pos[i] / L[i])} for i, r in enumerate(ok)]
    out["gate_c_note"] = "density = RING positive pairs / C(L,2) (length dependent); pos_per_res = positive pairs / L (the length-robust measure); classes = collapsed (strongest) type"
    if npz:
        z = load_ring_states(npz)
        jn = join_generated(designs, dmap, z, tar, npz, n_total, ex_["dups"], getattr(a, "allow_unverified_names", False))
        out["generated"] = {"meta": z["meta"], "n_npz": jn["n_npz"], "n_npz_unused": jn["n_npz_unused"], "join": jn["status"], "join_reasons": jn["reasons"], "binding": jn["binding"],
                            "n_join_missing": jn["n_missing"], "n_duplicate_ids": jn["n_duplicate_ids"]}
        comp["n_join_missing"] = jn["n_missing"]
        if not jn["ok"]:
            comp["ok"] = False
            out["gate_c_verdict"] = {"verdict": "INVALID(join)", "reasons": jn["reasons"], "rule": "the npz must belong to the same run as the archive (file-name stem, metadata seed / sampler, map count) and design_map.tsv must give every design exactly one npz map; no index fallback"}
            out["completeness"] = comp
            out["per_design"] = per_design
            out["sec"] = time.time() - t0
            return out
        gen = {r["name"]: z["ring"][jn["eid"][r["name"]]] for r in ok}
        use = [(i, r) for i, r in enumerate(ok) if gen[r["name"]].shape[0] == len(r["seq"])]
        comp["n_length_mismatch"] = len(ok) - len(use)
        out["generated"]["n_used"] = len(use)
        out["generated"]["n_length_mismatch"] = comp["n_length_mismatch"]
        comp["n_evaluated"] = len(use)
        if use:
            gpos = np.array([int(np.triu(gen[r["name"]].astype(np.int64) <= 21, 1).sum()) for _, r in use])
            gcls = np.array([gen_type_counts(gen[r["name"]]) for _, r in use])
            Lu = np.array([len(r["seq"]) for _, r in use])
            out["generated"]["density"] = density_summary(gpos, gcls, Lu, a.n_boot, a.seed + 100)
            by_len = collections.defaultdict(list)
            for k, (_, r) in enumerate(use):
                by_len[len(r["seq"])].append(k)
            per = []
            for k, (_, r) in enumerate(use):
                grp = by_len[len(r["seq"])]
                pos = grp.index(k)
                donors = [use[grp[(pos + s) % len(grp)]][1]["state"] for s in range(1, min(a.shifts, len(grp) - 1) + 1)]
                per.append(design_agreement(gen[r["name"]], r["state"], r["ca"], donors, a.min_sep, a.long_sep, a.cut))
            out["agreement"] = agreement_summary(per, a.n_boot, a.seed + 200)
            comp["frac"] = comp["n_evaluated"] / max(1, n_total)
            comp["ok"] = bool(comp["n_evaluated"] >= a.min_designs and comp["frac"] >= a.min_frac)
            out["agreement"]["verdict"] = out["gate_c_verdict"] = verdict(out["agreement"], comp, a.max_invalid_frac, a.min_informative, a.max_dropped)
            for k, (i, r) in enumerate(use):
                per_design[i].update({"gen_pairs": int(gpos[k]), "f1_gap3": prf(*conf_binary(per[k]["gap3"]["conf"])[:3])[2], "auc_dist_gap3": per[k]["gap3"]["auc_dist"]})
        else:
            comp["ok"] = False
            out["gate_c_verdict"] = {"verdict": "INCOMPLETE", "reasons": ["no design with a length-matched generated map"]}
    comp.setdefault("frac", comp["n_evaluated"] / max(1, n_total))
    comp.setdefault("ok", bool(comp["n_evaluated"] >= a.min_designs and comp["frac"] >= a.min_frac))
    out["completeness"] = comp
    out["per_design"] = per_design
    out["sec"] = time.time() - t0
    return out


def _mode(name: str, npz) -> str:
    for m in ("masked", "generated", "stock"):
        if f"_{m}_" in f"_{name}_" or name.endswith("_" + m):
            return m
    return "generated" if npz else "unknown"


def length_dist(Ls) -> dict:
    Ls = np.asarray(list(Ls), dtype=np.int64)
    if not Ls.size:
        return {"n": 0}
    h = collections.Counter(int(x) // 10 * 10 for x in Ls)
    return {"n": int(Ls.size), "mean": float(Ls.mean()), "median": float(np.median(Ls)), "min": int(Ls.min()), "max": int(Ls.max()), "hist_bin10": {str(k): h[k] for k in sorted(h)}}


def length_windows(lengths, frac: float) -> dict:
    """{design length L: [floor((1-frac) L), ceil((1+frac) L)]}: the natural chains a design of length L is compared with (L=100, frac 0.2 -> 80..120)."""
    return {int(L): [int(math.floor((1 - frac) * L)), int(math.ceil((1 + frac) * L))] for L in sorted({int(x) for x in lengths})}


def vs_natural(run: dict, nat_rows: list, B: int, seed: int, frac: float = 0.2) -> dict | None:
    """Run / natural ratios, LENGTH MATCHED: every design of length L is compared with the natural chains of length within L(1 +- frac); the natural mean is the average of the window
    means weighted by the share of designs of each length. Reported for positives per residue (primary: nearly length independent) and for density per C(L,2) (length dependent: only
    meaningful because of the matching). 95 % CI: designs resampled for the run, SEQUENCE clusters for the natural chains."""
    if "density" not in run or not nat_rows:
        return None
    pdz = run["per_design"]
    Ld = np.array([d["L"] for d in pdz], dtype=np.float64)
    posd = np.array([d["ring_pairs"] for d in pdz], dtype=np.float64)
    win = length_windows(Ld, frac)
    keys = sorted(win)
    Ln = np.array([r["L"] for r in nat_rows], dtype=np.float64)
    posn = np.array([r["n_pos"] for r in nat_rows], dtype=np.float64)
    members = [(Ln >= win[k][0]) & (Ln <= win[k][1]) for k in keys]
    out = {"frac": frac, "windows": {str(k): win[k] + [int(m.sum())] for k, m in zip(keys, members)}, "windows_note": "[lo, hi, natural chains in the window]"}
    if any(m.sum() == 0 for m in members):
        out["error"] = "no natural chains in the length window of L=" + ",".join(str(k) for k, m in zip(keys, members) if m.sum() == 0)
        return out
    uniq, inv = np.unique([r["cluster"] for r in nat_rows], return_inverse=True)
    gk = np.array([keys.index(int(x)) for x in Ld])
    for name, vd, vn in (("pos_per_res", posd / Ld, posn / Ln), ("density", posd / (Ld * (Ld - 1) / 2), posn / (Ln * (Ln - 1) / 2))):
        S, Cn = np.zeros((uniq.size, len(keys))), np.zeros((uniq.size, len(keys)))
        for kk, m in enumerate(members):
            np.add.at(S[:, kk], inv[m], vn[m])
            np.add.at(Cn[:, kk], inv[m], 1.0)

        def est(didx, cidx):
            w = np.bincount(gk[didx], minlength=len(keys)) / len(didx)
            c = Cn[cidx].sum(0)
            if np.any((c == 0) & (w > 0)):
                return float("nan"), float("nan")
            nm = np.where(c > 0, S[cidx].sum(0) / np.maximum(c, 1), 0.0)
            nat = float((w * nm).sum())
            return float(vd[didx].mean()), nat

        dm, nm = est(np.arange(Ld.size), np.arange(uniq.size))
        rng = np.random.default_rng(seed + (0 if name == "pos_per_res" else 1))
        bs = np.empty(B)
        for b in range(B):
            d_, n_ = est(rng.integers(0, Ld.size, Ld.size), rng.integers(0, uniq.size, uniq.size))
            bs[b] = d_ / n_ if n_ and n_ == n_ else float("nan")
        out[name] = {"ratio": dm / nm, "ci95": ci95(bs), "design_mean": dm, "natural_matched_mean": nm}
    out["ratio"], out["ci95"] = out["pos_per_res"]["ratio"], out["pos_per_res"]["ci95"]
    return out


# ------------------------------------------------------------------------------------------------------------- converter validation
def validate_converter(n: int, ring_bin: str, workers: int, manifest: str = MANIFEST_VAL, lmdb_path: str = LMDB_PATH) -> dict:
    """Run RING on the CIFs of ``n`` natural val chains (<= 12k atoms), convert raw edges with ``edge_type_index`` and compare the collapsed pair states with the lmdb record
    of the same chain. Chain numbering differs between the lmdb (label_seq_id) and RING (auth) for some entries: a constant offset is searched; chains without a >= 0.9 pair-set
    Jaccard are reported as `numbering_mismatch` and not scored."""
    import lmdb
    import pyarrow.parquet as pq

    ref = natural_reference(manifest, lmdb_path, max_chains=0)
    t = pq.read_table(manifest, columns=["pdb_id", "pn_unit_iid", "ring_key"]).to_pydict()
    key2 = {k: (p, u) for p, u, k in zip(t["pdb_id"], t["pn_unit_iid"], t["ring_key"])}
    idx = pq.read_table(TAR_INDEX, columns=["pdb_id", "offset_data", "size"]).to_pydict()
    pos = {p: (o, s) for p, o, s in zip(idx["pdb_id"], idx["offset_data"], idx["size"])}
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, subdir=True)
    work = tempfile.mkdtemp(prefix="gatec_val_", dir=tmp_root())
    rows = [r for r in ref["rows"]][: max(n * 3, n)]
    res = collections.Counter()
    types_cm = collections.Counter()
    try:
        done = 0
        for r in rows:
            if done >= n:
                break
            pdb_id, unit = key2[r["id"]]
            o, s = pos[pdb_id]
            with open(TAR_PATH, "rb") as fh:
                fh.seek(int(o))
                txt = gzip.decompress(fh.read(int(s))).decode()
            if txt.count("\nATOM") > 12000:
                continue
            cif = os.path.join(work, pdb_id + ".cif")
            Path(cif).write_text(txt)
            try:
                edges = parse_ring_edges(run_ring(cif, os.path.join(work, "o_" + pdb_id), ring_bin, 120))
            except Exception:   # noqa: BLE001
                res["ring_failed"] += 1
                continue
            done += 1
            ch = unit.split("_")[0]
            ii, jj, tt = [], [], []
            for n1, it, n2 in edges:
                if n1[0] != ch or n2[0] != ch or n1[3] not in STD_AA or n2[3] not in STD_AA:
                    continue
                k = edge_type_index(it, n1[3], n2[3])
                if k is not None:
                    ii.append(n1[1])
                    jj.append(n2[1])
                    tt.append(k)
            mi, mj, mt = collapse_pairs(np.array(ii), np.array(jj), np.array(tt))
            with env.begin() as txn:
                rec = pickle.loads(txn.get(r["id"].encode()))
            p = pairs_from_record(rec, unit)
            ref_set = {(a, b): c for a, b, c in zip(p["i"].tolist(), p["j"].tolist(), p["t"].tolist())}
            best = (-1.0, 0)
            for off in range(-600, 601):
                raw = {(a + off, b + off) for a, b in zip(mi.tolist(), mj.tolist())}
                jac = len(raw & set(ref_set)) / max(1, len(raw | set(ref_set)))
                if jac > best[0]:
                    best = (jac, off)
            if best[0] < 0.9:
                res["numbering_mismatch"] += 1
                continue
            res["chains_scored"] += 1
            mine = {(a + best[1], b + best[1]): c for a, b, c in zip(mi.tolist(), mj.tolist(), mt.tolist())}
            both = set(mine) & set(ref_set)
            res["pairs_ref"] += len(ref_set)
            res["pairs_mine"] += len(mine)
            res["pairs_both"] += len(both)
            res["pairs_same_class"] += sum(mine[k] == ref_set[k] for k in both)
            for k in both:
                types_cm[(TYPES[ref_set[k]], TYPES[mine[k]])] += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out = dict(res)
    out["pair_jaccard"] = res["pairs_both"] / max(1, res["pairs_ref"] + res["pairs_mine"] - res["pairs_both"])
    out["class_agreement_on_shared_pairs"] = res["pairs_same_class"] / max(1, res["pairs_both"])
    out["class_disagreements"] = {f"{a}->{b}": c for (a, b), c in types_cm.items() if a != b}
    return out


# ------------------------------------------------------------------------------------------------------------- output
def _f(x, nd=3) -> str:
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _v(d: dict, nd=3) -> str:
    return f"{_f(d['value'], nd)} [{_f(d['ci95'][0], nd)},{_f(d['ci95'][1], nd)}]"


def gatec_lines(js: dict) -> list:
    out = []
    nat = js.get("natural") or {}
    if nat.get("density"):
        d = nat["density"]
        ld = nat.get("matched_length_distribution") or {}
        out.append(f"GATEC natural LENGTH-MATCHED n={d['n']} clusters={d['density'].get('n_clusters')} L mean={_f(ld.get('mean'), 1)} median={_f(ld.get('median'), 0)} range={ld.get('min')}-{ld.get('max')} "
                   f"pos_per_res={_f(d['pos_per_res']['mean'], 3)} [{_f(d['pos_per_res']['ci95'][0], 3)},{_f(d['pos_per_res']['ci95'][1], 3)}] density/C(L,2)={_f(d['density']['mean'], 4)} "
                   f"[{_f(d['density']['ci95'][0], 4)},{_f(d['density']['ci95'][1], 4)}] windows={nat.get('windows')} | "
                   + " ".join(f"{k}={_f(v['mean'], 4)}" for k, v in d["classes"].items() if v["pairs_total"] > 0))
        out.append(f"GATEC natural_lengths loaded={ (nat.get('length_distribution') or {}).get('n')} hist_bin10={(nat.get('length_distribution') or {}).get('hist_bin10')}")
    elif nat:
        out.append(f"GATEC natural UNAVAILABLE: {nat.get('error')}")
    for name, r in js["runs"].items():
        if "density" not in r:
            out.append(f"GATEC density run={name} FAILED: {r.get('error')} ring_failures={len(r.get('ring_failures', {}))}")
            continue
        d = r["density"]
        cl = " ".join(f"{k}={_f(v['mean'], 4)}" for k, v in d["classes"].items() if v["pairs_total"] > 0)
        rn = r.get("vs_natural")
        cm = r.get("completeness") or {}
        rnt = ""
        if rn and "pos_per_res" in rn:
            rnt = (f" vs_natural(length-matched +-{rn['frac']:.0%}) pos_per_res={_f(rn['pos_per_res']['ratio'], 2)}x [{_f(rn['pos_per_res']['ci95'][0], 2)},{_f(rn['pos_per_res']['ci95'][1], 2)}] "
                   f"density/C(L,2)={_f(rn['density']['ratio'], 2)}x [{_f(rn['density']['ci95'][0], 2)},{_f(rn['density']['ci95'][1], 2)}]")
        elif rn:
            rnt = f" vs_natural={rn.get('error')}"
        out.append(f"GATEC density run={name} mode={r['mode']} ring_ok={r['n_ring_ok']}/{r.get('n_archived', r['n_designs'])} pos_per_res={_f(d['pos_per_res']['mean'], 3)} [{_f(d['pos_per_res']['ci95'][0], 3)},"
                   f"{_f(d['pos_per_res']['ci95'][1], 3)}] density/C(L,2)={_f(d['density']['mean'], 4)} [{_f(d['density']['ci95'][0], 4)},{_f(d['density']['ci95'][1], 4)}]" + rnt + f" | {cl}")
        g = r.get("generated") or {}
        if g.get("density"):
            out.append(f"GATEC generated_density run={name} density={_f(g['density']['density']['mean'], 4)} [{_f(g['density']['density']['ci95'][0], 4)},"
                       f"{_f(g['density']['density']['ci95'][1], 4)}] (labels of the npz, same definition) used={g['n_used']}")
        a = r.get("agreement")
        v0 = r.get("gate_c_verdict")
        if v0 and not a:
            out.append(f"GATEC verdict run={name} {v0['verdict']} {'; '.join(v0.get('reasons') or [])} | completeness={json.dumps(cm)}")
        if a:
            for s in ("gap3", "lr", "all"):
                x = a[s]
                out.append(f"GATEC agree run={name} stratum={s} n={a['n_designs']} P={_v(x['precision'])} R={_v(x['recall'])} F1={_v(x['f1'])} | distK_F1={_v(x['f1_distK'])} dist8_F1={_v(x['f1_dist8'])} "
                           f"AUC_dist={_v(x['auc_dist'])} AP_dist={_v(x['ap_dist'])} AUC_hard={_f(x['auc_hard']['value'])} | shuf_F1={_v(x['f1_shuffled'])} dF1_shuf={_v(x['dF1_shuf'])} "
                           f"dF1_distK={_v(x['dF1_distK'])} dF1_dist_best={_v(x['dF1_dist_best'])} | in_contact(d<={js['params']['dist_cut']:g}A) "
                           f"P(ring|gen)={_v(x['p_ring_given_gen_in_contact'])} P(ring|no gen)={_v(x['p_ring_given_nogen_in_contact'])} lift={_v(x['lift_in_contact'])}")
                if s != "all":
                    out.append(f"GATEC types run={name} stratum={s} (23-state level) exact_state_F1={_v(x['f1_state23'])} | type agreement among position-matched positives {_v(x['type_agree'])} vs majority-class baseline "
                             f"{_v(x['type_majority_baseline'])} -> d={_v(x['dtype_vs_majority'])} kappa={_v(x['type_kappa'])} n_matched={x['type_confusion']['n_matched']} by_ring_class="
                             + " ".join(f"{k}:{c['correct']}/{c['n']}" for k, c in x["type_confusion"]["by_ring_class"].items()))
            v, st = a["verdict"], a["states"]
            out.append(f"GATEC verdict run={name} {v['verdict']} (if valid: {v['verdict_if_valid']}) specific_gap3={v['specific_gap3']} specific_lr={v['specific_lr']} beats_distance_gap3={v['beats_distance_gap3']} "
                       f"beats_distance_lr={v['beats_distance_lr']} type_beats_majority gap3={v['type_beats_majority_gap3']} lr={v['type_beats_majority_lr']} adds_beyond_contact={v['adds_beyond_contact_gap3']} | "
                       f"evaluated {cm.get('n_evaluated')}/{cm.get('n_archived')} (ring failures {cm.get('n_ring_failed')}, join missing {cm.get('n_join_missing')}, length mismatch {cm.get('n_length_mismatch')}) | "
                       f"n_informative gap3={v['n_informative_gap3']} lr={v['n_informative_lr']} max_dropped_draws={v['max_dropped_draw_frac']:.3f} | "
                       f"invalid_states={st['invalid_pairs']} (inter={st['inter_state_pairs']} mask={st['mask_pairs']}; {st['designs_with_invalid']} designs; {st['invalid_frac']:.4f} of generated positives)"
                       + (" | " + "; ".join(v["reasons"]) if v["reasons"] else ""))
            for cname, c in a["gap3"]["classes"].items():
                if c["meaningful"]:
                    out.append(f"GATEC class run={name} {cname} support={c['support_ring']} generated={c['n_generated']} P={_v(c['precision'])} R={_v(c['recall'])} F1={_v(c['f1'])} AP_dist={_f(c['ap_dist'])}")
    if js.get("validate_converter"):
        out.append("GATEC converter " + " ".join(f"{k}={_f(v, 4) if isinstance(v, float) else v}" for k, v in js["validate_converter"].items() if k != "class_disagreements")
                   + f" disagreements={js['validate_converter'].get('class_disagreements')}")
    return out


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def parse_run(spec: str):
    name, eq, rest = spec.partition("=")
    if not eq or "/" in name:
        name, rest = "", spec
    tar, _, npz = rest.partition(",")
    return (name or run_name_from_tar(tar)), tar, (npz or None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="[NAME=]TAR[,NPZ]", help="design archive (+ generated-label npz for generated runs); repeatable")
    ap.add_argument("--json", default="", help="the ONE output json")
    ap.add_argument("--ring", default=os.environ.get("GATEC_RING", RING_BIN_DEFAULT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-designs", type=int, default=0, help="first N designs of each archive (0 = all); stated in the json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shifts", type=int, default=8, help="donor designs of the shuffled control (cyclic, same length)")
    ap.add_argument("--min-sep", type=int, default=MIN_SEP)
    ap.add_argument("--long-sep", type=int, default=LONG_SEP)
    ap.add_argument("--cut", type=float, default=DIST_CUT)
    ap.add_argument("--nat-window", type=float, default=0.2, help="natural chains compared with a design of length L: L(1 -/+ window) (default 0.2: L=100 -> 80..120)")
    ap.add_argument("--nat-range", type=int, nargs=2, default=None, help="override the natural length range (NOT length matched; flagged in the json)")
    ap.add_argument("--no-natural", action="store_true")
    ap.add_argument("--min-designs", type=int, default=30, help="verdict INCOMPLETE unless at least this many designs were evaluated ...")
    ap.add_argument("--min-frac", type=float, default=0.9, help="... and at least this fraction of the ARCHIVED designs (RING failures, join / length mismatches and --max-designs count against it)")
    ap.add_argument("--min-informative", type=int, default=30, help="verdict INCOMPLETE unless at least this many designs carry positive evidence (RING-measured AND generated positives) in BOTH strata")
    ap.add_argument("--max-dropped", type=float, default=0.05, help="verdict INCOMPLETE when more than this fraction of bootstrap draws of a verdict statistic lack evidence (NaN)")
    ap.add_argument("--allow-unverified-names", action="store_true", help="exploration only: join an npz whose file name does not follow eval_ckpt.sbatch (the same-run binding by file-name stem cannot be checked)")
    ap.add_argument("--max-invalid-frac", type=float, default=0.0, help="verdict INVALID_STATES when impossible states (inter-chain 11..21, MASK, PAD) exceed this fraction of the generated positive pairs")
    ap.add_argument("--validate-converter", type=int, default=0, metavar="N", help="check the edge->label conversion on N natural chains against the lmdb")
    a = ap.parse_args(argv)
    if not os.path.exists(a.ring):
        raise SystemExit(f"GATEC error: RING binary not found: {a.ring} (set --ring / GATEC_RING)")
    t0 = time.time()
    js = {"schema": "gate_c/v1", "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "host": os.uname().nodename,
          "ring": {"binary": a.ring, "flags": list(RING_FLAGS), "version": ring_version(a.ring), "version_expected_lmdb": RING_VERSION},
          "params": {"min_sep": a.min_sep, "long_sep": a.long_sep, "dist_cut": a.cut, "n_boot": a.n_boot, "seed": a.seed, "shifts": a.shifts, "max_designs": a.max_designs,
                     "nat_window": a.nat_window, "nat_range_override": list(a.nat_range) if a.nat_range else None, "min_designs": a.min_designs, "min_frac": a.min_frac, "min_informative": a.min_informative, "max_dropped": a.max_dropped,
                     "allow_unverified_names": a.allow_unverified_names, "max_invalid_frac": a.max_invalid_frac, "class_support_min": MIN_CLASS_SUPPORT, "hydrophobic_residues": sorted(HYDROPHOBIC_RES)},
          "runs": {}}
    for spec in a.run:
        name, tar, npz = parse_run(spec)
        js["runs"][name] = evaluate_run(name, tar, npz, a)
    nat_rows = []
    if not a.no_natural:
        try:
            lengths = [d["L"] for r in js["runs"].values() for d in r.get("per_design", [])] or [100]
            wins = length_windows(lengths, a.nat_window)
            lo, hi = (a.nat_range if a.nat_range else (min(w[0] for w in wins.values()), max(w[1] for w in wins.values())))
            ref = natural_reference(lo=lo, hi=hi)
            all_rows = ref["rows"]
            nat_rows = all_rows if a.nat_range else [r for r in all_rows if any(w[0] <= r["L"] <= w[1] for w in wins.values())]
            js["natural"] = {"density": density_summary([r["n_pos"] for r in nat_rows], [r["cls"] for r in nat_rows], [r["L"] for r in nat_rows], a.n_boot, a.seed + 500,
                                                       groups=[r["cluster"] for r in nat_rows]),
                             "length_matched": not a.nat_range, "windows": {str(k): v for k, v in wins.items()}, "length_distribution": length_dist(r["L"] for r in all_rows),
                             "matched_length_distribution": length_dist(r["L"] for r in nat_rows), "design_length_distribution": length_dist(lengths),
                             "skipped": ref["skipped"], "manifest": ref["manifest"], "lmdb": ref["lmdb"], "length_range": ref["length_range"]}
        except Exception as e:   # noqa: BLE001 - the reference is optional: say so, never silently
            js["natural"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        for i, (name, r) in enumerate(js["runs"].items()):
            r["vs_natural"] = vs_natural(r, nat_rows, a.n_boot, a.seed + 900 + i, a.nat_window)
    if a.validate_converter:
        js["validate_converter"] = validate_converter(a.validate_converter, a.ring, a.workers)
    js["sec"] = time.time() - t0
    lines = gatec_lines(js)
    js["lines"] = lines
    print("\n".join(lines))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(_clean(js), fh, indent=1)
        print(f"GATEC wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
