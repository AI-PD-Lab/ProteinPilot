"""TRIAD-R data transform: RING residue-pair labels attached to RFD4 training batches.

``RingDataset`` builds a sample from a PDB mmCIF with RFD4's OWN pipeline (``atomworks.io.parse(RCSB_WITH_METALS)`` ->
``build_design_train_pipeline``; crop, random rotation, atomisation, unindexing are all done by RFD4) and adds two token-space
``[I, I]`` tensors to ``sample['feats']`` AFTER the pipeline (so after crop / atomisation / unindexing):

  ring_state_gt  long   0..21 RING contact state, 22 NO_INTERACTION, ``RING_UNOBS`` (= 24 = RING_PAD) for every pair that is
                        not observed (and for the diagonal). Symmetric.
  ring_obs       bool   True iff a successfully read RING record demonstrably examined BOTH residues (no-coverage-leak rule),
                        or the pair carries a label. Symmetric, diagonal False. ``ring_state_gt == 22`` only where ring_obs.

Label namespace (identical to pair_stream.py / TRIAD_v3 constants): 0..10 = hbond, salt_bridge, ionic, hydrophobic, pi_stack,
pi_cation, disulfide, vdw, metal, halogen, pi_hbond for the SAME pn_unit (intra); 11..21 the same eleven for DIFFERENT
pn_units (inter); 22 NO_INTERACTION; 23 = MASK (model input only, never produced here); 24 = PAD / not applicable = UNOBS.
A pair with several interaction types is collapsed to the strongest one (COLLAPSE_PRIORITY, as in TRIAD_v3).

How tokens are tied to RING records (exact, no heuristics on the token side)
---------------------------------------------------------------------------
1. The CIF is parsed with the same call RFD4 uses; the assembly of the manifest row is taken (``assembly_id``), reduced to the
   row's pn_unit(s) (monomer row: ``pn_unit_iid``; interface row: ``pn_unit_1_iid`` + ``pn_unit_2_iid``).
2. Every residue of that array gets an ordinal; it is written to the atom array as the annotation ``ring_uid`` BEFORE the RFD4
   pipeline. Annotations survive RFD4's crop / atomisation / unindexing, so after the pipeline each token knows exactly which
   pre-pipeline residue it is (a token whose atoms disagree raises).
3. RING entries are joined to residues by IDENTITY: (``chain_k`` = pn_unit_iid, ``resnum_k`` = res_id, ``resname_k``). Verified on
   real data to be exact: RING was built on the same atomworks parse. Each joined side is additionally checked against the
   ENTRY COORDINATE (RING stores coord_k in the assembly frame of the CIF): the coordinate must lie within a tolerance of an atom
   of that very residue, otherwise the side is rejected and counted (never guessed, no fall-back to a different residue).
   An entry with both residues identified but a rejected coordinate leaves its pair UNKNOWN (``ring_state_gt`` = UNOBS, never
   NO_INTERACTION, even if another entry labels the pair). Only standard-amino-acid tokens carry labels (``include_ncaa=False``);
   atomised tokens, unindexed copies (``insertion_source_idx >= 0``), modified residues, ligands and nucleic acids are UNOBS.
4. Labels and coverage live in residue-ordinal space; ``labels_to_tokens`` slices them into token space once the pipeline has
   decided what survives. A pair whose residue was cropped / unresolved / atomised simply does not exist in token space: it is
   neither labelled nor "no interaction".
5. Coverage (``ring_obs``): per RING record the residues that appear in at least one of its joined entries ("touched") are
   certified, per pn_unit block, if the block has >= 2 touched residues (TRIAD_v3 MIN_TOUCHED_TOKENS_PER_BLOCK); a pair is observed
   iff both residues are certified by the SAME record (cross-block always, same block only if the record's metadata says that
   block's intra edges are in it) or the pair carries a label. No filling of untouched runs (MAX_UNEXAMINED_RUN = 0).

Replay mode (``ring_labels=False``): the SAME pipeline, rows and cropping, but no RING record is read and no ring key is written
(the collate zero-fills ``ring_state_gt`` = 24 / ``ring_obs`` = False, so the flow matcher shows an all-MASK stream and trains no ring
loss on those rows). S2 fine-tuning mixes ``ring_labels=True`` rows with ``ring_labels=False`` rows of the original design-task mix.

Hydra / trainer interface (rfproteina.train builds the loader from ``datasets.train.<leaf>.dataset`` with ``hydra.utils.instantiate``):
``name`` -> ``sample['dataset_name']``; ``transform`` (mapping, merged into the pipeline kwargs: its ``path_length_search_cutoff`` is what
``_adopt_bundle_architecture`` reads); ``design_tasks`` (a plain dict or a Hydra-instantiated preset node); ``.data`` (DataFrame with
``cluster`` / ``nres`` aligned with the dataset indices, for ``calculate_weights_by_inverse_cluster_size``); ``split='train'|'val'|None``.

Splits (deterministic, cluster-disjoint). ``split='val'`` reads the val manifest (``pdb_afdb_val_clustersplit.parquet``, de-duplicated,
ranked by a sha1 of the sequence-cluster id, first ``n_val`` rows); ``split='train'`` reads the train manifest MINUS every row whose
sequence-cluster id (the part of ``cluster`` before the first ``+``, so ``abc+HEM`` and ``abc+CA`` are one cluster) occurs anywhere in the
val manifest. P0 only removed the exact cluster strings of its 1000-row val subset; this is a superset (6 sequence clusters are shared
between the two manifests under different ligand suffixes). ``val_fraction=f`` instead splits ONE manifest by hash(cluster id) < f.

Model-side rule (no coverage leak): ``ring_obs`` (which residues "have some interaction") must NEVER be visible in the network
input. ``make_ring_state_input`` implements the rule of ``ring_flow.corrupt_batch``: reveal locations are drawn over ALL valid
off-diagonal pairs independent of coverage, a revealed unobserved pair shows NO_INTERACTION, hidden = MASK, padding/diagonal = PAD.
``ring_obs`` is for the LOSS mask only (observed AND hidden pairs).

Import side effect: ``register_ring_specs()`` adds ``ring_state_gt`` / ``ring_obs`` to ``rfproteina.datapipes.schema.proteina.SPECS``
(needed so ``collate_batch`` pads them as ``[I, I]``). Spawned DataLoader workers unpickle the dataset, which imports this module,
which re-registers. No upstream file is edited.

Running (login node / any node, CPU): the SIF python has atomworks + rfproteina but NOT lmdb / pytest. One-off, node-local or any dir
with a few files:  ``apptainer exec <sif> python -m pip install --target $PYLIBS lmdb pytest``  (lmdb 3.0.0 / pytest 9.1.1 used here).
The raw PDB mirror is not at /home/atuin/.../pdb_mirror any more; the same files exist as a tar (+ offset index) read in place:

  PYLIBS=<dir with lmdb>
  apptainer exec --bind /path/to/work:/w \
     --bind /path/to/work_rfd3:/mir:ro --bind /path/to/home/Training_Triad/RFD4_J:/fork \
     --pwd /fork /path/to/work/rfd4-cu13.sif env CLUSTER=local DATA_PATH=/w HF_HUB_OFFLINE=1 \
     TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/fork/src:/fork:$PYLIBS python ...

(``/w``-style defaults fall back to the host paths when the binds are absent.) The filtered manifest table is cached node-locally in
``$RING_ROWS_CACHE_DIR`` (default the system temp dir) as ONE npz file; ``RING_ROWS_CACHE=0`` disables it.
"""
from __future__ import annotations

import gzip
import io
import logging
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

# --------------------------------------------------------------------------------------------------------------------------
# constants (model side: triadj_ext/pair_stream.py: 0..22 states, 23 MASK, 24 PAD; asserted equal in tests/test_ring_data.py)
# --------------------------------------------------------------------------------------------------------------------------
RING_N_STATES = 23
NO_INTERACTION = 22
RING_MASK = 23
RING_PAD = 24
RING_UNOBS = RING_PAD          # "not observed / not a pair": the only code the model never learns as a target

RING_INTERACTION_TYPES = ("hbond", "salt_bridge", "ionic", "hydrophobic", "pi_stack", "pi_cation",
                          "disulfide", "vdw", "metal", "halogen", "pi_hbond")
N_TYPES = len(RING_INTERACTION_TYPES)                                 # 11 ; state = type + 11 * (0 intra | 1 inter)
COLLAPSE_PRIORITY = ("disulfide", "metal", "salt_bridge", "ionic", "pi_cation",
                     "pi_stack", "pi_hbond", "hbond", "halogen", "hydrophobic", "vdw")      # strongest first
TYPE_RANK = np.array([COLLAPSE_PRIORITY.index(t) for t in RING_INTERACTION_TYPES], dtype=np.int64)
#: RING record key -> interaction type. `ionic` is deliberately absent: it duplicates `salt_bridges` (re-checked on 5 fresh records
#: in tests, TRIAD_v3 measured identical pair sets in 4000/4000); `ionic_supplemented` carries the genuine ionic entries.
RING_RECORD_TO_TYPE = {
    "hbonds": "hbond", "salt_bridges": "salt_bridge", "ionic_supplemented": "ionic", "hydrophobic": "hydrophobic",
    "pi_stacking": "pi_stack", "pi_cation": "pi_cation", "pi_cation_supplemented": "pi_cation", "disulfides": "disulfide",
    "vdw": "vdw", "metal_ion": "metal", "halogen": "halogen", "pi_hbond": "pi_hbond",
}
#: records whose entries carry a ring CENTROID as coord_k (coordinate equals no atom)
PI_RECORDS = frozenset({"pi_stacking", "pi_cation", "pi_cation_supplemented", "pi_hbond"})
#: records whose coord_k is a group CENTROID rather than an atom (measured on 185,730 sides: ring centroids 1.1-1.4 A from the nearest
#: atom, salt-bridge charge-group centroids up to 3.0 A; every atom-valued record type coincides with an atom to 0.000 A in 99.95%)
CENTROID_RECORDS = PI_RECORDS | {"salt_bridges", "ionic_supplemented"}
RESNAME_ALIASES = {"MSE": "MET"}           # atomworks convert_mse_to_met
AMBIGUITY_MARGIN = 0.5                      # A: best candidate must beat the runner-up by this much, else the side FAILS (UNOBS)
MIN_TOUCHED_PER_BLOCK = 2
TOL_ATOM = 1.0                              # A: entry coordinate to nearest atom of the SAME residue (atom-valued records)
TOL_PI = 3.3                                # A: same for centroid-valued records (CENTROID_RECORDS)
TOKEN_TYPE_PROTEIN, TOKEN_TYPE_NCAA = 0, 6  # rfproteina.encoding.TOKEN_TYPE_IDS

TRAIN_MANIFEST = "pdb_afdb_train_clustersplit.parquet"
VAL_MANIFEST = "pdb_afdb_val_clustersplit.parquet"
_DEFAULT_ROOTS = {
    "manifest_dir": ("/w/manifests", "/path/to/work/manifests"),
    "ring_lmdb": ("/w/ring_pipeline/ring_interactions.lmdb", "/path/to/work/ring_pipeline/ring_interactions.lmdb"),
    "tar": ("/mir/pdb_mirror.tar", "/path/to/work_rfd3/pdb_mirror.tar"),
    "tar_index": ("/mir/pdb_mirror_tar_index.parquet", "/path/to/work_rfd3/pdb_mirror_tar_index.parquet"),
}
SCHEMA = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"


def _first_existing(*paths: str) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {paths} exists (bind the data dirs, see the ring_data module docstring)")


def default_path(name: str) -> str:
    return _first_existing(*_DEFAULT_ROOTS[name])


# --------------------------------------------------------------------------------------------------------------------------
# SPECS registration (collate pads [I, I] only for keys listed here; spawned workers re-run this at import)
# --------------------------------------------------------------------------------------------------------------------------
def register_ring_specs() -> None:
    from rfproteina.datapipes.schema import proteina as schema

    schema.SPECS.setdefault("ring_state_gt", (torch.int64, RING_UNOBS, "I I"))
    schema.SPECS.setdefault("ring_obs", (torch.bool, False, "I I"))
    # per-row validation override read by ring_flow (0 = training tube, 1 = all-MASK, 2 = full reveal); scalar -> [b] after collate
    schema.SPECS.setdefault("ring_val_mode", (torch.int64, 0, ""))   # identical to pair_stream.register_specs (C2)


register_ring_specs()


def _plain_dict(x) -> dict:
    """mapping | DictConfig | None -> plain dict with RESOLVED interpolations; nested non-config objects (instantiated design tasks) kept."""
    if x is None:
        return {}
    if hasattr(x, "_content") or type(x).__module__.startswith("omegaconf"):
        from omegaconf import OmegaConf

        return dict(OmegaConf.to_container(x, resolve=True))
    return dict(x)


def cluster_group(cluster) -> np.ndarray:
    """sequence-cluster id of manifest ``cluster`` strings: the part before the first ``+`` (``abc+HEM`` and ``abc+CA`` are one group);
    empty / missing -> ''."""
    c = np.asarray(cluster, dtype=object)
    return np.array([("" if (x is None or x != x) else str(x).split("+", 1)[0]) for x in c], dtype=object)


def hash_uniform(keys, salt: int = 0) -> np.ndarray:
    """deterministic u in [0, 1) per key (sha1 of ``salt|key``); equal keys give equal u (cluster-level, not row-level, decisions)."""
    import hashlib

    uniq, inv = np.unique(np.asarray(keys, dtype=object).astype(str), return_inverse=True)
    u = np.array([int.from_bytes(hashlib.sha1(f"{salt}|{k}".encode()).digest()[:8], "big") / 2.0**64 for k in uniq])
    return u[inv]


def quiet_atomworks() -> None:
    for name in ("atomworks", "atomworks.io", "biotite"):
        logging.getLogger(name).setLevel(logging.ERROR)


# --------------------------------------------------------------------------------------------------------------------------
# stores
# --------------------------------------------------------------------------------------------------------------------------
_ENVS: dict[tuple[str, int], Any] = {}
_TABLES: dict[str, dict] = {}                                        # node-local npz path -> filtered manifest table (per process)


def _ring_env(path: str):
    import lmdb

    key = (path, os.getpid())
    env = _ENVS.get(key)
    if env is None:
        env = _ENVS[key] = lmdb.open(path, readonly=True, lock=False, readahead=False, subdir=True, max_readers=512)
    return env


class RingStore:
    """RING LMDB reader: key -> record dict (13 record types + metadata). Reopens per process (spawn safe)."""

    def __init__(self, path: str | None = None):
        self.path = path or default_path("ring_lmdb")

    @staticmethod
    def pn_unit_key(pdb_id: str, asm: str, unit: str) -> str:
        return f"{{['pdb', 'pn_units']}}{{{pdb_id}}}{{{asm}}}{{['{unit}']}}"

    @staticmethod
    def interface_key(pdb_id: str, asm: str, u1: str, u2: str) -> str:
        return f"{{['pdb', 'interfaces']}}{{{pdb_id}}}{{{asm}}}{{['{u1}', '{u2}']}}"

    def get(self, key: str) -> dict | None:
        with _ring_env(self.path).begin() as txn:
            raw = txn.get(key.encode())
        return pickle.loads(raw) if raw is not None else None


class CifStore:
    """PDB mmCIF text by id. `mirror_dir` (RCSB divided layout, <dir>/<id[1:3]>/<id>.cif.gz) if it exists, otherwise the
    tar `pdb_mirror.tar` of the same files, read by byte offset (no extraction, no inodes)."""

    def __init__(self, tar_path: str | None = None, index_path: str | None = None, mirror_dir: str | None = None):
        self.mirror_dir = mirror_dir if (mirror_dir and os.path.isdir(mirror_dir)) else None
        self.tar_path = tar_path
        self.index_path = index_path
        self._ids = self._off = self._size = None
        if self.mirror_dir is None:
            self._load_index()

    def _load_index(self):
        import pyarrow.parquet as pq

        self.tar_path = self.tar_path or default_path("tar")
        t = pq.read_table(self.index_path or default_path("tar_index"), columns=["pdb_id", "offset_data", "size"])
        ids = np.asarray(t["pdb_id"].to_numpy(zero_copy_only=False)).astype("U8")
        o = np.argsort(ids)
        self._ids, self._off, self._size = ids[o], t["offset_data"].to_numpy()[o], t["size"].to_numpy()[o]

    def _pos(self, pdb_id: str) -> int:
        i = int(np.searchsorted(self._ids, pdb_id))
        if i >= len(self._ids) or self._ids[i] != pdb_id:
            raise KeyError(pdb_id)
        return i

    @staticmethod
    def _code(ids) -> np.ndarray:
        """4-character PDB ids -> int64 (byte-packed), vectorised; string searchsorted over ~10^6 ids is 100x slower."""
        b = np.ascontiguousarray(np.asarray(ids).astype("S4")).view(np.uint8).reshape(-1, 4).astype(np.int64)
        return (b[:, 0] << 24) | (b[:, 1] << 16) | (b[:, 2] << 8) | b[:, 3]

    def gz_size(self, pdb_ids: np.ndarray) -> np.ndarray:
        """compressed CIF size per id (-1 when unknown); used as a cheap proxy for parse cost."""
        if self.mirror_dir is not None:
            return np.array([os.path.getsize(self._mirror_path(p)) if os.path.exists(self._mirror_path(p)) else -1 for p in pdb_ids])
        codes = self._code(pdb_ids)
        if getattr(self, "_codes", None) is None:
            self._codes = self._code(self._ids)           # _ids sorted as strings; codes are monotone in that order
        pos = np.clip(np.searchsorted(self._codes, codes), 0, len(self._codes) - 1)
        ok = self._codes[pos] == codes
        return np.where(ok, self._size[pos], -1)

    def _mirror_path(self, pdb_id: str) -> str:
        return os.path.join(self.mirror_dir, pdb_id[1:3], f"{pdb_id}.cif.gz")

    def read_text(self, pdb_id: str) -> str:
        if self.mirror_dir is not None:
            with open(self._mirror_path(pdb_id), "rb") as f:
                raw = f.read()
        else:
            i = self._pos(pdb_id)
            with open(self.tar_path, "rb") as f:
                f.seek(int(self._off[i]))
                raw = f.read(int(self._size[i]))
        return gzip.decompress(raw).decode()


# --------------------------------------------------------------------------------------------------------------------------
# residue table of a pre-pipeline atom array
# --------------------------------------------------------------------------------------------------------------------------
@dataclass
class ResidueTable:
    n_res: int
    atom_res: np.ndarray        # [n_atoms] residue ordinal of every atom
    unit: np.ndarray            # [n_res] pn_unit_iid
    res_id: np.ndarray          # [n_res]
    res_name: np.ndarray        # [n_res]
    lookup: dict                # (unit, res_id) -> [ordinals]   (several only with insertion codes)


def build_residue_table(aa) -> ResidueTable:
    n = aa.array_length()
    chg = np.zeros(n, dtype=bool)
    chg[0] = True
    if n > 1:
        chg[1:] = ((aa.chain_iid[1:] != aa.chain_iid[:-1]) | (aa.res_id[1:] != aa.res_id[:-1])
                   | (aa.ins_code[1:] != aa.ins_code[:-1]) | (aa.res_name[1:] != aa.res_name[:-1]))
    starts = np.flatnonzero(chg)
    atom_res = np.cumsum(chg) - 1
    unit = np.asarray(aa.pn_unit_iid)[starts].astype(str)
    res_id = np.asarray(aa.res_id)[starts].astype(np.int64)
    res_name = np.asarray(aa.res_name)[starts].astype(str)
    lookup: dict[tuple[str, int], list[int]] = {}
    for o, (u, r) in enumerate(zip(unit.tolist(), res_id.tolist())):
        lookup.setdefault((u, r), []).append(o)
    return ResidueTable(len(starts), atom_res.astype(np.int64), unit, res_id, res_name, lookup)


# --------------------------------------------------------------------------------------------------------------------------
# RING records -> labels in residue-ordinal space
# --------------------------------------------------------------------------------------------------------------------------
@dataclass
class RingLabels:
    n_res: int
    pair_i: np.ndarray                      # int64, i < j  (residue ordinals)
    pair_j: np.ndarray
    pair_state: np.ndarray                  # int64 0..21 after collapse
    groups: list                            # list[np.ndarray] certified residue ordinals (one per record x pn_unit block)
    group_pairs: list                       # list[(ga, gb)] group pairs whose cross product is observed (ga == gb: intra block)
    stats: dict = field(default_factory=dict)
    poison_i: np.ndarray | None = None      # pairs (i<j) of an entry whose join was rejected by the coordinate check: never NO_INTERACTION
    poison_j: np.ndarray | None = None
    side_log: list | None = None            # optional per-side diagnostics: (record key, ordinal or -code, accepted, distance to own residue)


def _intra_units(rec: dict) -> set:
    """pn_units whose INTRA edges the record demonstrably contains: metadata must name the chain AND carry a positive
    ``n_ring_edges_intra_k``. Missing metadata means NOT certified (the intra block stays UNOBS, never NO_INTERACTION)."""
    meta = rec.get("metadata", {}) or {}
    out = set()
    for k in (1, 2):
        c = meta.get(f"chain_{k}")
        e = meta.get(f"n_ring_edges_intra_{k}")
        if c is not None and e is not None and int(e) > 0:
            out.add(str(c))
    return out


def build_ring_labels(aa, tab: ResidueTable, records: Sequence[dict], *, tol_atom: float = TOL_ATOM, tol_pi: float = TOL_PI,
                      ambiguity_margin: float = AMBIGUITY_MARGIN, keep_side_log: bool = False) -> RingLabels:
    """Join RING records to the residues of ``aa`` (identity join + coordinate check) and build pair labels + coverage.

    Statistics (all counted, nothing silent): n_entries, n_sides, n_out_of_scope (chain not among the kept pn_units, e.g. a
    ligand partner), n_unresolved (kept chain but resnum absent: unresolved residue), n_resname_mismatch, n_coord_reject,
    n_joined, n_self_pairs, n_pairs_labelled, n_pairs_observed_rowwise.
    """
    from scipy.spatial import cKDTree

    st = dict(n_records=0, n_entries=0, n_sides=0, n_out_of_scope=0, n_unresolved=0, n_resname_mismatch=0, n_ambiguous=0, n_ambiguous_unresolved=0,
              n_coord_reject=0, n_joined=0, n_self_pairs=0, n_pairs_labelled=0, n_entries_dropped_partial=0, n_pairs_poisoned=0)
    el = np.asarray(aa.element).astype(str)
    heavy = ~np.isin(el, ("H", "D"))
    xyz_all = np.asarray(aa.coord, dtype=np.float64)
    ok_xyz = np.isfinite(xyz_all).all(1) & heavy
    heavy_idx = np.flatnonzero(ok_xyz)
    tree = cKDTree(xyz_all[heavy_idx]) if heavy_idx.size else None
    atom_res_h = tab.atom_res[heavy_idx]
    units_here = set(tab.unit.tolist())

    all_i, all_j, all_state, all_rank = [], [], [], []
    poison_i, poison_j = [], []
    groups: list[np.ndarray] = []
    group_pairs: list[tuple[int, int]] = []
    side_log: list = []
    res_cache: dict[tuple, int] = {}
    amb_cands: dict[tuple, list] = {}

    for rec in records:
        if rec is None:
            continue
        st["n_records"] += 1
        sides_ord, sides_xyz, sides_pi, sides_ck, sides_rk, sides_atom, entry_type = [], [], [], [], [], [], []
        n_entry_rec = 0
        for rk, tname in RING_RECORD_TO_TYPE.items():
            full = (rec.get(rk) or {}).get("full") or []
            for e in full:
                n_entry_rec += 1
                entry_type.append(RING_INTERACTION_TYPES.index(tname))
                for k in (1, 2):
                    ch, rn = str(e[f"chain_{k}"]), int(e[f"resnum_{k}"])
                    nm = str(e.get(f"resname_{k}", "")).strip().upper()
                    nm = RESNAME_ALIASES.get(nm, nm)
                    ck = (ch, rn, nm)
                    o = res_cache.get(ck)
                    if o is None:
                        if ch not in units_here:
                            o = -1                                   # chain not kept (ligand partner, other chain)
                        else:
                            cands = tab.lookup.get((ch, rn))
                            if not cands:
                                o = -2                               # kept chain, residue absent (unresolved / removed)
                            else:
                                good = [c for c in cands if tab.res_name[c] == nm]
                                o = good[0] if len(good) == 1 else (-3 if not good else -4)   # -3 name mismatch, -4 ambiguous
                                if o == -4:
                                    amb_cands[ck] = good
                        res_cache[ck] = o
                    sides_ord.append(o)
                    sides_xyz.append(e[f"coord_{k}"])
                    sides_pi.append(rk in CENTROID_RECORDS)
                    sides_ck.append(ck)
                    sides_rk.append(rk)
                    sides_atom.append(str(e.get(f"atom_{k}", "")).strip())
        st["n_entries"] += n_entry_rec
        if n_entry_rec == 0:
            continue
        so = np.asarray(sides_ord, dtype=np.int64)
        sx = np.asarray(sides_xyz, dtype=np.float64).reshape(-1, 3)
        spi = np.asarray(sides_pi, dtype=bool)
        st["n_sides"] += so.size
        st["n_out_of_scope"] += int((so == -1).sum())
        st["n_unresolved"] += int((so == -2).sum())
        st["n_resname_mismatch"] += int((so == -3).sum())
        # ambiguous (same unit + resnum + resname, e.g. insertion codes): resolved ONLY when one candidate is clearly the nearest.
        # distance of a candidate = distance from the entry coordinate to its atom NAMED in the entry (atom-valued records) or to
        # its nearest atom (centroid-valued records / atom absent); the best must beat the runner-up by `ambiguity_margin`,
        # otherwise the side FAILS (n_ambiguous_unresolved) and every pair (other side, candidate) becomes UNKNOWN (poisoned).
        amb = np.flatnonzero(so == -4)
        st["n_ambiguous"] += amb.size
        amb_poison: dict[int, list[int]] = {}
        for s_ in amb:
            cands = amb_cands[sides_ck[s_]]
            dc = []
            for c in cands:
                a = np.flatnonzero((tab.atom_res == c) & ok_xyz)
                if a.size == 0:
                    dc.append(np.inf)
                    continue
                d_all = np.linalg.norm(xyz_all[a] - sx[s_], axis=1)
                named = a[np.asarray(aa.atom_name)[a].astype(str) == sides_atom[s_]] if not spi[s_] else a[:0]
                if named.size:
                    dc.append(float(np.linalg.norm(xyz_all[named] - sx[s_], axis=1).min()))
                else:
                    dc.append(float(d_all.min()))
            order_ = np.argsort(dc)
            if len(dc) >= 2 and np.isfinite(dc[order_[0]]) and (dc[order_[1]] - dc[order_[0]]) >= ambiguity_margin:
                so[s_] = cands[int(order_[0])]
            else:
                so[s_] = -5                                           # ambiguous and unresolved
                amb_poison[int(s_)] = [int(c) for c in cands]
                st["n_ambiguous_unresolved"] += 1
        # coordinate check for identity-joined sides
        cand_sides = np.flatnonzero(so >= 0)
        accept = np.zeros(so.size, dtype=bool)
        dist = np.full(so.size, np.inf)
        if cand_sides.size and tree is not None:
            d_nn, j_nn = tree.query(sx[cand_sides], k=1)
            same = atom_res_h[j_nn] == so[cand_sides]
            dist[cand_sides[same]] = d_nn[same]
            rest = cand_sides[~same]
            for s in rest:                                           # nearest atom belongs elsewhere: distance to OWN residue
                a = np.flatnonzero((tab.atom_res == so[s]) & ok_xyz)
                dist[s] = np.linalg.norm(xyz_all[a] - sx[s], axis=1).min() if a.size else np.inf
            tol = np.where(spi[cand_sides], tol_pi, tol_atom)
            accept[cand_sides] = dist[cand_sides] <= tol
        st["n_coord_reject"] += int((so >= 0).sum() - accept.sum())
        st["n_joined"] += int(accept.sum())
        if keep_side_log:
            for s in range(so.size):
                side_log.append((sides_rk[s], int(so[s]), bool(accept[s]), float(dist[s])))

        sides_ok = accept.reshape(-1, 2)
        so2 = so.reshape(-1, 2)
        both = sides_ok.all(1)
        # entry with both residues identified by identity but a rejected coordinate: the pair is UNKNOWN (poisoned), not "no interaction"
        pz = (so2 >= 0).all(1) & ~both & (so2[:, 0] != so2[:, 1])
        poison_i.append(np.minimum(so2[pz, 0], so2[pz, 1]))
        poison_j.append(np.maximum(so2[pz, 0], so2[pz, 1]))
        for s_, cands in amb_poison.items():                      # (other side, each candidate) pairs are unknown
            e_, k_ = divmod(s_, 2)
            other = so2[e_, 1 - k_]
            others = [int(other)] if other >= 0 else amb_poison.get(2 * e_ + (1 - k_), [])
            for o_ in others:
                for c_ in cands:
                    if o_ != c_:
                        poison_i.append(np.array([min(o_, c_)], dtype=np.int64))
                        poison_j.append(np.array([max(o_, c_)], dtype=np.int64))
        st["n_entries_dropped_partial"] += int((~both).sum())
        et = np.asarray(entry_type, dtype=np.int64)
        i_ord, j_ord = so2[both, 0], so2[both, 1]
        t_e = et[both]
        self_pair = i_ord == j_ord
        st["n_self_pairs"] += int(self_pair.sum())
        i_ord, j_ord, t_e = i_ord[~self_pair], j_ord[~self_pair], t_e[~self_pair]
        intra = tab.unit[i_ord] == tab.unit[j_ord]
        chan = t_e + N_TYPES * (~intra).astype(np.int64)
        lo, hi = np.minimum(i_ord, j_ord), np.maximum(i_ord, j_ord)
        all_i.append(lo)
        all_j.append(hi)
        all_state.append(chan)
        all_rank.append(TYPE_RANK[t_e])

        # coverage for THIS record: certified = joined residues, per pn_unit block with >= MIN_TOUCHED residues
        touched = np.unique(so[accept])
        if touched.size:
            intra_ok = _intra_units(rec)
            blk_units = tab.unit[touched]
            rec_groups = {}
            for u in np.unique(blk_units):
                members = touched[blk_units == u]
                if members.size >= MIN_TOUCHED_PER_BLOCK:
                    rec_groups[str(u)] = len(groups)
                    groups.append(members)
            us = list(rec_groups)
            for a in range(len(us)):
                for b in range(a, len(us)):
                    if a == b and us[a] not in intra_ok:
                        continue
                    group_pairs.append((rec_groups[us[a]], rec_groups[us[b]]))

    if all_i:
        pi, pj = np.concatenate(all_i), np.concatenate(all_j)
        ps, pr = np.concatenate(all_state), np.concatenate(all_rank)
    else:
        pi = pj = ps = pr = np.zeros(0, dtype=np.int64)
    if pi.size:                                                      # collapse: strongest type per pair
        key = pi * np.int64(tab.n_res) + pj
        order = np.lexsort((pr, key))
        key_s = key[order]
        first = np.ones(key_s.size, dtype=bool)
        first[1:] = key_s[1:] != key_s[:-1]
        sel = order[first]
        pi, pj, ps = pi[sel], pj[sel], ps[sel]
    st["n_pairs_labelled"] = int(pi.size)
    qi = np.concatenate(poison_i) if poison_i else np.zeros(0, dtype=np.int64)
    qj = np.concatenate(poison_j) if poison_j else np.zeros(0, dtype=np.int64)
    st["n_pairs_poisoned"] = int(qi.size)
    return RingLabels(tab.n_res, pi, pj, ps, groups, group_pairs, st, qi, qj, side_log if keep_side_log else None)


def check_join_not_degenerate(labels: RingLabels, key: str = "") -> None:
    """RowSkip('degenerate_ring_join') when a NON-EMPTY RING record produced no usable join: not one entry had BOTH sides joined
    (all sides out of scope / unresolved / name-mismatched / ambiguous / coordinate-rejected), or entries joined and >= 8 residues
    are certified yet not one pair is labelled (the signature of the old zero-map bug). A record without any entry is legitimate
    (every pair stays UNOBS)."""
    st = labels.stats
    n_full = st["n_entries"] - st["n_entries_dropped_partial"]          # entries with both sides joined
    if st["n_entries"] > 0 and n_full == 0:
        raise RowSkip("degenerate_ring_join", f"{key}: {st['n_entries']} entries, none with both sides joined {st}")
    if n_full > 0 and st["n_pairs_labelled"] == 0 and sum(len(g) for g in labels.groups) >= 8:
        raise RowSkip("degenerate_ring_join", f"{key}: entries joined and residues certified but 0 labelled pairs {st}")


def labels_to_tokens(labels: RingLabels, uid_tok: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slice residue-ordinal labels into token space. ``uid_tok[t]`` = residue ordinal of token t, ``eligible[t]`` = protein
    residue token (not atomised, not an unindexed copy, unique). Returns (state [I,I] int64, obs [I,I] bool)."""
    I = int(uid_tok.shape[0])
    tok_of = np.full(labels.n_res, -1, dtype=np.int64)
    idx = np.flatnonzero(eligible)
    tok_of[uid_tok[idx]] = idx
    obs = np.zeros((I, I), dtype=bool)
    for ga, gb in labels.group_pairs:
        ta, tb = tok_of[labels.groups[ga]], tok_of[labels.groups[gb]]
        ta, tb = ta[ta >= 0], tb[tb >= 0]
        if ta.size and tb.size:
            obs[np.ix_(ta, tb)] = True
            obs[np.ix_(tb, ta)] = True
    ti, tj = tok_of[labels.pair_i], tok_of[labels.pair_j]
    keep = (ti >= 0) & (tj >= 0)
    ti, tj, ts = ti[keep], tj[keep], labels.pair_state[keep]
    obs[ti, tj] = True                                               # a labelled pair is observed whatever the coverage says
    obs[tj, ti] = True
    obs[np.arange(I), np.arange(I)] = False
    state = np.full((I, I), RING_UNOBS, dtype=np.int64)
    state[obs] = NO_INTERACTION
    state[ti, tj] = ts
    state[tj, ti] = ts
    if labels.poison_i is not None and labels.poison_i.size:        # an entry of this pair was rejected by the coordinate check:
        pi_, pj_ = tok_of[labels.poison_i], tok_of[labels.poison_j]  # its true label is unknown (maybe a stronger type) -> UNOBS
        k = (pi_ >= 0) & (pj_ >= 0)
        for a_, b_ in ((pi_[k], pj_[k]), (pj_[k], pi_[k])):
            obs[a_, b_] = False
            state[a_, b_] = RING_UNOBS
    return state, obs


# --------------------------------------------------------------------------------------------------------------------------
# network-input helper (the no-coverage-leak construction)
# --------------------------------------------------------------------------------------------------------------------------
def make_ring_state_input(ring_state_gt: torch.Tensor, ring_obs: torch.Tensor, padding_mask: torch.Tensor, q,
                          generator: torch.Generator | None = None) -> torch.Tensor:
    """[b,n,n] network input for reveal probability ``q`` (scalar or [b]; 0 = nothing revealed, 1 = everything revealed).

    COVERAGE-INDEPENDENT rule (identical to ``triadj_ext.ring_flow.TriadJFlowMatcher.corrupt_batch``; the test file asserts bit
    equality for the same RNG state): reveal locations are drawn over ALL valid off-diagonal pairs with a symmetric uniform
    ``u < q`` (same u for (i,j) and (j,i)), independent of ``ring_obs``. A revealed pair shows its label where it is observed and
    NO_INTERACTION (22) where it is not; every hidden pair is MASK (23); padding and the diagonal are PAD (24). So neither the
    reveal pattern nor the MASK pattern depends on which pairs RING covered. ``ring_obs`` enters only the LOSS: the loss uses the
    pairs that are observed AND hidden (MASK) in this input. With ``generator=None`` the global torch RNG is used (as ring_flow).
    """
    b, n, _ = ring_state_gt.shape
    q = torch.as_tensor(q, dtype=torch.float32, device=ring_state_gt.device).reshape(-1, 1, 1)
    u = torch.rand((b, n, n), generator=generator, device=ring_state_gt.device)
    u = torch.triu(u, 1)
    u = u + u.transpose(-1, -2)                                      # symmetric, diagonal 0 (the same construction as ring_flow)
    pair = padding_mask[:, :, None] & padding_mask[:, None, :]
    valid = pair & ~torch.eye(n, dtype=torch.bool, device=pair.device)[None]
    gt = ring_state_gt.long()
    # a row WITHOUT any observed pair (replay rows: collate fills gt = 24 / obs = False) is all-MASK whatever q is: showing random
    # NO_INTERACTION pairs there would teach "22 is uninformative" on unlabelled data (ring_flow.corrupt_batch has the same row rule)
    has_ring = (ring_obs & valid).flatten(1).any(1)[:, None, None]
    shown = torch.where(ring_obs, gt, torch.full_like(gt, NO_INTERACTION))
    st = torch.where((u < q) & has_ring, shown, torch.full_like(gt, RING_MASK))
    return torch.where(valid, st, torch.full_like(st, RING_PAD))


def make_ring_state_input_from_batch(batch, q, generator: torch.Generator | None = None) -> torch.Tensor:
    """``make_ring_state_input`` on a collated batch (needs ``ring_state_gt``, ``ring_obs``, ``padding_mask``)."""
    return make_ring_state_input(batch["ring_state_gt"], batch["ring_obs"], batch["padding_mask"], q, generator)


# --------------------------------------------------------------------------------------------------------------------------
# the dataset
# --------------------------------------------------------------------------------------------------------------------------
_OFF = {   # all-off conditioning (same kwargs as the spike's s2_batch / tests/pipeline/test_pipeline_invariants.py)
    "rasa_kwargs": {"p_add_rasa_condition": 0.0},
    "hbond_kwargs": {"p_add_hbond_condition": 0.0},
    "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.0},
    "cath_kwargs": {"p_add_cath_c_condition": 0.0, "p_add_cath_ca_condition": 0.0, "p_add_cath_cat_condition": 0.0},
}


COUNT_KEYS = ("requested", "ok", "substituted", "failed", "no_ring_record", "unit_not_protein", "unit_not_in_assembly",
              "assembly_missing", "degenerate_ring_join", "exception")


class RowSkip(Exception):
    """A manifest row that cannot make a valid sample; counted by reason, replaced by a deterministic substitute row."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class RingDataset(torch.utils.data.Dataset):
    """PDB manifest rows -> RFD4 pipeline sample dict (``feats`` extended by ``ring_state_gt`` / ``ring_obs``).

    Cost model (measured, one CPU core, warm process): the sample time is >95% atomworks ``parse`` of the whole assembly and grows with
the compressed CIF size: <200 kB gz 0.35-0.9 s, 200-400 kB 1.0 s, 400-800 kB 2.0 s, 0.8-1.5 MB 6 s, 1.5-3 MB 18 s (the RFD4 pipeline
itself is 0.06-0.3 s). ``max_cif_bytes`` (default 500 kB gz = ~70% of the PDB monomer rows) bounds that tail; raise it only with
more workers. ``__getitem__`` returns the same per-sample dict RFD4's datasets return (keys ``example_id``, ``feats``, ``atom_array``,
    ``extra_info``, ``task`` ...) plus ``ring_stats``; ``collate`` (staticmethod / ``make_loader``) collates with RFD4's own
    ``collate_batch`` (pad to ``fixed_dimensions``, ``num_replicates``). Set ``return_atom_array=True`` to keep the pipeline's
    output atom array (verification; costs pickling).
    """

    def __init__(self, manifest: str | None = None, *, rows: str = "monomer", crop_size: int | None = 256,
                 crop_spatial_probability: float = 1.0, crop_contiguous_probability: float = 0.0,
                 min_residues: int = 30, max_residues: int = 1500, max_cif_bytes: int = 500_000, include_ncaa: bool = False,
                 ring_lmdb: str | None = None, tar_path: str | None = None, tar_index: str | None = None,
                 mirror_dir: str | None = None, seed: int = 0, design_tasks: dict | None = None,
                 return_atom_array: bool = False, return_debug: bool = False, max_retries: int = 8, require_ring: bool = True,
                 substitute_on_failure: bool = True, pipeline_kwargs: dict | None = None, row_filter=None, max_rows: int | None = None,
                 ring_labels: bool = True, name: str | None = None, transform: dict | None = None, split: str | None = None,
                 n_val: int | None = 512, val_fraction: float | None = None, split_seed: int = 0, val_mode: int | None = None):
        """New in S2 (all optional, defaults reproduce the previous behaviour):
        ring_labels  False = REPLAY: no RING join, no ring keys (see module docstring); True = labelled rows.
        name         written to ``sample['dataset_name']`` (dataset/task statistics callbacks); default ring / replay.
        transform    mapping merged into the pipeline kwargs (Hydra-style ``dataset.transform``); ``pipeline_kwargs`` wins on conflicts;
                     a ``design_tasks`` entry is used when the ``design_tasks`` argument is None.
        design_tasks a plain ``{name: {"transform": Task, "frequency": f}}`` dict or a Hydra-instantiated preset (DictConfig node).
        split        None | 'train' | 'val' (module docstring); ``n_val`` caps the val rows (None = all), ``val_fraction`` = hash split
                     of ONE manifest, ``split_seed`` salts the hash.
        val_mode     None | 0 | 1 | 2: constant per-row ``ring_val_mode`` (0 tube, 1 all-MASK, 2 full reveal) written to every sample;
                     read by ``ring_flow`` in validation only (None writes nothing; the collate fills 0 = tube).
        """
        assert rows in ("monomer", "interface", "all")
        assert split in (None, "train", "val"), split
        assert val_mode in (None, 0, 1, 2), val_mode
        self.ring_labels = bool(ring_labels)
        self.name = name or ("ring" if self.ring_labels else "replay")
        self.split, self.val_mode = split, (None if val_mode is None else int(val_mode))
        self.cfg = dict(manifest=manifest, rows=rows, crop_size=crop_size, crop_spatial_probability=crop_spatial_probability,
                        crop_contiguous_probability=crop_contiguous_probability, min_residues=min_residues,
                        max_residues=max_residues, max_cif_bytes=max_cif_bytes, include_ncaa=include_ncaa, ring_lmdb=ring_lmdb, tar_path=tar_path,
                        tar_index=tar_index, mirror_dir=mirror_dir, split=split, n_val=n_val, val_fraction=val_fraction,
                        split_seed=int(split_seed))
        self.seed = int(seed)
        self.substitute_on_failure = bool(substitute_on_failure)
        # state shared with (persistent, spawned) DataLoader workers: slot 0 = epoch, then the counters of COUNT_KEYS. An
        # mp synchronized array survives pickling INTO a spawned worker (inheritance) and is live afterwards.
        import multiprocessing as _mp

        self._shared = _mp.get_context("spawn").Array("q", 1 + len(COUNT_KEYS))
        tkw = _plain_dict(transform)
        for k in ("_target_", "_partial_", "_args_", "_convert_", "_recursive_"):
            tkw.pop(k, None)
        if design_tasks is None:
            design_tasks = tkw.pop("design_tasks", None)
        else:
            tkw.pop("design_tasks", None)
        self.design_tasks = design_tasks
        self.return_atom_array = return_atom_array
        self.return_debug = return_debug
        self.max_retries = max_retries
        self.require_ring = require_ring
        self.pipeline_kwargs = {**tkw, **_plain_dict(pipeline_kwargs)}
        self.drop_counts: dict[str, int] = {}
        self._pipe = None
        self._ring = None
        self._cif = None
        self._cache_path = None
        self._tab = None
        self._data_df = None
        self._select_rows(row_filter, max_rows)

    # ---- manifest ---------------------------------------------------------------------------------------------------
    def _select_rows(self, row_filter, max_rows):
        import hashlib

        c = self.cfg
        cache = None
        if row_filter is None and max_rows is None and os.environ.get("RING_ROWS_CACHE", "1") != "0":
            import tempfile

            h = hashlib.sha1(("v4" + repr(sorted(c.items()))).encode()).hexdigest()[:12]
            cache = os.path.join(os.environ.get("RING_ROWS_CACHE_DIR", tempfile.gettempdir()), f"ring_rows_{h}.npz")
            if os.path.exists(cache):                        # node-local cache of the filtered manifest table (1 file)
                self._cache_path = cache
                self._tab = None
                self._n = len(self._table()["pdb"])
                return
        self._select_rows_uncached(row_filter, max_rows)
        if cache is not None:
            tmp = f"{cache}.{os.getpid()}.tmp.npz"                   # unique per process: several ranks of one node build the same table
            np.savez(tmp, **self._tab)
            os.replace(tmp, cache)
            self._cache_path = cache
            _TABLES[cache] = self._tab

    def _table(self) -> dict:
        """filtered manifest table (numpy); reloaded from the node-local cache in DataLoader workers instead of being pickled"""
        if self._tab is None:
            t = _TABLES.get(self._cache_path)                        # one copy per process, shared by every dataset on the same table
            if t is None:
                z = np.load(self._cache_path)
                t = _TABLES[self._cache_path] = {k: z[k] for k in z.files}
            self._tab = t
        return self._tab

    def _select_rows_uncached(self, row_filter, max_rows):
        import pyarrow.parquet as pq

        c = self.cfg
        mdir = default_path("manifest_dir")
        default_manifest = VAL_MANIFEST if c["split"] == "val" and c["val_fraction"] is None else TRAIN_MANIFEST
        path = c["manifest"] or os.path.join(mdir, default_manifest)
        cols = ["pdb_id", "assembly_id", "pn_unit_iid", "pn_unit_1_iid", "pn_unit_2_iid", "n_residues", "is_monomer", "cluster", "source"]
        have = set(pq.ParquetFile(path).schema_arrow.names)
        df = pq.read_table(path, columns=[x for x in cols if x in have]).to_pandas()
        is_pdb = df["pdb_id"].astype(str).str.len().eq(4)
        if "source" in df:
            is_pdb &= df["source"].isna() | df["source"].eq("pdb")
        mono = is_pdb & df["pn_unit_iid"].notna() & df["is_monomer"].eq(True) if "is_monomer" in df else is_pdb & df["pn_unit_iid"].notna()
        iface = is_pdb & df["pn_unit_1_iid"].notna() & df["pn_unit_2_iid"].notna()
        sel = {"monomer": mono, "interface": iface, "all": mono | iface}[c["rows"]]
        nres = df["n_residues"].astype(float)
        sel &= nres >= c["min_residues"]
        if c["rows"] != "interface":
            sel &= nres <= c["max_residues"]
        df = df[sel].reset_index(drop=True)
        cif = CifStore(c["tar_path"], c["tar_index"], c["mirror_dir"])
        gz = cif.gz_size(df["pdb_id"].to_numpy(dtype="U8"))
        df = df[(gz > 0) & (gz <= c["max_cif_bytes"])].reset_index(drop=True)
        if c["split"] is not None:
            df = self._apply_split(df, mdir)
        if row_filter is not None:
            df = df[row_filter(df)].reset_index(drop=True)
        if max_rows is not None:
            df = df.iloc[:max_rows]
        has_if = c["rows"] != "monomer"                              # interface columns only when needed (memory per worker)
        if has_if and "pn_unit_1_iid" not in df:
            has_if = False
        self._tab = {
            "pdb": df["pdb_id"].to_numpy(dtype="U4"), "asm": df["assembly_id"].astype(str).to_numpy(dtype="U3"),
            "u0": df["pn_unit_iid"].fillna("").to_numpy(dtype="U10") if "pn_unit_iid" in df else np.zeros(len(df), "U1"),
            "u1": df["pn_unit_1_iid"].fillna("").to_numpy(dtype="U10") if has_if else np.zeros(len(df), "U1"),
            "u2": df["pn_unit_2_iid"].fillna("").to_numpy(dtype="U10") if has_if else np.zeros(len(df), "U1"),
            "nres": df["n_residues"].to_numpy(dtype=np.int32),
            "cluster": df["cluster"].fillna("").to_numpy(dtype="U32"),
        }
        self._n = len(df)

    def _apply_split(self, df, mdir: str):
        """cluster-disjoint train / val subsets (module docstring); ``df`` is already filtered by rows / size."""
        import pyarrow.parquet as pq

        c = self.cfg
        grp = cluster_group(df["cluster"].to_numpy(dtype=object)) if "cluster" in df else np.full(len(df), "", dtype=object)
        keyless = grp == ""                                          # no cluster id: such a row is its own cluster
        own = np.array([f"{a}_{b}" for a, b in zip(df["pdb_id"].astype(str), df["assembly_id"].astype(str))], dtype=object)
        grp = np.where(keyless, own, grp)
        if c["val_fraction"] is not None:                            # ONE manifest, split by hash(cluster id)
            u = hash_uniform(grp, c["split_seed"])
            is_val = u < float(c["val_fraction"])
            keep = is_val if c["split"] == "val" else ~is_val
            df, grp = df[keep].reset_index(drop=True), grp[keep]
        elif c["split"] == "train":                                  # train manifest minus every sequence cluster of the val manifest
            vp = os.path.join(mdir, VAL_MANIFEST)
            vgrp = set(cluster_group(pq.read_table(vp, columns=["cluster"]).to_pandas()["cluster"].to_numpy(dtype=object)).tolist())
            vgrp.discard("")
            df = df[~np.isin(grp, list(vgrp))].reset_index(drop=True)
            return df
        if c["split"] == "val":
            cols = [x for x in ("pdb_id", "assembly_id", "pn_unit_iid", "pn_unit_1_iid", "pn_unit_2_iid") if x in df]
            dup = df.duplicated(subset=cols).to_numpy()              # one row per (entry, assembly, unit(s)) like P0's val set
            df, grp = df[~dup].reset_index(drop=True), grp[~dup]
            if c["n_val"] is not None and len(df) > int(c["n_val"]):
                order = np.lexsort((np.arange(len(df)), hash_uniform(grp, c["split_seed"])))   # clusters in hash order, rows stable
                df = df.iloc[np.sort(order[: int(c["n_val"])])].reset_index(drop=True)
        return df

    def __len__(self) -> int:
        return self._n

    @property
    def data(self):
        """pandas DataFrame aligned with the dataset indices (one row per sample): ``cluster`` (manifest cluster string; a row without
        one is its own cluster), ``nres``, ``pdb_id``, ``kind``. Needed by ``atomworks.ml.samplers.calculate_weights_by_inverse_cluster_size``
        (``weights.dataset_df = dataset.data``, rfproteina/utils/datasets.py:121)."""
        if getattr(self, "_data_df", None) is None:
            import pandas as pd

            t = self._table()
            pdb, asm = pd.Series(t["pdb"]).astype(str), pd.Series(t["asm"]).astype(str)
            units = pd.Series(t["u0"]).astype(str) + "+" + pd.Series(t["u1"]).astype(str) + "+" + pd.Series(t["u2"]).astype(str)
            cl = pd.Series(t["cluster"]).astype(str)
            cl = cl.where(cl != "", "single_" + pdb + "_" + asm + "_" + units)
            self._data_df = pd.DataFrame({"pdb_id": pdb, "cluster": cl, "nres": np.asarray(t["nres"]),
                                          "kind": np.where(pd.Series(t["u1"]).astype(str) != "", "interface", "monomer")})
        return self._data_df

    # ---- example ids (atomworks ExampleIDProtocol: ConcatDatasetWithID / FallbackDatasetWrapper call idx_to_id on errors) ----------
    def idx_to_id(self, idx: int) -> str:
        r = self.row(int(idx))
        return f"{r['pdb_id']}_{r['assembly_id']}_{'+'.join(r['units'])}"

    def id_to_idx(self, example_id: str) -> int:
        ids = getattr(self, "_id_map", None)
        if ids is None:
            t = self._table()
            u = [(a, b, c, d) for a, b, c, d in zip(t["pdb"].tolist(), t["asm"].tolist(), t["u0"].tolist(), zip(t["u1"].tolist(), t["u2"].tolist()))]
            ids = {}
            for i, (pdb, asm, u0, (u1, u2)) in enumerate(u):
                ids.setdefault(f"{pdb}_{asm}_{u1 + '+' + u2 if u1 else u0}", i)
            self._id_map = ids
        return ids[example_id]

    def __contains__(self, example_id) -> bool:
        try:
            self.id_to_idx(example_id)
            return True
        except KeyError:
            return False

    @property
    def cluster_ids(self) -> np.ndarray:
        """per-row manifest ``cluster`` string (same string = same cluster) for cluster-balanced samplers / cluster-disjoint splits.
        ``ds._tab`` (dict of numpy arrays: pdb, asm, u0, u1, u2, nres, cluster) stays populated in the constructing process."""
        return self._table()["cluster"]

    @property
    def epoch(self) -> int:
        return int(self._shared[0])

    def set_epoch(self, epoch: int) -> None:
        """Takes effect for every sample built AFTER the call, in every worker process (shared counter), including persistent
        workers. Samples already prefetched (prefetch_factor x workers batches) were built with the old epoch: call this BEFORE
        ``iter(loader)`` of the new epoch (iter() re-issues the prefetch), which is the supported usage."""
        with self._shared.get_lock():
            self._shared[0] = int(epoch)

    def _count(self, key: str, k: int = 1) -> None:
        with self._shared.get_lock():
            self._shared[1 + COUNT_KEYS.index(key)] += k

    def counts(self) -> dict[str, int]:
        """Aggregated over ALL processes (shared counters): requested, ok, substituted, failed (strict mode sentinels) and one
        counter per skip reason (COUNT_KEYS); anything else is counted under 'exception'. Per-process detail with exception
        text stays in ``drop_counts``."""
        with self._shared.get_lock():
            return {k: int(self._shared[1 + i]) for i, k in enumerate(COUNT_KEYS)}

    def row(self, i: int) -> dict:
        t = self._table()
        is_if = bool(t["u1"][i])
        units = [str(t["u1"][i]), str(t["u2"][i])] if is_if else [str(t["u0"][i])]
        return dict(pdb_id=str(t["pdb"][i]), assembly_id=str(t["asm"][i]), units=units, kind="interface" if is_if else "monomer",
                    n_residues=int(t["nres"][i]), cluster=str(t["cluster"][i]))

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_pipe"] = d["_ring"] = d["_cif"] = None
        d["_data_df"] = None                                         # rebuilt lazily from the table; workers never need it
        d["_id_map"] = None
        if self._cache_path is not None and os.path.exists(self._cache_path):
            d["_tab"] = None                                         # workers re-read the node-local npz instead of unpickling it
        return d

    # ---- lazily built per process ----------------------------------------------------------------------------------
    def _pipeline(self):
        if self._pipe is None:
            quiet_atomworks()
            from rfproteina.datapipes.pipeline import build_design_train_pipeline
            from rfproteina.transforms.design_task import UnconditionalTask

            tasks = self.design_tasks or {"unconditional": {"transform": UnconditionalTask(global_design_task_kwargs=_OFF),
                                                            "frequency": 1.0}}
            kw = dict(crop_size=self.cfg["crop_size"], crop_contiguous_probability=self.cfg["crop_contiguous_probability"],
                      crop_spatial_probability=self.cfg["crop_spatial_probability"], design_tasks=tasks, is_inference=False,
                      p_random_rotation=1.0, path_length_search_cutoff=64,
                      loss_weight_cfg={"non_standard": 2.5, "unindexed": 2.5, "sampled_tip_atom": 0.0})
            kw.update(self.pipeline_kwargs)
            self._pipe = build_design_train_pipeline(**kw)
        return self._pipe

    def _stores(self):
        if self._cif is None:
            self._ring = RingStore(self.cfg["ring_lmdb"]) if getattr(self, "ring_labels", True) else None   # replay never opens lmdb
            self._cif = CifStore(self.cfg["tar_path"], self.cfg["tar_index"], self.cfg["mirror_dir"])
        return self._ring, self._cif

    # ---- one sample --------------------------------------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        """TRAINING mode (``substitute_on_failure=True``): a row that cannot be built is replaced by a deterministic other row
        (counted). STRICT mode (False, use for validation): the requested row is never replaced; on failure a sentinel
        ``{'failed': True, 'index', 'example_id', 'reason', 'drop_counts_snapshot'}`` is returned, which ``collate`` drops and
        ``split_failed`` reports. Every returned dict carries ``drop_counts_snapshot`` (this process's cumulative counts); the
        run-wide counts are ``counts()`` (shared across workers)."""
        n = len(self)
        last = None
        self._count("requested")
        attempts = (self.max_retries + 1) if self.substitute_on_failure else 1
        for attempt in range(attempts):
            j = int(idx) % n if attempt == 0 else (int(idx) * 7919 + attempt * 104729 + 13) % n
            try:
                sample = self.build_sample(j)
                self._count("ok")
                if attempt > 0:
                    self._count("substituted")
                sample["drop_counts_snapshot"] = self._snapshot()
                return sample
            except RowSkip as e:
                self.drop_counts[e.reason] = self.drop_counts.get(e.reason, 0) + 1
                self._count(e.reason if e.reason in COUNT_KEYS else "exception")
                last = e
            except Exception as e:  # noqa: BLE001  (parse/pipeline failures of single structures are counted, not fatal)
                key = "exc:" + type(e).__name__ + ":" + re.sub(r"\s+", " ", str(e))[:80]
                self.drop_counts[key] = self.drop_counts.get(key, 0) + 1
                self._count("exception")
                last = e
        if not self.substitute_on_failure:
            self._count("failed")
            r = self.row(int(idx) % n)
            return {"failed": True, "index": int(idx), "example_id": f"{r['pdb_id']}_{r['assembly_id']}_{'+'.join(r['units'])}",
                    "reason": (last.reason if isinstance(last, RowSkip) else repr(last)[:200]), "drop_counts_snapshot": self._snapshot()}
        raise RuntimeError(f"no valid sample after {self.max_retries + 1} attempts starting at {idx}; last: {last!r}; drops={self.drop_counts}")

    def _snapshot(self) -> dict:
        return {"pid": os.getpid(), "counts": dict(self.drop_counts)}

    def build_sample(self, i: int, seed: int | None = None) -> dict:
        from atomworks.io import parse
        from atomworks.io.utils.annotator import ensure_annotations
        from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
        from atomworks.ml.utils.token import get_token_starts
        from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
        from rfproteina.utils.parse_presets import RCSB_WITH_METALS

        import time

        t0 = time.perf_counter()
        row = self.row(i)
        ring, cif = self._stores()
        pdb, asm, units = row["pdb_id"], row["assembly_id"], row["units"]
        seed = int(seed if seed is not None else (self.seed * 1_000_003 + self.epoch * 10_007 + i * 97 + 12345) % (2**31 - 1))

        # ---- structure (RFD4's own parse) -----------------------------------------------------------------------------
        cfg = RCSB_WITH_METALS.replace(file_type="cif", build_assembly=(asm,))
        text = cif.read_text(pdb)
        t1 = time.perf_counter()
        data = parse(io.StringIO(text), config=cfg)
        t2 = time.perf_counter()
        if asm not in data["assemblies"]:
            raise RowSkip("assembly_missing", f"{pdb}:{asm}")
        full = data["assemblies"][asm][0]
        present = set(np.unique(full.pn_unit_iid).tolist())
        if not all(u in present for u in units):
            raise RowSkip("unit_not_in_assembly", f"{pdb}:{asm}:{units}")
        aa = full[np.isin(full.pn_unit_iid, units)]
        from atomworks.enums import ChainType

        prot = np.isin(aa.chain_type, [int(x) for x in ChainType.get_proteins()])
        if not prot.all():
            raise RowSkip("unit_not_protein", f"{pdb}:{units}")

        # ---- RING records of the row (labelled mode only; replay rows read no RING record at all) ---------------------
        labelled = getattr(self, "ring_labels", True)
        key = f"{pdb}_{asm}_{'+'.join(units)}"
        tab = labels = None
        if labelled:
            key = ring.pn_unit_key(pdb, asm, units[0]) if len(units) == 1 else ring.interface_key(pdb, asm, units[0], units[1])
            rec = ring.get(key)
            if rec is None and len(units) == 2:
                rec = ring.get(ring.interface_key(pdb, asm, units[1], units[0]))
            if rec is None and self.require_ring:
                raise RowSkip("no_ring_record", key)
            tab = build_residue_table(aa)
            labels = build_ring_labels(aa, tab, [rec] if rec is not None else [])
            if rec is not None:
                check_join_not_degenerate(labels, key)
            aa.set_annotation("ring_uid", tab.atom_res.astype(np.int64))
        ensure_annotations(aa, "is_protein")
        t3 = time.perf_counter()

        data["atom_array"] = aa
        data["example_id"] = f"{pdb}_{asm}_{'+'.join(units)}"
        data["seed"] = seed
        data["query_pn_unit_iids"] = list(units)
        pipe = self._pipeline()
        with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
            out = pipe(data)
        t4 = time.perf_counter()

        # ---- tokens -> residue ordinals -> token-space labels --------------------------------------------------------
        oa, feats = out["atom_array"], out["feats"]
        I = int(feats["residue_type"].shape[0])
        sample = {k: v for k, v in out.items() if k != "atom_array"}
        sample["dataset_name"] = getattr(self, "name", None) or "ring"
        sample["feats"] = feats
        if getattr(self, "val_mode", None) is not None:
            feats["ring_val_mode"] = torch.tensor(int(self.val_mode), dtype=torch.int64)
        if not labelled:                                              # replay: no ring keys, the collate fills gt = 24 / obs = False
            sample["ring_stats"] = {"n_tokens": I, "kind": row["kind"], "ring_key": key, "seed": seed, "ring_labels": False,
                                    "n_atoms_assembly": int(full.array_length()), "t_read_s": t1 - t0, "t_parse_s": t2 - t1,
                                    "t_pipeline_s": t4 - t3, "t_post_s": time.perf_counter() - t4}
            if self.return_atom_array:
                sample["atom_array"] = oa
            return sample
        ts = get_token_starts(oa, add_exclusive_stop=False).astype(np.int64)
        if len(ts) != I:
            raise AssertionError(f"token count mismatch: atom array {len(ts)} vs feats {I}")
        uid_atoms = np.asarray(oa.ring_uid)
        uid = uid_atoms[ts]
        if not (np.minimum.reduceat(uid_atoms, ts) == np.maximum.reduceat(uid_atoms, ts)).all():
            raise AssertionError("a token spans atoms of different pre-pipeline residues (ring_uid inconsistent)")
        ttype = feats["token_type"].numpy()
        atomized = np.asarray(oa.atomize)[ts].astype(bool)
        copy = (feats["insertion_source_idx"].numpy() >= 0)
        ok_types = (TOKEN_TYPE_PROTEIN, TOKEN_TYPE_NCAA) if self.cfg["include_ncaa"] else (TOKEN_TYPE_PROTEIN,)
        elig = np.isin(ttype, ok_types) & ~atomized & ~copy & (uid >= 0)
        dup = np.zeros(I, dtype=bool)
        uu, first = np.unique(np.where(elig, uid, -1 - np.arange(I)), return_index=True)   # unique per eligible uid
        keep_first = np.zeros(I, dtype=bool)
        keep_first[first] = True
        dup = elig & ~keep_first
        elig &= ~dup
        # identity gate: eligible standard-protein tokens must carry the residue name of their pre-pipeline residue
        rt = feats["residue_type"].numpy()
        std = elig & (ttype == TOKEN_TYPE_PROTEIN)
        tok_names = np.array([str(ENC.idx_to_token[int(x)]) for x in rt[std]])
        if not (tok_names == tab.res_name[uid[std]]).all():
            bad = np.flatnonzero(tok_names != tab.res_name[uid[std]])[:5]
            raise AssertionError(f"token residue names disagree with the pre-pipeline residues: {tok_names[bad]} vs "
                                 f"{tab.res_name[uid[std]][bad]}")
        state, obs = labels_to_tokens(labels, uid, elig)
        feats["ring_state_gt"] = torch.from_numpy(state)
        feats["ring_obs"] = torch.from_numpy(obs)

        sample["ring_stats"] = {**labels.stats, "n_tokens": I, "n_eligible": int(elig.sum()), "n_atomized_tokens": int(atomized.sum()),
                                "n_dup_tokens": int(dup.sum()), "n_pairs_obs_tok": int(np.triu(obs, 1).sum()),
                                "n_pairs_lab_tok": int(np.triu((state < NO_INTERACTION), 1).sum()), "n_res_pre": tab.n_res,
                                "kind": row["kind"], "ring_key": key, "seed": seed, "n_atoms_assembly": int(full.array_length()),
                                "t_read_s": t1 - t0, "t_parse_s": t2 - t1, "t_ring_join_s": t3 - t2, "t_pipeline_s": t4 - t3,
                                "t_post_s": time.perf_counter() - t4}
        if self.return_atom_array:
            sample["atom_array"] = oa
        if self.return_debug:
            sample["ring_debug"] = dict(row=row, token_uid=uid, token_eligible=elig, residue_unit=tab.unit, residue_id=tab.res_id,
                                        residue_name=tab.res_name, labels=labels)
        return sample

    # ---- collation / loader -------------------------------------------------------------------------------------------
    @staticmethod
    def split_failed(batch: list[dict]) -> tuple[list[dict], list[dict]]:
        """(valid samples, strict-mode sentinels) of a list of dataset items."""
        return [x for x in batch if not x.get("failed")], [x for x in batch if x.get("failed")]

    @staticmethod
    def collate(batch: list[dict], fixed_dimensions: dict | None = None, num_replicates: int = 1):
        """RFD4's own ``collate_batch`` on the VALID items; strict-mode sentinels are dropped (use ``split_failed`` first to
        record them). Returns None when every item failed."""
        from rfproteina.datapipes.collate import collate_batch

        valid = [x for x in batch if not x.get("failed")]
        if not valid:
            return None
        return collate_batch(valid, schema=SCHEMA, fixed_dimensions=fixed_dimensions, keep_atom_array=False,
                             num_replicates=num_replicates)


def collate_batch_drop_failed(batch: list, schema=SCHEMA, fixed_dimensions=None, keep_atom_array: bool = True, num_replicates: int = 1):
    """Drop-in for ``rfproteina.datapipes.collate.collate_batch`` (same arguments) that first removes the strict-mode sentinels of
    ``RingDataset(substitute_on_failure=False)`` (``{'failed': True, ...}``), so a validation row that cannot be built is DROPPED (and counted by
    ``RingDataset.counts()['failed']``) instead of being replaced by another row. A batch in which EVERY row failed raises (a systematic fault,
    e.g. a missing store, must not pass silently; upstream's validation_step cannot take an empty batch)."""
    from rfproteina.datapipes.collate import collate_batch

    good = [x for x in batch if not (isinstance(x, dict) and x.get("failed"))]
    if not good:
        reasons = sorted({str(x.get("reason")) for x in batch if isinstance(x, dict)})
        raise RuntimeError(f"all {len(batch)} rows of a strict-mode batch failed to build: {reasons}")
    return collate_batch(good, schema=schema, fixed_dimensions=fixed_dimensions, keep_atom_array=keep_atom_array, num_replicates=num_replicates)


def make_loader(ds: RingDataset, batch_size: int = 8, num_workers: int = 0, fixed_dimensions: dict | None = None,
                num_replicates: int = 1, shuffle: bool = False, **kw):
    """DataLoader with spawn workers (like configs/dataloaders/default.yaml) and RFD4's collate."""
    from functools import partial

    extra = dict(kw)
    if num_workers > 0:
        extra.setdefault("multiprocessing_context", "spawn")
        extra.setdefault("persistent_workers", True)
        extra.setdefault("prefetch_factor", 2)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False,
        collate_fn=partial(_collate_fn, fixed_dimensions=fixed_dimensions, num_replicates=num_replicates), **extra)


def _collate_fn(batch, fixed_dimensions=None, num_replicates=1):
    return RingDataset.collate(batch, fixed_dimensions=fixed_dimensions, num_replicates=num_replicates)
