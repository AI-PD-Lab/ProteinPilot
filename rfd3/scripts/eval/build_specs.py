#!/usr/bin/env python
"""Natural-structure spec builder of the HARMONIA-v2 guardrail harness (WP-EVAL step 1, CPU).

Specs are generated with the TRAINING label function (`foundry.data.ss_adj.get_ss_adj`, loaded from the repo
file) from real reference structures, so a spec can never contain a contact/non-contact the native fold does
not have ("round trip"; the old hand-written ferredoxin spec asked for a ring that the natural fold does not
have, N002 A2 B14). Dialect = RFD1-faithful v2: loops masked, ADJ only between helix/strand blocks, fill = 2,
diagonal 0 everywhere (the `adj_pairs` inference convention and WP-LABEL's choice), covalent-neighbour block
pairs masked.

Arms per reference (key `<target>__<arm>`):
  full         every H/E block pair = native 0/1, intra-block zeros as the label function writes them, SS fully given
  sparse8      k native contacts (k = 8, capped at ceil(n_contacts/2) so that some native contact stays hidden),
               everything else 2, SS fully given
  dense0_true  the SAME k contacts as sparse8 + EVERY native non-contact pair as 0; the remaining native contacts
               stay masked (2). Zeros are never anything but native zeros ("dense zeros without false zeros")
  ss_only      SS tensor only (no adj_file): the 'SS given, adjacency absent' regime
plus `uncond_L<len>__unconditional` (only `length`) for the unconditional panel.

usage (any env with numpy+biotite; pandas+pyarrow only to read the PDB-mirror index, i.e. atomworks_env):
  python build_specs.py build --pack /hnvme/.../v2/eval/specs_v2.tar.gz --manifest-out .../specs_v2_manifest.json
  python build_specs.py render --specs-tar specs_v2.tar.gz --root /tmp/specs      # -> /tmp/specs/inputs.json
Inputs live as .npy (the inference parser accepts .pt or .npy) inside ONE tar (/hnvme inode cap); the JSON
inside carries the placeholder @SPEC_ROOT@ that `render` replaces by the extraction directory.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import sys
import tarfile
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import design_metrics as dm  # noqa: E402
import eval_common as ec  # noqa: E402

SPEC_VERSION = 2
ARMS = ("full", "sparse8", "dense0_true", "ss_only")
UNCOND_LENGTHS = (64, 120, 200)
K_SPARSE = 8

# name, PDB id, chain, description. Chosen by id (gap-free single-domain chains in the PDB mirror, label function run):
#   1URN  U1A RRM (chain A, res 2-97, L=96): ferredoxin-like beta-alpha-beta-beta-alpha-beta (E H E E H E), 9 contacts.
#         (1RIS, the id first named in SPEC_v2, gives only 3 strands under P-SEA = a poor ferredoxin control: kept as
#         the opt-in extra `ferredoxin_1ris`.)
#   3CHY  CheY (A, 128 res): flavodoxin-like / Rossmann-like 3-layer a/b/a, 5 parallel strands + 4 helix blocks
#   1TIT  titin I27 (A, 89 res): Ig-like beta-sandwich, 6 strand blocks
#   2HMQ  hemerythrin (A): 4-helix bundle; the N-terminal tail with a numbering gap is trimmed automatically
#         (longest gap-free segment), bundle blocks keep 3 of 6 pairs in contact
#   2CCY  cytochrome c' (A, 127 res): up-down 4-helix bundle, 5 of 6 block pairs in contact (dense bundle)
REFERENCES = [
    dict(name="ferredoxin_1urn", pdb="1urn", chain="A", fold="ferredoxin-like (U1A RRM)"),
    dict(name="rossmann_3chy", pdb="3chy", chain="A", fold="Rossmann-like / flavodoxin-like alpha/beta (CheY)"),
    dict(name="sandwich_1tit", pdb="1tit", chain="A", fold="beta-sandwich (Ig-like, titin I27)"),
    dict(name="bundle_2hmq", pdb="2hmq", chain="A", fold="4-helix bundle (hemerythrin, N-terminal tail trimmed)"),
    dict(name="bundle_2ccy", pdb="2ccy", chain="A", fold="4-helix bundle, dense (cytochrome c')"),
]
EXTRA_REFERENCES = [
    dict(name="ferredoxin_1ris", pdb="1ris", chain="A", fold="ferredoxin-like, only 3 P-SEA strands (ribosomal protein S6)"),
]


# --------------------------------------------------------------------------- reference loading
def mirror_bytes(pdb_id: str, tar_path: str = ec.PDB_MIRROR_TAR, index_path: str = ec.PDB_MIRROR_INDEX) -> bytes:
    """Gzipped mmCIF bytes of `pdb_id` straight out of the mirror tar via the offset index (needs pandas+pyarrow)."""
    import pandas as pd

    idx = pd.read_parquet(index_path)
    row = idx[idx.pdb_id == pdb_id.lower()]
    if row.empty:
        raise KeyError(f"{pdb_id} not in {index_path}")
    r = row.iloc[0]
    with open(tar_path, "rb") as fh:
        fh.seek(int(r["offset_data"]))
        return fh.read(int(r["size"]))


def gap_free_segment(chain: "ec.Chain", max_len: int | None = None) -> tuple[int, int]:
    """Longest [lo, hi) run of residues with consecutive residue numbers and complete N/CA/C."""
    ok = np.isfinite(chain.n).all(1) & np.isfinite(chain.ca).all(1) & np.isfinite(chain.c).all(1)
    best = (0, 0)
    start = 0
    for i in range(1, len(chain) + 1):
        if i == len(chain) or not ok[i] or chain.res_ids[i] != chain.res_ids[i - 1] + 1:
            seg = (start, i)
            if ok[start] and seg[1] - seg[0] > best[1] - best[0]:
                best = seg
            start = i
    if max_len and best[1] - best[0] > max_len:
        best = (best[0], best[0] + max_len)
    return best


def select_residue_range(atoms, lo: int, hi: int):
    """Atoms of residues [lo, hi) of a single-chain AtomArray, by residue POSITION (robust to duplicate numbers / insertion codes)."""
    import biotite.structure as struc

    starts = struc.get_residue_starts(atoms)
    res_idx = np.zeros(atoms.array_length(), dtype=int)
    res_idx[starts] = 1
    res_idx = np.cumsum(res_idx) - 1
    return atoms[(res_idx >= lo) & (res_idx < hi)]


def load_reference(ref: dict, lbl, mirror_tar=ec.PDB_MIRROR_TAR, mirror_index=ec.PDB_MIRROR_INDEX, path=None):
    """-> (chain_atoms AtomArray restricted to the gap-free segment, Chain) for one reference."""
    import biotite.structure as struc

    if path:
        atoms = ec.read_structure(path, os.path.basename(path))
    else:
        atoms = ec.read_structure(mirror_bytes(ref["pdb"], mirror_tar, mirror_index), ref["pdb"] + ".cif.gz")
    atoms = atoms[struc.filter_amino_acids(atoms) & (atoms.element != "H") & (atoms.chain_id == ref["chain"])]
    chains = ec.protein_chains(atoms)
    if len(chains) != 1:
        raise ValueError(f"{ref['name']}: chain {ref['chain']} not found / ambiguous")
    ch = chains[0]
    lo, hi = gap_free_segment(ch, ref.get("max_len"))
    sel = select_residue_range(atoms, lo, hi)
    chains = ec.protein_chains(sel)
    assert len(chains) == 1 and len(chains[0]) == hi - lo, "segment extraction mismatch"
    return sel, chains[0]


# --------------------------------------------------------------------------- native labels -> target record
def native_labels(chain_atoms, lbl):
    """(ss [L] int, adj [L,L] float) from the TRAINING label function, loops masked (loop_adj=False)."""
    ss, adj = lbl.get_ss_adj(chain_atoms, loop_adj=False)
    return np.asarray(ss).astype(int), np.asarray(adj, dtype=np.float32)


def pair_state(adj: np.ndarray, blocks: list[dict], i: int, j: int) -> int:
    """Native state of an inter-block rectangle: 1 if any entry is 1, 0 if all 0, else 2."""
    rect = adj[np.ix_(ec.block_idx(blocks, i), ec.block_idx(blocks, j))]
    u = set(np.unique(rect).astype(int).tolist())
    if u == {0}:
        return 0
    if 1 in u:
        if 0 in u:
            raise ValueError(f"mixed 0/1 rectangle for block pair ({i},{j}): label function is not block-uniform")
        return 1
    return 2


def build_target(ref: dict, chain_atoms, chain: "ec.Chain", lbl) -> tuple[dict, dict]:
    """Target record (JSON-able manifest entry) + arrays {ss, adj_label, ca, cb}."""
    ss, adj = native_labels(chain_atoms, lbl)
    L = len(chain)
    assert len(ss) == L and adj.shape == (L, L)
    blocks = ec.he_blocks_from_ss(ss)
    nb = len(blocks)
    cov = ec.covalent_neighbours(blocks)
    pos, neg, cov_masked = [], [], []
    pair_n6, pair_min = {}, {}
    for i in range(nb):
        for j in range(i + 1, nb):
            st = pair_state(adj, blocks, i, j)
            ia, ib = ec.block_idx(blocks, i), ec.block_idx(blocks, j)
            pair_n6[f"{i}-{j}"] = ec.block_n_below(chain.cb, ia, ib)
            pair_min[f"{i}-{j}"] = round(ec.block_min_dist(chain.cb, ia, ib), 3)
            if (i, j) in cov:
                cov_masked.append([i, j])
            elif st == 1:
                pos.append([i, j])
            elif st == 0:
                neg.append([i, j])
    h = nb // 2
    target = {
        "name": ref["name"], "pdb_id": ref["pdb"], "chain": ref["chain"], "fold": ref["fold"], "n_res": L,
        "res_range": [int(chain.res_ids[0]), int(chain.res_ids[-1])], "n_blocks": nb, "blocks": blocks,
        "half1_blocks": list(range(h)), "half2_blocks": list(range(h, nb)),
        "pos_pairs": pos, "neg_pairs": neg, "cov_masked_pairs": cov_masked,
        "pair_n6": pair_n6, "pair_min": pair_min, "ss": ss.tolist(),
    }
    nat = dm.geometry_metrics(chain.ca, chain.cb, chain.heavy, chain.heavy_res, ss)
    nat.update(dm.conditioned_metrics(chain.cb, ss, target, {"pos_pairs": pos, "neg_pairs": neg, "ss": ss.tolist()}, chain.cb_leg))
    target["native"] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in nat.items()}
    return target, {"ss": ss, "adj_label": adj, "ca": chain.ca, "cb": chain.cb}


# --------------------------------------------------------------------------- arms
def sparse_k(n_pos: int, k_req: int = K_SPARSE) -> int:
    """Contacts revealed in the sparse arms: k_req, but never all of them (some native contact stays hidden)."""
    if n_pos <= 1:
        return n_pos
    if n_pos == 2:
        return 1
    return min(k_req, int(math.ceil(n_pos / 2)))


def _rng_for(target_name: str, seed: int) -> np.random.RandomState:
    return np.random.RandomState((zlib.crc32(target_name.encode()) ^ (seed * 2654435761)) & 0x7FFFFFFF)


def assemble_adj(L: int, blocks: list[dict], pos, neg, intra_zero: bool = False) -> np.ndarray:
    """All-2 matrix with the given block-pair rectangles set; diag 0; optional H/E intra-block zeros."""
    adj = np.full((L, L), ec.ADJ_MASK, dtype=np.float32)
    for pairs, v in ((pos, ec.ADJ_YES), (neg, ec.ADJ_NO)):
        for i, j in pairs:
            ia, ib = ec.block_idx(blocks, i), ec.block_idx(blocks, j)
            adj[np.ix_(ia, ib)] = v
            adj[np.ix_(ib, ia)] = v
    if intra_zero:
        for b in blocks:
            ia = np.arange(b["start"], b["stop"])
            adj[np.ix_(ia, ia)] = ec.ADJ_NO
    np.fill_diagonal(adj, ec.ADJ_NO)
    return adj


def make_arms(target: dict, seed: int = 0, k_req: int = K_SPARSE) -> dict[str, dict]:
    """arm name -> {record, ss (np int | None), adj (np float32 | None)}."""
    L, blocks = target["n_res"], target["blocks"]
    pos_all, neg_all = [tuple(p) for p in target["pos_pairs"]], [tuple(p) for p in target["neg_pairs"]]
    ss = np.asarray(target["ss"], dtype=np.float32)
    k = sparse_k(len(pos_all), k_req)
    rng = _rng_for(target["name"], seed)
    sel = sorted(rng.choice(len(pos_all), size=k, replace=False).tolist()) if k else []
    pos_sel = [pos_all[i] for i in sel]

    arms: dict[str, dict] = {}

    def rec(arm, pos, neg, has_adj, **extra):
        return {"target": target["name"], "arm": arm, "key": f"{target['name']}__{arm}", "length": L,
                "ss": ss.astype(int).tolist(), "has_adj": has_adj,
                "pos_pairs": [list(p) for p in pos], "neg_pairs": [list(p) for p in neg], **extra}

    arms["full"] = dict(
        record=rec("full", pos_all, neg_all, True, intra_zero=True, k_requested=None, k_used=len(pos_all)),
        ss=ss, adj=assemble_adj(L, blocks, pos_all, neg_all, intra_zero=True))
    arms["sparse8"] = dict(
        record=rec("sparse8", pos_sel, [], True, intra_zero=False, k_requested=k_req, k_used=k),
        ss=ss, adj=assemble_adj(L, blocks, pos_sel, [], intra_zero=False))
    arms["dense0_true"] = dict(
        record=rec("dense0_true", pos_sel, neg_all, True, intra_zero=False, k_requested=k_req, k_used=k),
        ss=ss, adj=assemble_adj(L, blocks, pos_sel, neg_all, intra_zero=False))
    arms["ss_only"] = dict(record=rec("ss_only", [], [], False, intra_zero=False), ss=ss, adj=None)
    return arms


def matrix_stats(adj: np.ndarray | None) -> dict:
    if adj is None:
        return {}
    off = ~np.eye(len(adj), dtype=bool)
    v = adj[off]
    return {"f0": float((v == 0).mean()), "f1": float((v == 1).mean()), "f2": float((v == 2).mean()),
            "n0": int((v == 0).sum()), "n1": int((v == 1).sum()), "n2": int((v == 2).sum())}


# --------------------------------------------------------------------------- legacy ferredoxin manifest
LEGACY_DIR = "/path/to/checkpoints/p6_benchmark"


def legacy_ferredoxin_manifest(blocks_json: str = LEGACY_DIR + "/ferredoxin_blocks.json",
                               manifest_json: str = LEGACY_DIR + "/ferredoxin_manifest.json") -> dict:
    """Scorer manifest for the OLD hand-written 64-residue ferredoxin benchmark (arms ferredoxin_{dense0,loopmask,allmask}),
    so that this harness can be validated against the legacy tables of N002/N003 (cfg_probe_scores.txt, half_sep
    over spans [3,32] vs [37,63], b1b4_n6 = ends_n6 between B1 and B4). Use with `score_designs.py score --manifest`."""
    bl = ec.read_json(blocks_json)
    fm = ec.read_json(manifest_json)
    order = ["B1", "A1", "B2", "B3", "A2", "B4"]
    blocks = [{"id": i, "type": "E" if n.startswith("B") else "H", "start": bl[n][0] - 1, "stop": bl[n][1]}
              for i, n in enumerate(order)]
    span_to_id = {tuple(bl[n]): i for i, n in enumerate(order)}
    tgt0 = fm["ferredoxin_dense0"]["target"]
    pos = sorted({tuple(sorted((span_to_id[tuple(a)], span_to_id[tuple(b)]))) for a, b in tgt0["contact_blocks"]})
    allp = [(i, j) for i in range(len(blocks)) for j in range(i + 1, len(blocks))]
    neg = [p for p in allp if p not in pos]
    ss = [int(x) for x in tgt0["ss_labels"]]
    target = {"name": "ferredoxin_legacy64", "pdb_id": "synthetic", "chain": "A", "fold": "legacy hand-written ferredoxin",
              "n_res": len(ss), "n_blocks": len(blocks), "blocks": blocks, "half1_span": [bl["half1"][0] - 1, bl["half1"][1]],
              "half2_span": [bl["half2"][0] - 1, bl["half2"][1]], "pos_pairs": [list(p) for p in pos],
              "neg_pairs": [list(p) for p in neg], "cov_masked_pairs": [], "pair_n6": {}, "ss": ss, "native": {}}
    arms = {}
    for name, with_neg in (("dense0", True), ("loopmask", True), ("allmask", False)):
        key = f"ferredoxin_{name}"
        arms[key] = {"target": target["name"], "arm": name, "key": key, "length": len(ss), "ss": ss, "has_adj": True,
                     "pos_pairs": [list(p) for p in pos], "neg_pairs": [list(p) for p in neg] if with_neg else []}
    return {"spec_version": 0, "dialect": "legacy phase-1..5 benchmark", "seed": None, "k_requested": None,
            "targets": {target["name"]: target}, "arms": arms, "unconditional": {}}


# --------------------------------------------------------------------------- build / pack / render
def build(out_dir: str, refs=None, seed: int = 0, k_req: int = K_SPARSE, uncond_lengths=UNCOND_LENGTHS,
          repo: str | None = None, ref_files: dict | None = None, mirror_tar=ec.PDB_MIRROR_TAR,
          mirror_index=ec.PDB_MIRROR_INDEX) -> dict:
    """Write tensors/*.npy, refs.npz, inputs_template.json and manifest.json under out_dir; return the manifest."""
    lbl = ec.load_label_module(repo)
    refs = REFERENCES if refs is None else refs
    ref_files = ref_files or {}
    os.makedirs(os.path.join(out_dir, "tensors"), exist_ok=True)
    manifest = {"spec_version": SPEC_VERSION, "dialect": "v2: loops masked, ADJ on H/E blocks only, fill=2, diag=0",
                "seed": seed, "k_requested": k_req, "label_function": "foundry.data.ss_adj.get_ss_adj(loop_adj=False)",
                "targets": {}, "arms": {}, "unconditional": {}}
    inputs: dict[str, dict] = {}
    refs_npz: dict[str, np.ndarray] = {}
    for ref in refs:
        atoms, chain = load_reference(ref, lbl, mirror_tar, mirror_index, path=ref_files.get(ref["name"]))
        target, arr = build_target(ref, atoms, chain, lbl)
        manifest["targets"][ref["name"]] = target
        refs_npz[ref["name"] + "_ca"] = arr["ca"].astype(np.float32)
        refs_npz[ref["name"] + "_cb"] = arr["cb"].astype(np.float32)
        for arm_name, a in make_arms(target, seed, k_req).items():
            r = a["record"]
            key = r["key"]
            spec = {"dialect": 2, "length": str(target["n_res"]),
                    "extra": {"harmonia_eval": {"spec_version": SPEC_VERSION, "target": ref["name"], "arm": arm_name}}}
            ss_rel = f"tensors/{key}.ss.npy"
            np.save(os.path.join(out_dir, ss_rel), a["ss"].astype(np.float32))
            spec["ss_file"] = f"{ec.SPEC_ROOT_TOKEN}/{ss_rel}"
            if a["adj"] is not None:
                adj_rel = f"tensors/{key}.adj.npy"
                np.save(os.path.join(out_dir, adj_rel), a["adj"].astype(np.float32))
                spec["adj_file"] = f"{ec.SPEC_ROOT_TOKEN}/{adj_rel}"
            r["matrix"] = matrix_stats(a["adj"])
            inputs[key] = spec
            manifest["arms"][key] = r
    for L in uncond_lengths:
        tname = f"uncond_L{L:03d}"
        key = f"{tname}__unconditional"
        inputs[key] = {"dialect": 2, "length": str(L),
                       "extra": {"harmonia_eval": {"spec_version": SPEC_VERSION, "target": tname, "arm": "unconditional"}}}
        manifest["unconditional"][key] = {"target": tname, "arm": "unconditional", "key": key, "length": L}
    np.savez_compressed(os.path.join(out_dir, "refs.npz"), **refs_npz)
    ec.write_json(os.path.join(out_dir, "inputs_template.json"), inputs, indent=1)
    ec.write_json(os.path.join(out_dir, "manifest.json"), manifest, indent=None)
    return manifest


def pack_dir(src_dir: str, tar_path: str) -> str:
    """Deterministic .tar.gz of src_dir's files (sorted names, mtime 0): identical inputs give an identical sha256."""
    os.makedirs(os.path.dirname(os.path.abspath(tar_path)), exist_ok=True)
    names = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            names.append(os.path.relpath(os.path.join(root, f), src_dir))
    names.sort()
    with open(tar_path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tf:
                for n in names:
                    ti = tarfile.TarInfo(n)
                    data = open(os.path.join(src_dir, n), "rb").read()
                    ti.size, ti.mtime, ti.mode, ti.uid, ti.gid = len(data), 0, 0o644, 0, 0
                    tf.addfile(ti, io.BytesIO(data))
    return tar_path


def render_inputs(template: dict, root: str) -> dict:
    """Replace the @SPEC_ROOT@ token in every string of the template by `root`."""
    def rec(o):
        if isinstance(o, str):
            return o.replace(ec.SPEC_ROOT_TOKEN, root)
        if isinstance(o, dict):
            return {k: rec(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rec(v) for v in o]
        return o
    return rec(template)


def unpack_and_render(specs_tar: str, root: str) -> tuple[dict, dict]:
    """Extract the pack into root, write root/inputs.json (rendered); return (inputs, manifest)."""
    ec.safe_extract(specs_tar, root)
    template = ec.read_json(os.path.join(root, "inputs_template.json"))
    inputs = render_inputs(template, os.path.abspath(root))
    ec.write_json(os.path.join(root, "inputs.json"), inputs, indent=1)
    return inputs, ec.read_json(os.path.join(root, "manifest.json"))


def read_manifest_from_tar(specs_tar: str) -> dict:
    with tarfile.open(specs_tar, "r:*") as tf:
        return json.load(tf.extractfile("manifest.json"))


def summary_text(manifest: dict) -> str:
    lines = [f"spec_version {manifest['spec_version']} | {manifest['dialect']} | seed {manifest['seed']} | k_req {manifest['k_requested']}"]
    for name, t in manifest["targets"].items():
        nat = t["native"]
        lines.append(f"- {name}: {t['pdb_id'].upper()} chain {t['chain']} res {t['res_range'][0]}-{t['res_range'][1]} "
                     f"(L={t['n_res']}), {t['fold']}; {t['n_blocks']} H/E blocks "
                     f"({''.join(b['type'] for b in t['blocks'])}); native contacts {len(t['pos_pairs'])}, "
                     f"non-contacts {len(t['neg_pairs'])}, covalent-masked {len(t['cov_masked_pairs'])}; "
                     f"native half_sep {ec.fmt(nat.get('half_sep'))} A, ends_n6 {ec.fmt(nat.get('ends_n6'), 0)}, Rg/N^0.4 {ec.fmt(nat['rg_n04'], 3)}, lr_cb {ec.fmt(nat['lr_cb'])}")
    lines.append("arm matrices (off-diagonal fractions of states 0/1/2; k = contacts revealed):")
    for key, a in manifest["arms"].items():
        m = a.get("matrix") or {}
        st = (f"0:{m['f0']:.3f} 1:{m['f1']:.3f} 2:{m['f2']:.3f}  (n0 {m['n0']}, n1 {m['n1']})") if m else "no adj_file"
        lines.append(f"  {a['key']:<28} L={a['length']:<4} pos {len(a['pos_pairs']):>2} neg {len(a['neg_pairs']):>2} "
                     f"k_used {a.get('k_used')}  {st}")
    lines.append("unconditional panel: " + ", ".join(f"L={u['length']}" for u in manifest["unconditional"].values()))
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build specs from the reference PDBs and pack them into one tar.gz")
    b.add_argument("--pack", default=ec.SPECS_TAR_DEFAULT, help="output .tar.gz (default %(default)s)")
    b.add_argument("--manifest-out", default=None, help="also write the manifest.json here (default <pack stem>_manifest.json)")
    b.add_argument("--work-dir", default=None, help="keep the loose files here instead of a temp dir")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--k-sparse", type=int, default=K_SPARSE)
    b.add_argument("--uncond-lengths", type=int, nargs="+", default=list(UNCOND_LENGTHS))
    b.add_argument("--repo", default=None, help="repo whose foundry/data/ss_adj.py provides the label function")
    b.add_argument("--extra-refs", nargs="*", default=[], metavar="NAME",
                   help="also build these opt-in references: " + ", ".join(r["name"] for r in EXTRA_REFERENCES))
    b.add_argument("--only", nargs="*", default=None, metavar="NAME", help="build only these reference names")
    b.add_argument("--ref", action="append", default=[], metavar="NAME=PATH",
                   help="read reference NAME from a local .cif/.pdb(.gz) instead of the PDB mirror")
    lg = sub.add_parser("legacy-manifest", help="write the scorer manifest of the old 64-res ferredoxin benchmark")
    lg.add_argument("--out", required=True)
    r = sub.add_parser("render", help="extract a spec pack and write the rendered inputs.json")
    r.add_argument("--specs-tar", default=ec.SPECS_TAR_DEFAULT)
    r.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        ref_files = dict(x.split("=", 1) for x in a.ref)
        refs = [r for r in REFERENCES + [x for x in EXTRA_REFERENCES if x["name"] in a.extra_refs]
                if a.only is None or r["name"] in a.only]
        tmp = None if a.work_dir else tempfile.TemporaryDirectory(prefix="harmonia_specs_")
        work = a.work_dir or tmp.name
        try:
            man = build(work, refs=refs, seed=a.seed, k_req=a.k_sparse, uncond_lengths=a.uncond_lengths, repo=a.repo,
                        ref_files=ref_files)
            pack_dir(work, a.pack)
        finally:
            if tmp:
                tmp.cleanup()
        mo = a.manifest_out or a.pack.replace(".tar.gz", "") + "_manifest.json"
        ec.write_json(mo, man, indent=None)
        print(summary_text(man))
        print(f"packed {a.pack} ({os.path.getsize(a.pack)} bytes); manifest {mo}")
    elif a.cmd == "legacy-manifest":
        ec.write_json(a.out, legacy_ferredoxin_manifest(), indent=None)
        print(f"wrote {a.out}")
    else:
        inputs, man = unpack_and_render(a.specs_tar, a.root)
        print(f"rendered {len(inputs)} specs -> {os.path.join(a.root, 'inputs.json')}")


if __name__ == "__main__":
    main()
