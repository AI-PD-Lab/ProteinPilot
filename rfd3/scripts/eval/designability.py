#!/usr/bin/env python
"""Designability of eval designs (WP-EVAL step 4; a separate GPU step, NOT part of the per-checkpoint generation job).

Protocol = the MicroGreens `scripts/protein_designability.py` one, so numbers are comparable with that campaign:
  1. LigandMPNN `ligand_mpnn` model (protein-only PDB = no ligand context), T = 0.1, K = 8 sequences per design, seed 37
  2. ESMFold2 (pinned HF snapshots; the 6B ESMC language model is loaded and asserted, a missing LM silently gives a
     random one) folds every sequence, num_loops=4, num_sampling_steps=20, fold seed 0
  3. scRMSD = CA RMSD (optimal superposition) of the refolded structure vs the generated backbone;
     DESIGNABLE = best of the K sequences has scRMSD < 2 A AND pLDDT > 80 (the 70 variant is reported too)
  4. fold-conditioned designs: TM-score of the generated backbone (and of the refolded one) to the REFERENCE fold the
     spec was built from; criterion TM >= 0.5. Index-aligned numpy TM-score (design residue i <-> reference residue i; a
     LOWER bound of the TM-align value, so passing it implies passing TM-align). The sequence-order-independent value
     is available through tm_score.foldseek_tm (foldseek exhaustive TM-align) but is not wired into the stages.
Joint metric reported: designable AND TM >= 0.5 ("designable and on-fold").

Stages (the sbatch `designability.sbatch` runs them; each is also CPU-testable except `fold`):
  prepare   <eval tar> -> PDBs of the selected designs (+ index.json), then LigandMPNN sequences (GPU 0 or CPU)
  fold      one shard of the (design, sequence) list through ESMFold2 -> shard JSON   (GPU; needs esm + transformers)
  summarize shard JSONs + index -> designability.json / .csv / .md with Wilson CIs per cell      (CPU)

  python designability.py prepare --tar <tag>.tar.gz --specs-tar specs_v2.tar.gz --work $TMP/d --cfgs 1.0 1.5 \
         --arms full sparse8 dense0_true --max-per-cell 8 [--skip-mpnn]
  python designability.py fold --work $TMP/d --shard 0/4 --out $TMP/d/fold_0.json
  python designability.py summarize --work $TMP/d --out-prefix <dir>/<tag>.designability

STATUS (2026-10-14): written + the CPU stages (prepare w/o MPNN, summarize, TM) tested; `fold` and the LigandMPNN call
were NOT executed here (GPU step, no job submitted). Tooling verified present: esmfold2 env (esm 3.3.0), ESMFold2 + ESMC-6B
snapshots in the triad HF cache, LigandMPNN ligand_mpnn weights, foldseek.
"""
from __future__ import annotations

import argparse
import glob
import io
import math
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import eval_common as ec  # noqa: E402
import tm_score as tm  # noqa: E402

HF_CACHE = "/path/to/work/cache/huggingface/hub"
F2 = f"{HF_CACHE}/models--biohub--ESMFold2/snapshots/8fc3ff471022fdce52c77030685eb775de0c00a3"
ESMC = f"{HF_CACHE}/models--biohub--ESMC-6B/snapshots/45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a"
MG_W = "/path/to/work"
LMPNN_SRC = "/path/to/home/Protein_Design_Models/Macrocycledesign/LigandMPNN"
SCRMSD_MAX, PLDDT_MIN, PLDDT_MIN_LOOSE, TM_MIN = 2.0, 80.0, 70.0, 0.5
ESM_PY = "/path/to/env/bin/python"


# --------------------------------------------------------------------------- prepare
def select_designs(tar_path: str, cfgs, arms, targets, max_per_cell: int):
    """[(member_name, cell dict)] of the first `max_per_cell` designs (by batch, model) of each selected cell."""
    cells: dict = {}
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf:
            if not m.isfile() or "/designs/" not in m.name:
                continue
            mt = ec.DESIGN_RE.match(os.path.basename(m.name))
            cm = ec.CFG_DIR_RE.search(m.name.split("/")[-2])
            if not mt or not cm or not m.name.endswith(".cif.gz"):
                continue
            key = mt.group("key")
            target, _, arm = key.partition("__")
            cfg = float(cm.group("cfg"))
            if (cfgs and cfg not in cfgs) or (arms and arm not in arms) or (targets and target not in targets):
                continue
            cells.setdefault((target, arm, cfg), []).append((int(mt.group("batch")), int(mt.group("model")), m.name, key))
    sel = []
    for (target, arm, cfg), items in sorted(cells.items()):
        for b, mo, name, key in sorted(items)[:max_per_cell]:
            sel.append((name, {"target": target, "arm": arm, "cfg": cfg, "key": key, "batch": b, "model": mo}))
    return sel


def write_design_pdb(data: bytes, name: str, path: str) -> int:
    """Write the first protein chain of a design as a PDB; returns its length."""
    import biotite.structure.io.pdb as pdb

    atoms = ec.read_structure(data, name)
    import biotite.structure as struc

    atoms = atoms[struc.filter_amino_acids(atoms) & (atoms.element != "H")]
    first = atoms.chain_id[0]
    atoms = atoms[atoms.chain_id == first]
    f = pdb.PDBFile()
    f.set_structure(atoms)
    f.write(path)
    return len(set(atoms.res_id))


def prepare(a) -> dict:
    os.makedirs(os.path.join(a.work, "pdbs"), exist_ok=True)
    sel = select_designs(a.tar, a.cfgs, a.arms, a.targets, a.max_per_cell)
    if not sel:
        raise SystemExit("no designs selected")
    want = {n: c for n, c in sel}
    index = []
    with tarfile.open(a.tar, "r:*") as tf:
        for m in tf:
            if m.name in want:
                c = want[m.name]
                stem = f"{c['target']}__{c['arm']}__cfg{c['cfg']:.1f}__b{c['batch']}m{c['model']}"
                pdb_path = os.path.join(a.work, "pdbs", stem + ".pdb")
                n = write_design_pdb(tf.extractfile(m).read(), m.name, pdb_path)
                index.append(dict(c, stem=stem, pdb=pdb_path, n_res=n))
    ec.write_json(os.path.join(a.work, "index.json"), index)
    # reference CA traces for the TM criterion
    refs_out = {}
    if a.specs_tar:
        with tarfile.open(a.specs_tar, "r:*") as tf:
            raw = tf.extractfile("refs.npz").read()
        z = np.load(io.BytesIO(raw))
        for t in {c["target"] for c in index}:
            if f"{t}_ca" in z:
                refs_out[t] = z[f"{t}_ca"]
        np.savez_compressed(os.path.join(a.work, "refs_ca.npz"), **{k: v for k, v in refs_out.items()})
    print(f"prepared {len(index)} designs ({len({(c['target'], c['arm'], c['cfg']) for c in index})} cells) in {a.work}")
    if not a.skip_mpnn:
        run_ligandmpnn(a.work, k=a.k, temperature=a.temperature, seed=a.mpnn_seed, python=a.python)
    return {"n": len(index)}


def run_ligandmpnn(work: str, k: int = 8, temperature: float = 0.1, seed: int = 37, python: str = ESM_PY):
    """LigandMPNN ligand_mpnn on all prepared PDBs in ONE process (`--pdb_path_multi`), sequences -> <work>/seqs/<stem>.fa."""
    index = ec.read_json(os.path.join(work, "index.json"))
    multi = os.path.join(work, "pdb_multi.json")
    ec.write_json(multi, {c["pdb"]: "" for c in index})
    out = os.path.join(work, "lmpnn_out")
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=f"{LMPNN_SRC}/_pydeps:{MG_W}/pydeps:" + os.environ.get("PYTHONPATH", ""))
    cmd = [python, "run.py", "--model_type", "ligand_mpnn", "--pdb_path_multi", multi, "--out_folder", out,
           "--checkpoint_ligand_mpnn", f"{MG_W}/ligandmpnn/model_params/ligandmpnn_v_32_010_25.pt",
           "--number_of_batches", "1", "--batch_size", str(k), "--temperature", str(temperature), "--seed", str(seed),
           "--save_stats", "0", "--pack_side_chains", "0"]
    r = subprocess.run(cmd, cwd=f"{MG_W}/ligandmpnn", env=env, capture_output=True, text=True)
    with open(os.path.join(work, "lmpnn.log"), "w") as fh:
        fh.write(r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"LigandMPNN failed rc={r.returncode}; see {work}/lmpnn.log")
    seqs = os.path.join(work, "seqs")
    os.makedirs(seqs, exist_ok=True)
    for fa in glob.glob(os.path.join(out, "seqs", "*.fa")):
        os.replace(fa, os.path.join(seqs, os.path.basename(fa)))
    print(f"LigandMPNN: {len(glob.glob(seqs + '/*.fa'))} fasta files")


def read_designed_seqs(fa_path: str, k: int) -> list[str]:
    """Designed sequences of a LigandMPNN fasta: record 0 is the input backbone's own sequence and is skipped."""
    seqs, cur = [], []
    for line in open(fa_path):
        line = line.strip()
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        elif line:
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return [s.replace(":", "") for s in seqs[1:1 + k]]


# --------------------------------------------------------------------------- fold (GPU)
def ca_from_pdb(path: str) -> np.ndarray:
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                     for l in open(path) if l.startswith("ATOM") and l[12:16].strip() == "CA"])


def fold_shard(a) -> list[dict]:
    """ESMFold2 refold of one shard of designs (round-robin over the index). Needs a GPU, esm and transformers."""
    import torch
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    si, sn = (int(x) for x in a.shard.split("/"))
    index = ec.read_json(os.path.join(a.work, "index.json"))[si::sn]
    refs = np.load(os.path.join(a.work, "refs_ca.npz")) if os.path.exists(os.path.join(a.work, "refs_ca.npz")) else {}
    model = ESMFold2Model.from_pretrained(F2, load_esmc=False)
    model.load_esmc(ESMC, precision="bf16")
    n_par = sum(p.numel() for p in model.parameters())
    assert n_par > 6.0e9, f"language model did not load ({n_par:,} params): refusing to fold"
    model = model.cuda().eval()
    builder = ESMFold2InputBuilder()
    print(f"# ESMFold2 loaded ({n_par:,} params); shard {a.shard}: {len(index)} designs", flush=True)
    rows = []
    for c in index:
        ca = ca_from_pdb(c["pdb"])
        fa = os.path.join(a.work, "seqs", c["stem"] + ".fa")
        row = dict(c, per_seq=[])
        if not os.path.exists(fa):
            row["error"] = "no sequences"
            rows.append(row)
            continue
        for i, seq in enumerate(read_designed_seqs(fa, a.k)):
            if len(seq) != len(ca):
                continue
            try:
                with torch.no_grad():
                    r = builder.fold(model, StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)]),
                                     num_loops=4, num_sampling_steps=20, seed=a.fold_seed)
                r = r[0] if isinstance(r, list) else r
                cx = r.complex
                L = len(seq)
                t2a = np.asarray(cx.token_to_atoms)
                names = np.asarray(cx.atom_names)
                pos = np.asarray(cx.atom_positions, dtype=np.float64)
                pca = np.array([pos[int(lo) + [str(x) for x in names[int(lo):int(hi)]].index("CA")] for lo, hi in t2a[:L]])
                pl = r.plddt.detach().cpu().numpy().reshape(-1)
                pl = float(np.mean(pl[:L]))
                pl = pl * 100.0 if pl <= 1.0 else pl
                ps = dict(i=i, scrmsd=tm.rmsd_superposed(pca, ca), plddt=pl)
                ref = refs.get(c["target"]) if hasattr(refs, "get") else None
                if ref is not None and len(ref) == len(pca):
                    ps["tm_pred_ref"] = tm.tm_score(pca, ref)
                row["per_seq"].append(ps)
            except Exception as e:  # noqa: BLE001
                row["per_seq"].append(dict(i=i, error=f"{type(e).__name__}: {str(e)[:120]}"))
        ref = refs.get(c["target"]) if hasattr(refs, "get") else None
        if ref is not None and len(ref) == len(ca):
            row["tm_design_ref"] = tm.tm_score(ca, ref)
        rows.append(row)
        ok = [s for s in row["per_seq"] if "scrmsd" in s]
        if ok:
            b = min(ok, key=lambda s: s["scrmsd"])
            print(f"  {c['stem']:<60} best scRMSD {b['scrmsd']:5.2f} pLDDT {b['plddt']:5.1f}", flush=True)
    ec.write_json(a.out, rows, indent=None)
    return rows


# --------------------------------------------------------------------------- summarize (CPU)
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def design_verdict(row: dict, plddt_min=PLDDT_MIN, scrmsd_max=SCRMSD_MAX, tm_min=TM_MIN) -> dict:
    ok = [s for s in row.get("per_seq", []) if "scrmsd" in s]
    out = {"n_seq": len(ok), "best_scrmsd": float("nan"), "best_plddt": float("nan"), "designable": False,
           "designable70": False, "tm_design_ref": row.get("tm_design_ref"), "tm_pred_ref": float("nan")}
    if not ok:
        return out
    best = min(ok, key=lambda s: s["scrmsd"])
    out["best_scrmsd"], out["best_plddt"] = best["scrmsd"], best["plddt"]
    out["tm_pred_ref"] = best.get("tm_pred_ref", float("nan"))
    out["designable"] = any(s["scrmsd"] < scrmsd_max and s["plddt"] > plddt_min for s in ok)
    out["designable70"] = any(s["scrmsd"] < scrmsd_max and s["plddt"] > PLDDT_MIN_LOOSE for s in ok)
    tmd = row.get("tm_design_ref")
    out["on_fold"] = (tmd is not None and tmd >= tm_min)
    out["designable_on_fold"] = bool(out["designable"] and out["on_fold"])
    return out


def summarize(rows: list[dict]) -> dict:
    """Per (target, arm, cfg): designable rate (+Wilson CI), on-fold rate, joint rate, medians."""
    cells: dict = {}
    for r in rows:
        v = design_verdict(r)
        cells.setdefault((r["target"], r["arm"], r["cfg"]), []).append(v)
    out = {}
    for (t, arm, cfg), vs in sorted(cells.items()):
        n = len(vs)
        scored = [v for v in vs if v["n_seq"] > 0]

        def rate(f):
            k = sum(1 for v in scored if v.get(f))
            return dict(zip(("p", "lo", "hi"), wilson(k, len(scored))), k=k, n=len(scored))
        tm_vals = [v["tm_design_ref"] for v in scored if v.get("tm_design_ref") is not None]
        out[f"{t}|{arm}|{cfg:.1f}"] = {
            "target": t, "arm": arm, "cfg": cfg, "n": n, "n_scored": len(scored),
            "designable": rate("designable"), "designable70": rate("designable70"),
            "on_fold": rate("on_fold"), "designable_on_fold": rate("designable_on_fold"),
            "median_best_scrmsd": float(np.median([v["best_scrmsd"] for v in scored])) if scored else float("nan"),
            "median_best_plddt": float(np.median([v["best_plddt"] for v in scored])) if scored else float("nan"),
            "median_tm_design_ref": float(np.median(tm_vals)) if tm_vals else float("nan"),
        }
    return out


def summary_md(cells: dict, tag: str = "") -> str:
    L = [f"# Designability {tag}", "",
         f"LigandMPNN K=8 T=0.1 -> ESMFold2; designable = best seq scRMSD < {SCRMSD_MAX:g} A and pLDDT > {PLDDT_MIN:g} "
         f"({PLDDT_MIN_LOOSE:g} in brackets); on-fold = index-aligned TM(design, reference) >= {TM_MIN}. Wilson 95% CIs.", "",
         "| target | arm | CFG | n | designable | (pLDDT>70) | on-fold | designable & on-fold | median scRMSD | median pLDDT | median TM |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]

    def pr(d):
        return f"{ec.fmt(d['p'], 2)} [{ec.fmt(d['lo'], 2)},{ec.fmt(d['hi'], 2)}] ({d['k']}/{d['n']})"
    for c in cells.values():
        L.append(f"| {c['target']} | {c['arm']} | {c['cfg']:.1f} | {c['n_scored']}/{c['n']} | {pr(c['designable'])} | {ec.fmt(c['designable70']['p'], 2)} | "
                 f"{pr(c['on_fold'])} | {pr(c['designable_on_fold'])} | {ec.fmt(c['median_best_scrmsd'])} | "
                 f"{ec.fmt(c['median_best_plddt'], 1)} | {ec.fmt(c['median_tm_design_ref'], 2)} |")
    return "\n".join(L) + "\n"


def cmd_summarize(a):
    index = ec.read_json(os.path.join(a.work, "index.json"))
    by_stem = {c["stem"]: c for c in index}
    rows = []
    for f in sorted(glob.glob(os.path.join(a.work, "fold_*.json"))):
        for r in ec.read_json(f):
            rows.append(r)
    missing = set(by_stem) - {r["stem"] for r in rows}
    if missing:
        print(f"WARNING: {len(missing)} designs have no fold result (failed/missing shard)", file=sys.stderr)
    cells = summarize(rows)
    ec.write_json(a.out_prefix + ".json", {"cells": cells, "rows": rows}, indent=None)
    md = summary_md(cells, os.path.basename(a.out_prefix))
    with open(a.out_prefix + ".md", "w") as fh:
        fh.write(md)
    print(md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tar", required=True)
    p.add_argument("--specs-tar", default=ec.SPECS_TAR_DEFAULT)
    p.add_argument("--work", required=True)
    p.add_argument("--cfgs", type=float, nargs="*", default=[1.0, 1.5])
    p.add_argument("--arms", nargs="*", default=["full", "sparse8", "dense0_true"])
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--max-per-cell", type=int, default=8)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--mpnn-seed", type=int, default=37)
    p.add_argument("--python", default=ESM_PY)
    p.add_argument("--skip-mpnn", action="store_true")
    f = sub.add_parser("fold")
    f.add_argument("--work", required=True)
    f.add_argument("--shard", default="0/1")
    f.add_argument("--out", required=True)
    f.add_argument("--k", type=int, default=8)
    f.add_argument("--fold-seed", type=int, default=0)
    s = sub.add_parser("summarize")
    s.add_argument("--work", required=True)
    s.add_argument("--out-prefix", required=True)
    a = ap.parse_args(argv)
    {"prepare": prepare, "fold": fold_shard, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
