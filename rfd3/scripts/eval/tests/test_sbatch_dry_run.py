"""Whole-pipeline dry run of scripts/slurm/eval_checkpoint.sbatch on CPU: DRY=1 swaps rfd3 for fake_run_inference.py. Checks the
bash plumbing (pre-flight, plan, one worker per 'GPU', status/validity logic, meta.json, ONE tar per checkpoint, stock run)
and that the produced tar is scorable. No env restore, no GPU, no real checkpoint."""
import json
import os
import subprocess
import sys
import tarfile

import pytest

import build_specs as bs
import conftest as cf
import eval_common as ec
import score_designs as sd

ES = "/path/to/env/bin/python"
SBATCH = os.path.join(ec.REPO_DEFAULT, "scripts", "slurm", "eval_checkpoint.sbatch")
FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_run_inference.py")
torch = pytest.importorskip("torch")
import adapter_diag as ad  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(ES), reason="esmfold2 env python not available")


def _pack(tmp_path):
    import biotite.structure.io.pdb as pdb

    f = pdb.PDBFile()
    f.set_structure(cf.atoms_from_ca(cf.helix_bundle_ca()))
    p = str(tmp_path / "syn.pdb")
    f.write(p)
    ref = dict(name="bundle_syn", pdb="syn", chain="A", fold="synthetic")
    w = str(tmp_path / "specs_w")
    bs.build(w, refs=[ref], seed=1, ref_files={"bundle_syn": p}, uncond_lengths=(30,))
    tar = str(tmp_path / "out" / "specs_v2.tar.gz")
    bs.pack_dir(w, tar)
    return tar


def _ckpt(path, adj_rows):
    sd_ = {}
    for src in ("shadow", "model"):
        if adj_rows:
            sd_[f"{src}.{ad.ADJ_KEY}"] = torch.randn(adj_rows, 8)
            sd_[f"{src}.{ad.SS_KEY}"] = torch.randn(4, 8)
        sd_[f"{src}.x.weight"] = torch.zeros(2, 2)
    torch.save({"model": sd_}, path)


def _run(tmp_path, extra_env=None, ckpt_rows=None):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    specs = _pack(tmp_path) if not (out / "specs_v2.tar.gz").exists() else str(out / "specs_v2.tar.gz")
    tree = ad.expected_rows(ec.REPO_DEFAULT)
    ck, st = str(tmp_path / "ck.ckpt"), str(tmp_path / "stock.ckpt")
    _ckpt(ck, ckpt_rows or tree["adj"])
    _ckpt(st, None)
    env = dict(os.environ, CKPT=ck, TAG="tagA", SPECS=specs, OUT_DIR=str(out), TMPBASE=str(tmp_path), DRY="1", DRY_PY=sys.executable,
               DRY_INFER=FAKE, N="4", BATCH="2", CFGS="1.0 2.0", NGPU="2", STOCK="1", STOCK_CKPT=st, REPO=ec.REPO_DEFAULT)
    env.update(extra_env or {})
    r = subprocess.run(["bash", SBATCH], env=env, capture_output=True, text=True, timeout=300)
    return r, out, specs


def test_dry_run_produces_one_valid_scorable_tar_per_checkpoint(tmp_path):
    r, out, specs = _run(tmp_path)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    for tag in ("tagA", "stock"):
        tar = out / f"{tag}.tar.gz"
        assert tar.exists()
        with tarfile.open(tar) as tf:
            names = tf.getnames()
            meta = json.load(tf.extractfile(f"{tag}/meta.json"))
        assert {f"{tag}/meta.json", f"{tag}/plan.json", f"{tag}/adapter_diag.json"} <= set(names)
        assert meta["valid"] is True and meta["seed"] == 20261014 and meta["n_per_key"] == 4 and meta["cfgs"] == ["1.0", "2.0"]
        assert meta["n_designs_found"] == meta["n_designs_expected"] > 0 and meta["is_stock"] == (tag == "stock")
        assert len(meta["specs_sha256"]) == 64
        # 4 keys x 4 designs x 2 CFGs + uncond 4 designs (CFG 1.0 only)
        assert sum(1 for n in names if n.endswith(".cif.gz")) == 4 * 4 * 2 + 4
        assert not any(n.startswith(f"{tag}/designs/cfg2.0/uncond") for n in names)
        plan = json.load(tarfile.open(tar).extractfile(f"{tag}/plan.json"))
        assert {s["gpu"] for s in plan} == {0, 1} and all(isinstance(s["seed"], int) for s in plan)
    assert "score with" in r.stdout and not list(out.glob(".*part"))
    # scorable
    man = bs.read_manifest_from_tar(specs)
    rows = sd.score_tar(str(out / "tagA.tar.gz"), man)
    assert len(rows) == 36 and not [x for x in rows if x["error"]]
    cells = sd.aggregate(rows, boot=50)
    assert cells["bundle_syn|full|2.0"]["n"] == 4 and cells["uncond_L030|unconditional|1.0"]["n"] == 4
    assert not os.path.exists(tmp_path / "harmonia_eval_local")      # node-local staging was removed


def test_dry_run_flags_failed_shard_but_still_writes_tar(tmp_path):
    r, out, _ = _run(tmp_path, {"FAKE_FAIL": "cfg2.0__bundle_syn", "STOCK": "0"})
    assert r.returncode == 3, r.stdout[-2000:]
    meta = json.load(tarfile.open(out / "tagA.tar.gz").extractfile("tagA/meta.json"))
    assert meta["valid"] is False and "non-zero" in meta["note"]


def test_preflight_mismatch_is_refused_before_any_inference(tmp_path):
    tree = ad.expected_rows(ec.REPO_DEFAULT)
    r, out, _ = _run(tmp_path, {"STOCK": "0"}, ckpt_rows=tree["adj"] + 2)
    assert r.returncode == 2 and "FATAL: pre-flight" in r.stdout and not (out / "tagA.tar.gz").exists()
    r2, out2, _ = _run(tmp_path, {"STOCK": "0", "ALLOW_MISMATCH": "1"}, ckpt_rows=tree["adj"] + 2)
    assert r2.returncode == 3 and json.load(tarfile.open(out2 / "tagA.tar.gz").extractfile("tagA/meta.json"))["valid"] is False
