"""TM-score, designability summary logic (CPU stages), adapter diagnostics + pre-flight."""
import io
import json
import os
import tarfile

import numpy as np
import pytest

import conftest as cf
import designability as dz
import eval_common as ec
import tm_score as tm


def helix_ca(n=60):
    return cf.helix_bundle_ca(n_helix=1, helix_len=n)


def test_tm_score_identity_rotation_noise_and_random():
    rng = np.random.RandomState(0)
    ca = np.concatenate([helix_ca(30), helix_ca(30)[::-1] + np.array([9.0, 0, 0])])
    Q, _ = np.linalg.qr(rng.randn(3, 3))
    P = ca @ Q + np.array([10.0, -5, 3])
    assert tm.tm_score(P, ca) == pytest.approx(1.0, abs=1e-6)
    assert tm.rmsd_superposed(P, ca) < 1e-6
    noisy = tm.tm_score(P + rng.randn(*P.shape) * 1.2, ca)
    assert 0.6 < noisy < 0.95
    assert tm.tm_score(np.cumsum(rng.randn(len(ca), 3) * 2.2, 0), ca) < 0.3
    assert tm.d0_of(100) == pytest.approx(1.24 * 85 ** (1 / 3) - 1.8)
    with pytest.raises(ValueError):
        tm.tm_score(ca[:-1], ca)


def test_tm_score_length_normalisation():
    ca = helix_ca(40)
    assert tm.tm_score(ca, ca, L_ref=80) == pytest.approx(0.5, abs=1e-6)       # 40 perfectly placed residues out of 80


def test_wilson_and_verdict_thresholds():
    p, lo, hi = dz.wilson(8, 10)
    assert p == 0.8 and lo < 0.8 < hi and 0.4 < lo < 0.55 and hi < 0.97
    assert np.isnan(dz.wilson(0, 0)[0])
    row = {"per_seq": [{"i": 0, "scrmsd": 1.5, "plddt": 75.0, "tm_pred_ref": .7}, {"i": 1, "scrmsd": 2.5, "plddt": 90.0}], "tm_design_ref": 0.62}
    v = dz.design_verdict(row)
    assert v["designable"] is False and v["designable70"] is True and v["on_fold"] and v["best_scrmsd"] == 1.5     # pLDDT 75 passes 70 but not 80
    row["per_seq"].append({"i": 2, "scrmsd": 1.2, "plddt": 85.0})
    v2 = dz.design_verdict(row)
    assert v2["designable"] and v2["designable_on_fold"] and v2["best_plddt"] == 85.0
    assert dz.design_verdict({"per_seq": [{"i": 0, "error": "x"}]})["n_seq"] == 0
    assert dz.design_verdict(dict(row, tm_design_ref=0.4))["designable_on_fold"] is False


def test_summarize_cells_and_markdown():
    rows = []
    for i in range(10):
        rows.append({"target": "t", "arm": "full", "cfg": 1.0, "per_seq": [{"i": 0, "scrmsd": 1.0 if i < 7 else 3.0, "plddt": 85.0}],
                     "tm_design_ref": 0.7 if i < 8 else 0.3})
    cells = dz.summarize(rows)
    c = cells["t|full|1.0"]
    # designs 0-6 are designable (scRMSD 1.0), designs 0-7 are on-fold (TM 0.7): joint = designs 0-6
    assert c["designable"]["k"] == 7 and c["on_fold"]["k"] == 8 and c["designable_on_fold"]["k"] == 7
    assert c["designable_on_fold"]["n"] == 10 and c["designable"]["lo"] < 0.7 < c["designable"]["hi"]
    md = dz.summary_md(cells, "x")
    assert "| t | full | 1.0 |" in md and "designable" in md


def test_read_designed_seqs_skips_native_record(tmp_path):
    fa = tmp_path / "a.fa"
    fa.write_text(">native, T=0.1\nAAAA\n>d1\nMKLV\n>d2\nGGGG\n>d3\nAC:DE\n")
    assert dz.read_designed_seqs(str(fa), 2) == ["MKLV", "GGGG"]
    assert dz.read_designed_seqs(str(fa), 8)[-1] == "ACDE"


def test_select_designs_and_prepare_without_mpnn(tmp_path):
    ca = np.array([[3.8 * i, 0, 0] for i in range(20)], float)
    blob = cf.to_cif_gz_bytes(cf.atoms_from_ca(ca))
    tar = tmp_path / "e.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for cfg in (1.0, 1.5, 2.0):
            for arm in ("full", "ss_only"):
                for b in range(3):
                    name = f"tag/designs/cfg{cfg:.1f}/tgt__{arm}_{b}_model_0.cif.gz"
                    ti = tarfile.TarInfo(name)
                    ti.size = len(blob)
                    tf.addfile(ti, io.BytesIO(blob))
                    ti2 = tarfile.TarInfo(name.replace(".cif.gz", ".json"))
                    ti2.size = 2
                    tf.addfile(ti2, io.BytesIO(b"{}"))
    sel = dz.select_designs(str(tar), [1.0, 1.5], ["full"], None, 2)
    assert len(sel) == 4 and {c["cfg"] for _, c in sel} == {1.0, 1.5} and {c["batch"] for _, c in sel} == {0, 1}
    args = dz.argparse.Namespace(tar=str(tar), specs_tar=None, work=str(tmp_path / "w"), cfgs=[1.0], arms=["full"], targets=None,
                                 max_per_cell=3, skip_mpnn=True, k=8, temperature=0.1, mpnn_seed=37, python="python")
    assert dz.prepare(args) == {"n": 3}
    idx = json.load(open(tmp_path / "w" / "index.json"))
    assert len(idx) == 3 and all(os.path.exists(c["pdb"]) and c["n_res"] == 20 for c in idx)
    assert dz.ca_from_pdb(idx[0]["pdb"]).shape == (20, 3)


torch = pytest.importorskip("torch")
import adapter_diag as ad  # noqa: E402


def _fake_ckpt(path, adj_rows=None, ss_rows=4, c=8, both=True):
    sd = {}
    g = torch.Generator().manual_seed(0)
    for src in (("shadow", "model") if both else ("shadow",)):
        if adj_rows:
            sd[f"{src}.{ad.ADJ_KEY}"] = torch.randn(adj_rows, c, generator=g)
            sd[f"{src}.{ad.ADJ_KEY}"][2] = 0
        sd[f"{src}.{ad.SS_KEY}"] = torch.randn(ss_rows, c, generator=g)
        sd[f"{src}.other.weight"] = torch.randn(3, 3, generator=g)
    torch.save({"model": sd, "epoch": 1}, path)


def test_table_stats_known_values():
    w = torch.zeros(5, 4)
    w[0] = torch.tensor([1.0, 0, 0, 0])
    w[1] = torch.tensor([0, 2.0, 0, 0])
    w[3] = torch.tensor([0, 2.0, 0, 0.5])
    w[4] = torch.tensor([0.5, 0, 0, 0])
    s = ad.table_stats(w)
    assert s["n_states"] == 5 and s["norms"][:2] == [1.0, 2.0] and s["norm_row2"] == 0.0
    assert s["contrast"] == pytest.approx(5 ** 0.5) and s["cos01"] == pytest.approx(0.0)
    assert s["cos_1_3"] == pytest.approx(2 * 2 / (2 * (4.25 ** 0.5))) and s["cos_0_4"] == pytest.approx(1.0)
    assert s["hot_gap"] == pytest.approx(0.5) and s["cold_gap"] == pytest.approx(0.5)
    assert np.isnan(ad.table_stats(torch.tensor([[1.0, 0], [0, 0]]))["cos01"])           # zero row -> undefined cosine


def test_diagnostics_read_table_size_from_checkpoint(tmp_path):
    for rows in (3, 5):
        p = str(tmp_path / f"c{rows}.ckpt")
        _fake_ckpt(p, adj_rows=rows)
        d = ad.adapter_diagnostics(p)
        assert d["has_adapters"] and d["shadow"]["adj"]["n_states"] == rows and d["model"]["ss"]["n_states"] == 4
        assert ("cos_1_3" in d["shadow"]["adj"]) == (rows == 5)
    p = str(tmp_path / "stock.ckpt")
    torch.save({"model": {"shadow.x.weight": torch.zeros(2, 2)}}, p)
    assert ad.adapter_diagnostics(p)["has_adapters"] is False


REPO = ec.REPO_DEFAULT


def test_preflight_refuses_table_size_mismatch(tmp_path):
    tree = ad.expected_rows(REPO)
    assert tree["ss"] == 4 and tree["adj"] >= 3
    ok = str(tmp_path / "ok.ckpt")
    _fake_ckpt(ok, adj_rows=tree["adj"], ss_rows=tree["ss"])
    rc, info = ad.preflight(ok, REPO)
    assert rc == 0 and info["ok"] and info["ckpt_rows"] == tree
    bad = str(tmp_path / "bad.ckpt")
    _fake_ckpt(bad, adj_rows=tree["adj"] + 2, ss_rows=tree["ss"])         # e.g. the phase-8 3-row table in the 5-row tree
    rc, info = ad.preflight(bad, REPO)
    assert rc == 2 and not info["ok"] and "silently" in info["message"]
    stock = str(tmp_path / "stock.ckpt")
    torch.save({"model": {"shadow.a.weight": torch.zeros(2, 2), "model.a.weight": torch.zeros(2, 2)}}, stock)
    rc, info = ad.preflight(stock, REPO)
    assert rc == 0 and info["has_adapters"] is False                       # stock: no adapter keys = zero-init adapters, expected
    rc, info = ad.preflight(str(tmp_path / "missing.ckpt"), REPO)
    assert rc == 3
