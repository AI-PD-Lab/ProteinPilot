"""Scorer tests: synthetic structures with KNOWN half_sep / recovery / violation / extent, tar reading, aggregation,
compare logic, and a reproduction of the legacy N002 tables (skipped when those tars are not on this machine)."""
import io
import json
import os
import tarfile

import numpy as np
import pytest

import conftest as cf
import design_metrics as dm
import eval_common as ec
import score_designs as sd

TARGET = {"name": "two_strands", "n_res": 22, "n_blocks": 2,
          "blocks": [{"id": 0, "type": "E", "start": 0, "stop": 10}, {"id": 1, "type": "E", "start": 12, "stop": 22}],
          "half1_blocks": [0], "half2_blocks": [1], "pair_n6": {"0-1": 10}, "native": {}}
ARM_POS = {"target": "two_strands", "arm": "pos", "key": "two_strands__pos", "length": 22, "ss": None, "has_adj": True,
           "pos_pairs": [[0, 1]], "neg_pairs": []}
ARM_NEG = dict(ARM_POS, arm="neg", key="two_strands__neg", pos_pairs=[], neg_pairs=[[0, 1]])


def chain_for(d):
    ca, _, _ = cf.two_strands_ca(d)
    return ec.protein_chains(cf.atoms_from_ca(ca))[0]


def test_reconstructed_cb_is_fixed_offset():
    ch = chain_for(5.0)
    assert np.allclose(ch.cb - ch.ca, cf.CB_V, atol=1e-9)
    assert np.allclose(ch.cb_leg - ch.ca, cf.CB_V, atol=1e-5)   # real CB written at the same offset


@pytest.mark.parametrize("d,rec,viol,n6", [(5.0, 1.0, 1.0, 10), (8.0, 0.0, 0.0, 0)])
def test_conditioned_metrics_known_values(d, rec, viol, n6):
    ch = chain_for(d)
    sse = ec.sse_labels(ch.ca)
    m = dm.conditioned_metrics(ch.cb, sse, TARGET, ARM_POS, ch.cb_leg)
    assert m["recovery"] == rec
    assert m["half_sep"] == pytest.approx(d, abs=1e-4)          # centroids of the two strands are exactly d apart
    assert m["extent_n6"] == n6 and m["extent_ratio"] == pytest.approx(n6 / 10)
    assert m["pos_min_d"] == pytest.approx(d, abs=1e-4)
    assert np.isnan(m["violation"])
    m2 = dm.conditioned_metrics(ch.cb, sse, TARGET, ARM_NEG, ch.cb_leg)
    assert m2["violation"] == viol and np.isnan(m2["recovery"])
    assert m2["ends_n6"] == n6                                  # first vs last block = the two strands


def test_half_spans_include_loops_and_reproduce_ferredoxin_spans():
    # synthetic ferredoxin B1 A1 B2 | B3 A2 B4, 1-based spans [3,8] [12,22] [27,32] [37,42] [46,54] [59,63]
    spans = [(3, 8), (12, 22), (27, 32), (37, 42), (46, 54), (59, 63)]
    blocks = [{"id": i, "type": "E", "start": a - 1, "stop": b} for i, (a, b) in enumerate(spans)]
    t = {"blocks": blocks, "half1_blocks": [0, 1, 2], "half2_blocks": [3, 4, 5]}
    (a0, a1), (b0, b1) = dm.half_spans(t)
    assert (a0 + 1, a1) == (3, 32) and (b0 + 1, b1) == (37, 63)     # project spans [3,32] and [37,63]


def test_geometry_metrics_basic_counts():
    n = 30
    ca = np.array([[3.8 * i, 0, 0] for i in range(n)], float)
    ca[15:] += np.array([0.0, 7.0, 0.0])                            # one chain break (7 A jump)
    g = dm.geometry_metrics(ca, ca, None, None, np.full(n, 2))
    assert g["N"] == n and g["ca_breaks"] == 1
    cen = ca.mean(0)
    assert g["rg"] == pytest.approx(np.sqrt(((ca - cen) ** 2).sum(1).mean()))
    assert g["rg_n04"] == pytest.approx(g["rg"] / n ** 0.4)
    assert g["coil"] == 1.0 and g["helix"] == 0.0
    assert g["ca_clash"] == 0
    # two CAs 2 A apart 5 residues apart -> one CA clash and (with heavy atoms) one heavy clash
    ca2 = ca.copy()
    ca2[20] = ca2[10] + np.array([0.0, 0.0, 2.0])
    heavy = np.concatenate([ca2, ca2[[20]] + 0.5])
    hres = np.concatenate([np.arange(n), [20]])
    g2 = dm.geometry_metrics(ca2, ca2, heavy, hres, np.full(n, 2))
    assert g2["ca_clash"] >= 1 and g2["heavy_clash"] >= 1


def test_long_range_density_counts_pairs():
    n = 20
    cb = np.array([[0.0, 0, 0]] + [[100.0 + 10 * i, 0, 0] for i in range(1, n - 1)] + [[5.0, 0, 0]])
    g = dm.geometry_metrics(cb, cb, None, None, np.full(n, 2))
    assert g["lr_cb"] == pytest.approx(1 / n)                        # exactly one pair (0, 19): |i-j|=19>12 and 5 A < 8 A


def test_design_row_roundtrip_through_cif_gz_and_length_check():
    ca, _, _ = cf.two_strands_ca(5.0)
    atoms = cf.atoms_from_ca(ca)
    blob = cf.to_cif_gz_bytes(atoms)
    ch = ec.protein_chains(ec.read_structure(blob, "x_0_model_0.cif.gz"))[0]
    assert len(ch) == 22 and np.allclose(ch.ca, ca, atol=1e-3)
    row = dm.design_row(ch, TARGET, ARM_POS)
    assert row["len_ok"] and row["recovery"] == 1.0 and row["half_sep"] == pytest.approx(5.0, abs=1e-2)
    bad = dm.design_row(ch, dict(TARGET, n_res=30), ARM_POS)
    assert bad["len_ok"] is False and "recovery" not in bad


def _write_eval_tar(path, tag, items, layout="designs"):
    """items: [(cfg, key, batch, model, bytes)]"""
    with tarfile.open(path, "w:gz") as tf:
        for cfg, key, b, m, blob in items:
            name = (f"{tag}/designs/cfg{cfg:.1f}/{key}_{b}_model_{m}.cif.gz" if layout == "designs"
                    else f"{tag}_cfg{cfg:.1f}/{key}_{b}_model_{m}.cif.gz")
            ti = tarfile.TarInfo(name)
            ti.size = len(blob)
            tf.addfile(ti, io.BytesIO(blob))


@pytest.fixture
def manifest():
    return {"spec_version": 2, "dialect": "test", "seed": 0, "k_requested": 8,
            "targets": {"two_strands": TARGET}, "arms": {ARM_POS["key"]: ARM_POS, ARM_NEG["key"]: ARM_NEG},
            "unconditional": {"uncond_L022__unconditional": {"target": "uncond_L022", "arm": "unconditional",
                                                              "key": "uncond_L022__unconditional", "length": 22}}}


def _items():
    items = []
    for cfg, ds in ((1.0, [4.5, 5.0, 5.5, 5.0]), (2.0, [7.0, 8.0, 9.0, 8.0])):
        for i, d in enumerate(ds):
            ca, _, _ = cf.two_strands_ca(d)
            blob = cf.to_cif_gz_bytes(cf.atoms_from_ca(ca))
            items.append((cfg, "two_strands__pos", i, 0, blob))
            items.append((cfg, "two_strands__neg", i, 0, blob))
    ca, _, _ = cf.two_strands_ca(5.0)
    items.append((1.0, "uncond_L022__unconditional", 0, 0, cf.to_cif_gz_bytes(cf.atoms_from_ca(ca))))
    items.append((1.0, "two_strands__pos", 99, 0, b"not a structure"))           # unreadable -> error row, excluded
    items.append((1.0, "mystery__arm", 0, 0, cf.to_cif_gz_bytes(cf.atoms_from_ca(ca))))   # unknown key
    return items


@pytest.mark.parametrize("procs", [1, 2])
def test_score_tar_and_aggregate(tmp_path, manifest, procs):
    tar = tmp_path / "t.tar.gz"
    _write_eval_tar(str(tar), "t", _items())
    rows = sd.score_tar(str(tar), manifest, procs=procs)
    errs = [r for r in rows if r["error"]]
    assert {r["error"].split(":")[0] for r in errs} >= {"key not in manifest"} and len(errs) == 2
    cells = sd.aggregate(rows, boot=200)
    c1 = cells["two_strands|pos|1.0"]
    assert c1["n"] == 5 and c1["n_err"] == 1                                  # the unreadable file is counted, not scored
    assert c1["metrics"]["recovery"]["mean"] == 1.0
    assert c1["metrics"]["half_sep"]["mean"] == pytest.approx(np.mean([4.5, 5.0, 5.5, 5.0]), abs=1e-3)
    c2 = cells["two_strands|pos|2.0"]
    assert c2["metrics"]["recovery"]["mean"] == 0.0 and c2["metrics"]["half_sep"]["median"] == pytest.approx(8.0, abs=1e-3)
    assert cells["two_strands|neg|1.0"]["metrics"]["violation"]["mean"] == 1.0
    assert cells["two_strands|neg|2.0"]["metrics"]["violation"]["mean"] == 0.0
    u = cells["uncond_L022|unconditional|1.0"]
    assert "rg_n04" in u["metrics"] and "recovery" not in u["metrics"]
    lo, hi = c1["metrics"]["half_sep"]["lo"], c1["metrics"]["half_sep"]["hi"]
    assert lo <= c1["metrics"]["half_sep"]["mean"] <= hi


def test_flat_layout_reads_cfg_from_dir_name(tmp_path, manifest):
    ca, _, _ = cf.two_strands_ca(5.0)
    tar = tmp_path / "legacy.tar.gz"
    _write_eval_tar(str(tar), "p0694_allmask", [(2.0, "two_strands__pos", 0, 0, cf.to_cif_gz_bytes(cf.atoms_from_ca(ca)))], layout="flat")
    assert sd.score_tar(str(tar), manifest, layout="designs") == []
    rows = sd.score_tar(str(tar), manifest, layout="flat")
    assert len(rows) == 1 and rows[0]["cfg"] == 2.0 and rows[0]["recovery"] == 1.0


def test_summary_and_cli_outputs(tmp_path, manifest):
    tar = tmp_path / "tagx.tar.gz"
    _write_eval_tar(str(tar), "tagx", _items())
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    out = tmp_path / "out"
    sd.main(["score", "--tar", str(tar), "--out", str(out), "--manifest", str(mpath), "--procs", "1"])
    for ext in ("designs.csv", "cells.csv", "scores.json", "summary.md"):
        assert (out / f"tagx.{ext}").exists()
    md = (out / "tagx.summary.md").read_text()
    assert "Conditioned arms" in md and "CFG effect" in md and "Unconditional panel" in md
    sc = json.loads((out / "tagx.scores.json").read_text())
    assert "two_strands|pos|1.0" in sc["cells"] and sc["n_designs"] >= 10


def test_bootstrap_and_diff_ci():
    x = np.arange(100, dtype=float)
    m, lo, hi = ec.bootstrap_ci(x, n_boot=500, seed=3)
    assert m == pytest.approx(49.5) and lo < m < hi and hi - lo < 15
    d, dlo, dhi = ec.diff_ci(x + 10, x, n_boot=500, seed=3)
    assert d == pytest.approx(10.0) and dlo < 10 < dhi
    assert all(np.isnan(v) for v in ec.bootstrap_ci([np.nan]))


def _scores(tag, half_seps_by_cell, uncond=None):
    cells = {}
    for (arm, cfg), vals in half_seps_by_cell.items():
        cells[f"t|{arm}|{cfg:.1f}"] = {"target": "t", "arm": arm, "cfg": cfg, "n": len(vals), "n_err": 0, "n_len_bad": 0,
                                       "metrics": {"half_sep": {"mean": float(np.mean(vals)), "lo": 0, "hi": 0, "median": float(np.median(vals)), "n": len(vals)}},
                                       "values": {"half_sep": list(vals)}}
    return {"meta": {"tag": tag}, "cells": cells, "adapter": {"shadow": {"adj": {"n_states": 5, "norms": [1, 2, 0, 0, 0], "contrast": 2.2, "cos01": -0.1}}}}


def test_compare_flags_noise_worse_and_kill():
    rng = np.random.RandomState(0)
    stock = _scores("stock", {("full", 1.0): rng.normal(9, 1, 32), ("full", 2.0): rng.normal(9, 1, 32)})
    ck = _scores("ck", {("full", 1.0): rng.normal(9.5, 1, 32), ("full", 2.0): rng.normal(14, 1, 32)})
    md = sd.compare_md({"ckpt": ck, "stock": stock, "baseline": None})
    lines = {l.split("|")[3].strip(): l for l in md.splitlines() if l.startswith("| t |")}
    assert "~noise" in lines["1.0"] and "KILL" not in lines["1.0"]            # +0.5 A is inside the 2 A noise floor
    assert "WORSE" in lines["2.0"]                                            # +5 A with CI > 0
    ck2 = _scores("ck2", {("full", 1.0): rng.normal(11.5, 1, 32)})
    md2 = sd.compare_md({"ckpt": ck2, "stock": stock, "baseline": None})
    assert "KILL flags: t/full" in md2                                         # CFG-off excess > 1.5 A with CI > 0


LEGACY_TAR = "/path/to/work_rfd3/n002_smoke/fd_p0716_dense0_cfg2.0.tar.gz"


@pytest.mark.skipif(not os.path.exists(LEGACY_TAR) or not os.path.isdir("/path/to/checkpoints/p6_benchmark"),
                    reason="legacy N002 probe tars not available")
def test_reproduces_legacy_n002_medians():
    import build_specs as bs

    man = bs.legacy_ferredoxin_manifest()
    cell = list(sd.aggregate(sd.score_tar(LEGACY_TAR, man, procs=1, layout="flat"), boot=100).values())[0]
    m = cell["metrics"]
    # N002/cfg_probe_scores.txt: p0716_dense0_cfg2.0  half_sep=16.62 b1b4_n6=1 Rg=11.93 (medians, n=32)
    assert cell["n"] == 32 and m["half_sep"]["median"] == pytest.approx(16.62, abs=0.006)
    assert round(m["ends_n6"]["median"]) == 1 and m["rg"]["median"] == pytest.approx(11.93, abs=0.006)
