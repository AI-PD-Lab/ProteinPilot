"""Tests for summarize.py (stdlib unittest). The anatomy tests use the EXISTING N019 cache tags: seed 1 (v2_E2_e0596_h02r1cen2m0)
vs itself -> every delta exactly 0, every CI covers 0, hairpin cut 0; seed 2 (…_s20261015) vs seed 1 -> within the CI, no flag.
designs.csv files are copied from a populated cache (v2/eval/cache or this session's scratch cache) into a temp cache; if none has
them, score_all.sh scores the tars on demand (~3 min per tag). Also: guardrails vs the N019 record, designability pooling (71/120),
the Rossmann scorer on the N019 baseline tars (25.0 / 17.2 % recorded), derive_ckpt on a synthetic checkpoint, sacct parsing,
criteria evaluation on synthetic metrics, and the whole summarize() on a chain made of existing outputs (no GPU, no sacct).
run: /path/to/env/bin/python -m unittest test_summarize   (from this directory, ~2-3 min)"""
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("summarize", os.path.join(HERE, "summarize.py"))
sm = importlib.util.module_from_spec(spec)
sys.modules["summarize"] = sm
spec.loader.exec_module(sm)
W = sm.W
BASE = "v2_E2_e0596_h02r1cen2m0"
SEED2 = "v2_E2_e0596_h02r1cen2m0_s20261015"
CACHE_SOURCES = [W + "/v2/eval/cache",
                 "/path/to/harmonia_n017"]
TMP = tempfile.mkdtemp(prefix="evalchain_test_")
CACHE = os.path.join(TMP, "cache")
os.makedirs(CACHE)
for tag in (BASE, SEED2):
    for src in CACHE_SOURCES:
        p = f"{src}/{tag}.designs.csv"
        if os.path.exists(p):
            shutil.copy(p, CACHE)
            for ext in (".tm.csv", ".cells.csv"):
                if os.path.exists(f"{src}/{tag}{ext}"):
                    shutil.copy(f"{src}/{tag}{ext}", CACHE)
            break
HAVE_TARS = all(os.path.exists(f"{sm.EVAL_DIR}/{t}.tar.gz") for t in (BASE, SEED2))


@unittest.skipUnless(HAVE_TARS, "N019 baseline tars not in v2/eval")
class AnatomyOnCacheTags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T = sm.run_anatomy([SEED2], BASE, 1.5, CACHE, os.path.join(TMP, "anat.csv"), procs=16)
        assert cls.T is not None

    def test_seed1_vs_itself_all_deltas_zero(self):
        s = sm.sfar_stats(self.T, BASE, BASE)
        self.assertEqual(s["delta"], 0.0)
        self.assertLessEqual(s["delta_lo"], 0.0)
        self.assertGreaterEqual(s["delta_hi"], 0.0)
        self.assertFalse(s["sig"] or s["flag"])
        self.assertAlmostEqual(s["pass"], 0.7542, places=4)      # N019 seed 1 SFAR@1.5 (goal_metric parity)
        self.assertEqual(s["n"], 480)
        sw = sm.sandwich_stats(self.T, BASE, BASE)
        self.assertEqual(sw["hairpin12_cut"], 0.0)
        self.assertEqual(sw["delta"], 0.0)
        self.assertAlmostEqual(sw["p12"], sw["base_p12"])
        self.assertAlmostEqual(sw["p14"], sw["base_p14"])
        self.assertEqual(sw["n"], 96)
        for key, cell in self.T[BASE].items():            # every cell against itself: zero point delta
            d, lo, hi = sm.ec.diff_ci(cell["_pass_vec"], cell["_pass_vec"], n_boot=2000, seed=0)
            self.assertEqual(d, 0.0)
            self.assertTrue(lo <= 0 <= hi, key)

    def test_seed2_vs_seed1_within_ci(self):
        s = sm.sfar_stats(self.T, SEED2, BASE)
        self.assertTrue(s["delta_lo"] <= 0 <= s["delta_hi"], s)
        self.assertLess(abs(s["delta"]), 2 * sm.SD_RECIPE)      # README_score_anatomy: seed-2 delta -0.021, CI covers 0, no flag
        self.assertFalse(s["flag"])
        self.assertFalse(s["sig"])
        sw = sm.sandwich_stats(self.T, SEED2, BASE)
        self.assertTrue(np.isfinite(sw["hairpin12_cut"]))
        self.assertTrue(-1.0 < sw["hairpin12_cut"] < 1.0)

    def test_baseline_criteria_statuses(self):
        """the baseline evaluated against itself: SFAR passes its own threshold, the improvement criteria fail (by construction)"""
        M = {"sfar": {1.5: sm.sfar_stats(self.T, BASE, BASE)}, "sandwich": {1.5: sm.sandwich_stats(self.T, BASE, BASE)},
             "guardrails": sm.guardrails(CACHE, BASE, [1.5, 2.0]), "validity": {"eval": {"exists": True, "valid": True}}}
        st = {r["id"]: r["status"] for r in sm.evaluate(M)}
        self.assertEqual(st["sfar15"], "PASS")
        self.assertEqual(st["hairpin12_cut"], "FAIL")
        self.assertEqual(st["paired14"], "FAIL")          # baseline p14 0.594 < 0.70
        self.assertEqual(st["sandwich_pass"], "FAIL")     # baseline 0.417 < 0.50
        self.assertEqual(st["split"], "PASS")
        self.assertEqual(st["breaks"], "PASS")
        self.assertEqual(st["designability"], "UNVERIFIED")
        self.assertEqual(st["heldout"], "UNVERIFIED")
        self.assertEqual(st["sfar15_3seed"], "SKIPPED")
        self.assertEqual(sm.verdict(sm.evaluate(M)), "KILL")


class Guardrails(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(f"{CACHE}/{BASE}.designs.csv"), "baseline designs.csv not available")
    def test_baseline_split_breaks_match_n019_record(self):
        G = sm.guardrails(CACHE, BASE, [1.5, 2.0])
        self.assertAlmostEqual(G[(1.5, "ALL")]["split"], 3 / 480)      # N019: split @1.5 0.6 %
        self.assertAlmostEqual(G[(2.0, "ALL")]["split"], 13 / 480)     # @2.0 2.7 %
        self.assertEqual(G[(1.5, "ALL")]["breaks"], 0.0)
        self.assertLessEqual(max(v["split"] for (c, a), v in G.items() if a != "ALL"), 0.05)
        self.assertEqual(G[(1.5, "full")]["n"], 160)

    def test_designability_pool_n019(self):
        p = sm.CRIT["baseline"]["designability_json"]
        if not os.path.exists(p):
            p = sorted(glob.glob(sm.RESULTS + "/N019-*/v2_E2_e0596_h02r1cen2m0_cfg2.0.designability.json"))[0]
        D = sm.designability_pool(p)
        self.assertEqual((D[2.0]["k"], D[2.0]["n"]), (71, 120))      # N019: 71 = 59.2 %
        self.assertAlmostEqual(D[2.0]["p"], 71 / 120)
        self.assertTrue(D[2.0]["lo"] < D[2.0]["p"] < D[2.0]["hi"])

    def test_wilson(self):
        p, lo, hi = sm.wilson(0, 10)
        self.assertEqual(p, 0.0)
        self.assertTrue(abs(lo) < 1e-9 and 0.2 < hi < 0.35)
        self.assertTrue(all(np.isnan(sm.wilson(0, 0))))


class Rossmann(unittest.TestCase):
    @unittest.skipUnless(all(os.path.exists(p) for p in sm.CRIT["baseline"]["ross_tars"]), "N019 Rossmann baseline tars missing")
    def test_baseline_ross_split_matches_record(self):
        sc = {}
        for p in sm.CRIT["baseline"]["ross_tars"]:
            s = sm.ross_scores(p, procs=16)
            sc.update({c: v for c, v in s.items() if not c.startswith("_")})
        full = {c: v["full_rossmann_like"]["split_pct"] for c, v in sc.items()}
        self.assertEqual(sorted(full), ["v2E2h02r1cen2m0s1_cfg2.0", "v2E2h02r1cen2m0s2_cfg2.0"])
        self.assertAlmostEqual(full["v2E2h02r1cen2m0s1_cfg2.0"], 25.0, places=1)    # ross_cen2m0_vs_h02r1_E2_p0694.txt
        self.assertAlmostEqual(full["v2E2h02r1cen2m0s2_cfg2.0"], 17.2, places=1)
        self.assertAlmostEqual(sm.ross_case_mean(sc, "full_rossmann_like"), 21.1, places=1)
        self.assertEqual(sc["v2E2h02r1cen2m0s1_cfg2.0"]["adjonly_rossmann_like"]["n"], 64)


class DeriveCkpt(unittest.TestCase):
    def test_raw_mode_copies_model_over_shadow(self):
        import torch
        d = tempfile.mkdtemp(dir=TMP)
        sd = {"model.a.weight": torch.randn(3, 2), "shadow.a.weight": torch.randn(3, 2), "model.b": torch.zeros(2), "shadow.b": torch.ones(2)}
        torch.save({"model": sd, "train_cfg": {"x": 1}, "epoch": 596}, f"{d}/in.ckpt")
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "derive_ckpt.py"), "--in", f"{d}/in.ckpt", "--out", f"{d}/out.ckpt", "--mode", "raw"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        ck = torch.load(f"{d}/out.ckpt", weights_only=False)
        self.assertTrue(torch.equal(ck["model"]["shadow.a.weight"], sd["model.a.weight"]))
        self.assertTrue(torch.equal(ck["model"]["shadow.b"], torch.zeros(2)))
        self.assertEqual(ck["train_cfg"], {"x": 1}) and self.assertEqual(ck["epoch"], 596)
        r2 = subprocess.run([sys.executable, "-I", os.path.join(HERE, "derive_ckpt.py"), "--in", f"{d}/in.ckpt", "--mode", "ema"], capture_output=True, text=True)
        self.assertEqual(r2.returncode, 0)
        self.assertIn('"n_pairs_differ": 2', r2.stdout)
        torch.save({"model": {"model.a": torch.zeros(1)}}, f"{d}/noema.ckpt")
        r3 = subprocess.run([sys.executable, "-I", os.path.join(HERE, "derive_ckpt.py"), "--in", f"{d}/noema.ckpt", "--out", f"{d}/x.ckpt", "--mode", "raw"], capture_output=True, text=True)
        self.assertEqual(r3.returncode, 2)


class Criteria(unittest.TestCase):
    def test_sacct_parse(self):
        txt = "1214841|ds_E2|464|billing=64,cpu=64,gres/gpu:h100=2,gres/gpu=2,mem=350G,node=1|COMPLETED|preempt\n9|x|100|cpu=16|COMPLETED|preempt_cpu\n"
        a = sm.parse_sacct(txt)
        self.assertAlmostEqual(a["1214841"]["gpuh"], 2 * 464 / 3600)
        self.assertEqual(a["9"]["gpus"], 0)

    def test_evaluate_synthetic(self):
        M = {"validity": {"eval": {"exists": True, "valid": True}, "ross": {"exists": True, "valid": True}},
             "sfar": {1.5: dict(n=480, **{"pass": 0.80}, lo=0.76, hi=0.84, base_pass=0.754, delta=0.046, delta_lo=0.0, delta_hi=0.09, sig=True, rec_lt09=0.1, sep_gt115=0.1, breaks=0.0, clash_gt4=0.01, pass_native=0.5)},
             "sandwich": {1.5: dict(n=96, **{"pass": 0.56}, pass_lo=0.46, pass_hi=0.66, p12=0.10, L12=0.4, p14=0.75, L14=2.8, p05=0.7, base_pass=0.417, base_p12=0.292, base_p14=0.594, base_L12=1.2, hairpin12_cut=1 - 0.10 / 0.292)},
             "guardrails": {(1.5, "full"): dict(split=0.01, breaks=0.0), (1.5, "ALL"): dict(split=0.01, breaks=0.0), (2.0, "sparse8"): dict(split=0.06, breaks=0.0), (2.0, "ALL"): dict(split=0.03, breaks=0.004)},
             "designability": {2.0: dict(k=70, n=120, p=70 / 120, lo=0.5, hi=0.67, n_cells=15, joint_p=0.2, p70=0.7)},
             "heldout": dict(**{"pass": 0.40}, n=192, base_pass=0.45, base_n=192, delta=-0.05, delta_lo=-0.12, delta_hi=0.02, sig=False),
             "ross": dict(full_split_pct=28.0, cells=2, adjonly_split_pct=4.0, ssonly_split_pct=30.0, base_full_split_pct=21.1)}
        rows = sm.evaluate(M)
        st = {r["id"]: r["status"] for r in rows}
        self.assertEqual(st["valid"], "PASS")
        self.assertEqual(st["sfar15"], "PASS")
        self.assertEqual(st["sandwich_pass"], "PASS")
        self.assertEqual(st["sandwich_target"], "PASS")
        self.assertEqual(st["hairpin12_cut"], "PASS")
        self.assertEqual(st["paired14"], "PASS")
        self.assertEqual(st["split"], "FAIL")              # sparse8@2.0 = 6 % > 5 %
        self.assertEqual(st["breaks"], "PASS")
        self.assertEqual(st["designability"], "PASS")
        self.assertEqual(st["heldout"], "PASS")            # -0.05 but CI covers 0 -> not significantly worse
        self.assertEqual(st["ross"], "PASS")
        self.assertEqual(sm.verdict(rows), "KILL")
        M["guardrails"][(2.0, "sparse8")]["split"] = 0.02
        M["heldout"].update(delta=-0.08, delta_lo=-0.14, delta_hi=-0.02, sig=True)
        self.assertEqual({r["id"]: r["status"] for r in sm.evaluate(M)}["heldout"], "FAIL")
        M["heldout"].update(delta=0.01, delta_lo=-0.05, delta_hi=0.07, sig=False)
        self.assertEqual(sm.verdict(sm.evaluate(M)), "PASS")
        M["validity"]["ross"] = {"exists": False, "valid": None, "job_state": "FAILED"}
        self.assertEqual({r["id"]: r["status"] for r in sm.evaluate(M)}["valid"], "UNVERIFIED")
        self.assertEqual(sm.verdict(sm.evaluate(M)), "INCOMPLETE")


@unittest.skipUnless(HAVE_TARS, "N019 baseline tars not in v2/eval")
class WholeSummarize(unittest.TestCase):
    def test_summarize_on_existing_outputs(self):
        """whole CPU step on a chain assembled from existing outputs: seed-2 eval tar as the candidate, the N019 Rossmann tar and
        designability json as its ross/design stages, a missing held-out tar -> report written, held-out UNVERIFIED, verdict KILL"""
        out = os.path.join(TMP, "res")
        chain = dict(tag=SEED2, item="N019", out_dir=out, ckpt="/dev/null", weights="ema", seed=20261015, cfgs="1.5 2.0", n=32, recipe="n019",
                     extra="", cache=CACHE, eval_dir=sm.EVAL_DIR, estimate=dict(expected_gpuh=4.0, cap_gpuh=9.0), submitted="test",
                     stages=dict(eval=dict(job="1214842", tag=SEED2), heldout=dict(job="0", tag="does_not_exist_heldout"),
                                 ross=dict(job="1214844", tar=sm.CRIT["baseline"]["ross_tars"][0]),
                                 design=dict(job="1214841", json=sm.CRIT["baseline"]["designability_json"]), summarize=dict(job="0")))
        res = sm.summarize(chain, procs=16, no_sacct=True)
        self.assertTrue(os.path.exists(os.path.join(out, "summary.md")) and os.path.exists(os.path.join(out, "summary.json")))
        st = {r["id"]: r["status"] for r in res["criteria"]}
        self.assertEqual(st["heldout"], "UNVERIFIED")
        self.assertEqual(st["valid"], "UNVERIFIED")        # the held-out output is missing
        self.assertEqual(st["sfar15"], "FAIL")             # seed 2: 0.733 < 0.754
        self.assertEqual(st["designability"], "PASS")      # 59.2 % >= 55 %
        self.assertEqual(st["ross"], "PASS")               # 25.0 % <= 30 %
        self.assertEqual(res["verdict"], "KILL")
        j = json.load(open(os.path.join(out, "summary.json")))
        self.assertEqual(j["verdict"], "KILL")
        self.assertAlmostEqual(j["metrics"]["sfar"]["1.5"]["pass"], 0.733, places=2)
        self.assertTrue(os.path.exists(os.path.join(out, "anatomy_cfg1.5.csv")) and os.path.exists(os.path.join(out, "guardrails.csv")))


if __name__ == "__main__":
    unittest.main()
