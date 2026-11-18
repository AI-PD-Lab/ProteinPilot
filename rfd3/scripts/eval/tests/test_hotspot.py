"""Hotspot specificity: epitope definition, spec strings, complex scoring, swap matrix, matched-decoy AUC (synthetic complexes)."""
import numpy as np
import pytest

import conftest as cf
import eval_common as ec
import hotspot_eval as he

N_RES = 60


def long_helix():
    """One 60-residue ideal helix (~90 A long): two far-apart surface patches exist."""
    atoms = cf.atoms_from_ca(cf.helix_bundle_ca(n_helix=1, helix_len=N_RES))
    return atoms, ec.protein_chains(atoms)[0]


def test_define_epitopes_disjoint_and_separated():
    atoms, ch = long_helix()
    rasa = he.residue_rasa(atoms)
    assert rasa.shape == (N_RES,) and np.all(rasa >= 0) and rasa.mean() > 0.25    # a lone helix is mostly exposed
    epi = he.define_epitopes(ch, rasa, None, n_epi=6, min_sep=12.0)
    A, B = set(epi["A"]), set(epi["B"])
    assert not (A & B) and len(A) == 6 and len(B) >= 4
    assert epi["min_sep"] >= 12.0
    d = np.linalg.norm(ch.cb_leg[sorted(A)][:, None] - ch.cb_leg[sorted(B)][None], axis=-1).min()
    assert d == pytest.approx(epi["min_sep"])


def test_epitope_A_follows_the_native_partner():
    atoms, ch = long_helix()
    rasa = he.residue_rasa(atoms)
    axis_xy = np.array([0.0, 0.0])
    patch = ch.cb_leg[8:16]
    out = patch.copy()
    out[:, :2] += (patch[:, :2] - axis_xy) / np.linalg.norm(patch[:, :2] - axis_xy, axis=1, keepdims=True) * 4.0   # 4 A above the surface
    epi = he.define_epitopes(ch, rasa, out, n_epi=6, min_sep=12.0)
    assert set(epi["A"]) <= set(range(5, 20)), epi["A"]
    contact, _ = he.contacted_residues(ch, out)
    assert contact[8:16].sum() >= 4


def test_define_epitopes_raises_when_no_disjoint_patch():
    atoms, ch = long_helix()
    with pytest.raises(ValueError):
        he.define_epitopes(ch, he.residue_rasa(atoms), None, n_epi=6, min_sep=500.0)


def test_spec_strings():
    assert he._ranges([3, 4, 5, 9, 11, 12]) == "A3-5,A9,A11-12"
    epi = {"A": [2, 3, 4, 10], "B": [30, 31, 40], "min_sep": 15.0}
    inputs, recs = he.make_arms("hot_x", 60, 70, epi, gly={10})   # 0-based residue 10 = A11 is Gly -> CA
    assert set(inputs) == {f"hot_x__{a}" for a in ("none", "hotA_1d", "hotB_1d", "hotA_adj", "hotB_adj", "hotA_adj_coldB", "hotB_adj_coldA")}
    assert inputs["hot_x__hotA_1d"]["select_hotspots"] == {"A3": "CB", "A4": "CB", "A5": "CB", "A11": "CA"}
    assert inputs["hot_x__hotA_adj"]["adj_pairs"] == ["A3-5,A11 :: B1-70"]
    assert inputs["hot_x__hotB_adj_coldA"]["adj_pairs"] == ["A31-32,A41 :: B1-70", "A3-5,A11 !!: B1-70"]
    assert inputs["hot_x__none"]["contig"] == "A1-60,/0,70-70" and "select_hotspots" not in inputs["hot_x__none"]
    assert {v["infer_ori_strategy"] for v in inputs.values()} == {"com"}        # like with like (rfd3 asserts hotspots needs select_hotspots)
    stock, _ = he.make_arms("hot_x", 60, 70, epi, gly=set(), ori_1d="hotspots")
    assert stock["hot_x__hotA_1d"]["infer_ori_strategy"] == "hotspots" and stock["hot_x__hotA_adj"]["infer_ori_strategy"] == "com"
    assert recs["hot_x__hotB_adj_coldA"]["hot"] == "B" and recs["hot_x__hotB_adj_coldA"]["cold"] == "A" and recs["hot_x__none"]["mode"] is None


def tinfo_for(ch, rasa, epi):
    return {"n_res": len(ch), "epitope_A": epi["A"], "epitope_B": epi["B"], "surface": [int(i) for i in np.nonzero(rasa >= 0.25)[0]],
            "rasa": rasa.tolist(), "cb": ch.cb_leg.tolist()}


def binder_near(ch, idx, offset=3.4, n=12):
    """A small binder chain B whose atoms sit `offset` A radially outside target residues idx."""
    pts = ch.cb_leg[list(idx)]
    xy = pts[:, :2] / np.linalg.norm(pts[:, :2], axis=1, keepdims=True)
    ca = pts.copy()
    ca[:, :2] += xy * (offset + 1.5)
    ca = np.concatenate([ca, ca[-1:] + np.array([[0.0, 0.0, 3.8 * (k + 1)] for k in range(max(0, n - len(ca)))]).reshape(-1, 3)])
    return ec.protein_chains(cf.atoms_from_ca(ca, chain_id="B"))[0]


def test_score_complex_hit_and_blanket_metrics():
    atoms, ch = long_helix()
    rasa = he.residue_rasa(atoms)
    epi = {"A": [10, 11, 12, 13, 14, 15], "B": [45, 46, 47, 48, 49, 50]}
    t = tinfo_for(ch, rasa, epi)
    b = binder_near(ch, epi["A"], offset=1.0)
    r = he.score_complex([ch, b], t, {"hot": "A", "cold": None})
    assert r["ok"] and r["hit_A"] == 1.0 and r["hit_B"] == 0.0 and r["frac_A"] >= 0.5 and r["frac_B"] == 0.0
    assert r["n_contact_res"] >= 4 and 0 < r["frac_surface_contacted"] < 0.5
    assert r["auc_matched"] > 0.5 and r["binder_rg"] > 0
    assert he.score_complex([ch], t, {"hot": "A"})["ok"] is False


def test_matched_decoy_auc_known_values():
    n = 40
    cb = np.array([[1.9 * i, 0.0, 0.0] for i in range(n)])      # 1.9 A apart: everything is within 15 A of the hotspot centroid
    rasa = np.full(n, 0.5)
    hot = [18, 19, 20]
    dmin = np.full(n, 9.0)
    dmin[hot] = 3.0
    auc, auc_b = he.matched_decoy_auc(cb, rasa, hot, dmin)
    assert auc == 1.0 and auc_b == 1.0
    dmin2 = np.full(n, 3.0)
    dmin2[hot] = 9.0
    assert he.matched_decoy_auc(cb, rasa, hot, dmin2) == (0.0, 0.0)
    assert he.matched_decoy_auc(cb, rasa, hot, np.full(n, 5.0)) == (0.5, 0.5)
    rasa_far = rasa.copy()
    rasa_far[[i for i in range(n) if i not in hot]] = 0.05 + 0.0           # decoys not equally exposed -> no matched decoys
    assert np.isnan(he.matched_decoy_auc(cb, rasa_far, hot, dmin)[0])


def _rows(n, pattern):
    """pattern: {(hot, cold, mode): (hitA, hitB)} -> rows"""
    rows = []
    for (hot, cold, mode), (ha, hb) in pattern.items():
        for i in range(n):
            rows.append({"target": "t", "ok": True, "hot": hot, "cold": cold, "mode": mode, "hit_A": float(ha), "hit_B": float(hb),
                         "auc_matched": 0.8 if hot else float("nan")})
    return rows


def test_swap_matrix_diagonal_dominance():
    good = _rows(20, {("A", None, "adj"): (1, 0), ("B", None, "adj"): (0, 1), (None, None, None): (0, 0)})
    m = he.swap_matrix(good, "t", n_boot=200)["adj"]
    assert m["diag_dominance"] == 1.0 and m["S_A"] == 1.0 and m["S_B"] == 1.0
    assert m["diag_dominance_ci"][0] > 0.9 and m["gain_A"] == 1.0 and m["auc_matched_mean"] == pytest.approx(0.8)
    # a model that always docks on patch A regardless of the spec: perfect 'hit A | spec A' but ZERO diagonal dominance
    fav = _rows(20, {("A", None, "adj"): (1, 0), ("B", None, "adj"): (1, 0), (None, None, None): (1, 0)})
    mf = he.swap_matrix(fav, "t", n_boot=200)["adj"]
    assert mf["diag_dominance"] == 0.0 and mf["gain_A"] == 0.0 and mf["specA"]["A"] == 1.0
    # a model that ignores the spec and binds randomly: dominance ~ 0
    rng = np.random.RandomState(0)
    rows = []
    for hot in ("A", "B", None):
        for i in range(200):
            rows.append({"target": "t", "ok": True, "hot": hot, "cold": None, "mode": "adj" if hot else None, "hit_A": float(rng.rand() < .5),
                         "hit_B": float(rng.rand() < .5), "auc_matched": 0.5})
    mr = he.swap_matrix(rows, "t", n_boot=300)["adj"]
    assert abs(mr["diag_dominance"]) < 0.12 and mr["diag_dominance_ci"][0] < 0 < mr["diag_dominance_ci"][1]


def test_coldspot_credit_compares_with_no_spec_baseline():
    rows = _rows(10, {("A", None, "adj"): (1, 1), ("A", "B", "adj"): (1, 0), ("B", None, "adj"): (0, 1), (None, None, None): (0, 1)})
    c = he.swap_matrix(rows, "t", n_boot=50)["coldspot"]["hotA_coldB"]
    assert c == {"cold_hit_with_cold_spec": 0.0, "cold_hit_hot_only": 1.0, "cold_hit_no_spec": 1.0}
