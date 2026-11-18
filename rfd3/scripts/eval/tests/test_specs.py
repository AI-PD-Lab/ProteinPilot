"""Spec-builder invariants: no zero that is not a native zero, fill = 2, symmetric, diagonal 0, loops masked, covalent
neighbours masked, sparse/dense arm semantics, deterministic pack, shard plan. Runs on a synthetic helix bundle through the
REAL training label function (loaded from the repo file) and, when the PDB mirror is reachable, on the real references."""
import hashlib
import json
import os

import numpy as np
import pytest

import build_specs as bs
import conftest as cf
import eval_common as ec
import make_plan as mp

LBL = ec.load_label_module()
REF = dict(name="bundle_syn", pdb="syn", chain="A", fold="synthetic 4-helix bundle")
MIRROR_OK = os.path.exists(ec.PDB_MIRROR_TAR) and os.path.exists(ec.PDB_MIRROR_INDEX)


def synthetic_target():
    atoms = cf.atoms_from_ca(cf.helix_bundle_ca())
    chain = ec.protein_chains(atoms)[0]
    target, arr = bs.build_target(REF, atoms, chain, LBL)
    return target, arr, chain


def native_pair_min(chain, blocks, i, j):
    return ec.block_min_dist(chain.cb, ec.block_idx(blocks, i), ec.block_idx(blocks, j))


def check_arm_invariants(target, chain, arms):
    """The invariants of the task statement, checked against an independent recomputation of CB-CB distances."""
    L, blocks = target["n_res"], target["blocks"]
    ss = np.asarray(target["ss"])
    loop_tok = np.isin(ss, (ec.SS_LOOP,))
    nat_pos = {tuple(p) for p in target["pos_pairs"]}
    nat_neg = {tuple(p) for p in target["neg_pairs"]}
    cov = {tuple(p) for p in target["cov_masked_pairs"]}
    for name, a in arms.items():
        adj = a["adj"]
        if name == "ss_only":
            assert adj is None and not a["record"]["has_adj"] and 3 not in set(np.asarray(a["ss"]).tolist())
            continue
        assert adj.shape == (L, L) and adj.dtype == np.float32
        assert set(np.unique(adj).tolist()) <= {0.0, 1.0, 2.0}
        assert np.array_equal(adj, adj.T), name                               # symmetric
        assert np.all(np.diag(adj) == 0), name                                # diagonal 0 (adj_pairs / WP-LABEL convention)
        off = ~np.eye(L, dtype=bool)
        # loops are masked in every arm, rows and columns (fill = 2)
        assert np.all(adj[loop_tok][off[loop_tok]] == 2) and np.all(adj[:, loop_tok].T[off[loop_tok]] == 2), name
        in_block = np.zeros(L, bool)
        for b in blocks:
            in_block[b["start"]:b["stop"]] = True
        assert np.all(adj[~in_block][off[~in_block]] == 2), name
        # block pairs: 0 only where the native pair is a native zero (CB-CB >= 6 A, independently recomputed); 1 only where < 6 A
        nb = len(blocks)
        for i in range(nb):
            for j in range(i + 1, nb):
                rect = adj[np.ix_(ec.block_idx(blocks, i), ec.block_idx(blocks, j))]
                u = set(np.unique(rect).tolist())
                assert len(u) == 1, f"{name}: non-uniform rectangle {(i, j)}"
                v = u.pop()
                dmin = native_pair_min(chain, blocks, i, j)
                if v == 0:
                    assert dmin >= ec.CONTACT_CUT and (i, j) in nat_neg, f"{name}: FALSE ZERO at {(i, j)} (min CB {dmin:.2f})"
                if v == 1:
                    assert dmin < ec.CONTACT_CUT and (i, j) in nat_pos, f"{name}: false contact at {(i, j)}"
                assert not ((i, j) in cov and v != 2), f"{name}: covalent neighbours must stay masked"
        # intra-block zeros exist only in the arm that copies the label function's block-diagonal convention
        intra_zero = any(adj[b["start"]:b["stop"], b["start"]:b["stop"]][~np.eye(b["stop"] - b["start"], dtype=bool)].min() == 0 for b in blocks)
        assert intra_zero == a["record"]["intra_zero"], name


def test_synthetic_bundle_arms_obey_all_invariants():
    target, _, chain = synthetic_target()
    assert target["n_blocks"] == 4 and len(target["pos_pairs"]) + len(target["neg_pairs"]) >= 5
    arms = bs.make_arms(target, seed=1)
    assert list(arms) == ["full", "sparse8", "dense0_true", "ss_only"]
    check_arm_invariants(target, chain, arms)


def test_arm_semantics():
    target, arr, chain = synthetic_target()
    arms = bs.make_arms(target, seed=3)
    pos, neg = {tuple(p) for p in target["pos_pairs"]}, {tuple(p) for p in target["neg_pairs"]}
    rf, rs, rd = (arms[k]["record"] for k in ("full", "sparse8", "dense0_true"))
    assert {tuple(p) for p in rf["pos_pairs"]} == pos and {tuple(p) for p in rf["neg_pairs"]} == neg
    k = bs.sparse_k(len(pos))
    assert len(rs["pos_pairs"]) == k == len(rd["pos_pairs"])
    assert 0 < k < len(pos)                                                      # some native contact stays hidden
    assert {tuple(p) for p in rs["pos_pairs"]} <= pos and rs["pos_pairs"] == rd["pos_pairs"]   # same revealed contacts
    assert rs["neg_pairs"] == [] and {tuple(p) for p in rd["neg_pairs"]} == neg                  # dense0_true = ALL native zeros
    off = ~np.eye(len(arr["ss"]), dtype=bool)
    assert (arms["sparse8"]["adj"][off] == 0).sum() == 0                          # sparse arm reveals no zeros at all
    # the hidden native contacts are masked, never zero, in dense0_true
    hidden = pos - {tuple(p) for p in rd["pos_pairs"]}
    blocks = target["blocks"]
    for i, j in hidden:
        assert np.all(arms["dense0_true"]["adj"][np.ix_(ec.block_idx(blocks, i), ec.block_idx(blocks, j))] == 2)
    # full == label function output with loops masked (compare on the H/E x H/E inter-block entries)
    lab = arr["adj_label"]
    for i, j in pos | neg:
        r = (np.ix_(ec.block_idx(blocks, i), ec.block_idx(blocks, j)))
        assert np.array_equal(arms["full"]["adj"][r], lab[r])


def test_sparse_k_rule_and_determinism():
    assert [bs.sparse_k(n) for n in (0, 1, 2, 3, 4, 5, 6, 15, 40)] == [0, 1, 1, 2, 2, 3, 3, 8, 8]
    target, _, _ = synthetic_target()
    a1, a2 = bs.make_arms(target, seed=5), bs.make_arms(target, seed=5)
    assert all(np.array_equal(a1[k]["adj"], a2[k]["adj"]) for k in ("full", "sparse8", "dense0_true"))
    assert a1["sparse8"]["record"]["pos_pairs"] == a2["sparse8"]["record"]["pos_pairs"]


def test_covalent_neighbour_pairs_are_masked(monkeypatch):
    """Two helices with NO residue between them are sequence neighbours: their pair must be 2 whatever the label says."""
    ss = np.array([2] * 3 + [0] * 8 + [0 + 1] * 6 + [2] * 3)          # H(3-10) directly followed by E(11-16)
    L = len(ss)
    adj = np.full((L, L), 2.0, np.float32)
    adj[3:11, 11:17] = 1.0
    adj[11:17, 3:11] = 1.0
    np.fill_diagonal(adj, 0)
    monkeypatch.setattr(bs, "native_labels", lambda atoms, lbl: (ss, adj))
    atoms = cf.atoms_from_ca(np.array([[3.8 * i, 0.1 * (i % 2), 0] for i in range(L)]))
    chain = ec.protein_chains(atoms)[0]
    target, _ = bs.build_target(REF, atoms, chain, LBL)
    assert target["cov_masked_pairs"] == [[0, 1]] and target["pos_pairs"] == [] and target["neg_pairs"] == []
    assert ec.covalent_neighbours(target["blocks"]) == {(0, 1)}


def _write_synthetic_ref(tmp_path):
    import biotite.structure.io.pdb as pdb

    f = pdb.PDBFile()
    f.set_structure(cf.atoms_from_ca(cf.helix_bundle_ca()))
    p = str(tmp_path / "syn.pdb")
    f.write(p)
    return p


def test_build_pack_render_pipeline_is_deterministic(tmp_path):
    ref = _write_synthetic_ref(tmp_path)
    out1, out2 = str(tmp_path / "w1"), str(tmp_path / "w2")
    m1 = bs.build(out1, refs=[REF], seed=7, ref_files={"bundle_syn": ref})
    bs.build(out2, refs=[REF], seed=7, ref_files={"bundle_syn": ref})
    t1, t2 = str(tmp_path / "a.tar.gz"), str(tmp_path / "b.tar.gz")
    bs.pack_dir(out1, t1)
    bs.pack_dir(out2, t2)
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert sha(t1) == sha(t2)                                                  # byte-identical pack for identical inputs
    inputs, man = bs.unpack_and_render(t1, str(tmp_path / "root"))
    assert set(inputs) == set(man["arms"]) | set(man["unconditional"])
    spec = inputs["bundle_syn__full"]
    assert os.path.exists(spec["ss_file"]) and os.path.exists(spec["adj_file"]) and ec.SPEC_ROOT_TOKEN not in json.dumps(inputs)
    assert "adj_file" not in inputs["bundle_syn__ss_only"] and "ss_file" in inputs["bundle_syn__ss_only"]
    assert inputs["uncond_L064__unconditional"] == {"dialect": 2, "length": "64",
                                                    "extra": {"harmonia_eval": {"spec_version": 2, "target": "uncond_L064", "arm": "unconditional"}}}
    adj = np.load(spec["adj_file"])
    ss = np.load(spec["ss_file"])
    assert adj.shape == (len(ss), len(ss)) and spec["length"] == str(len(ss))
    z = np.load(os.path.join(str(tmp_path / "root"), "refs.npz"))
    assert z["bundle_syn_ca"].shape == (len(ss), 3)
    assert bs.summary_text(m1).count("bundle_syn__") >= 4
    # arms are keyed <target>__<arm>; manifest records the revealed pairs the scorer needs
    assert man["arms"]["bundle_syn__dense0_true"]["neg_pairs"] == man["targets"]["bundle_syn"]["neg_pairs"]


def test_make_plan_balances_shards_and_is_seed_deterministic(tmp_path):
    ref = _write_synthetic_ref(tmp_path)
    w = str(tmp_path / "w")
    bs.build(w, refs=[REF], seed=1, ref_files={"bundle_syn": ref})
    tar = str(tmp_path / "s.tar.gz")
    bs.pack_dir(w, tar)
    inputs, man = bs.unpack_and_render(tar, str(tmp_path / "root"))
    sh = mp.make_plan(inputs, man, [1.0, 1.5, 2.0], [1.0], n=16, batch=4, gpus=4, base_seed=11, out_dir=str(tmp_path / "plan"))
    names = {s["shard"] for s in sh}
    assert names == {"cfg1.0__bundle_syn", "cfg1.5__bundle_syn", "cfg2.0__bundle_syn", "cfg1.0__uncond_L064", "cfg1.0__uncond_L120", "cfg1.0__uncond_L200"}
    assert all(s["n_batches"] == 4 and s["batch_size"] == 4 for s in sh)
    assert all(not s["shard"].startswith(("cfg1.5__uncond", "cfg2.0__uncond")) for s in sh)    # CFG is a no-op without conditioning
    for cfg in (1.0, 1.5, 2.0):                      # every conditioned key exactly once per CFG; unconditional keys only at 1.0
        covered = [k for s in sh if s["cfg"] == cfg for k in s["keys"]]
        want = {k for k in inputs if "uncond" not in k} | ({k for k in inputs if "uncond" in k} if cfg == 1.0 else set())
        assert len(covered) == len(set(covered)) and set(covered) == want, cfg
    assert all(os.path.exists(s["input_json"]) for s in sh)
    mp.write_plan(sh, str(tmp_path / "plan"))
    lines = open(tmp_path / "plan" / "plan.tsv").read().splitlines()
    assert lines[0].split("\t")[:4] == ["shard", "gpu", "cfg", "seed"] and len(lines) == len(sh) + 1
    sh2 = mp.make_plan(inputs, man, [1.0, 1.5, 2.0], [1.0], n=16, batch=4, gpus=4, base_seed=11, out_dir=str(tmp_path / "plan2"))
    assert {s["shard"]: s["seed"] for s in sh} == {s["shard"]: s["seed"] for s in sh2}


@pytest.mark.skipif(not MIRROR_OK, reason="PDB mirror not reachable")
def test_real_references_obey_all_invariants_and_label_function_agrees_with_pSEA():
    pytest.importorskip("pyarrow")
    for ref in bs.REFERENCES + bs.EXTRA_REFERENCES:
        atoms, chain = bs.load_reference(ref, LBL)
        target, arr = bs.build_target(ref, atoms, chain, LBL)
        assert target["n_blocks"] >= 3, ref["name"]
        # the label function's SS equals the CA-only P-SEA the scorer applies to designs
        assert np.array_equal(arr["ss"], ec.sse_labels(chain.ca)), ref["name"]
        arms = bs.make_arms(target, seed=0)
        check_arm_invariants(target, chain, arms)
        # reference residues 1:1 with the spec length
        assert target["n_res"] == len(chain) == len(arr["ss"])
