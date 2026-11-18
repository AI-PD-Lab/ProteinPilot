#!/usr/bin/env python
"""Cross-chain (binder <-> target) test of the ADJ HOTSPOT / COLDSPOT states, the counterpart of hotcold_effect.py (which tests intra-fold block pairs).

Semantics (user, 2026-10-18): HOT = a contact enforced more strongly with as much interface as possible ("maximum alignment with the target");
COLD = a non-contact enforced more strongly. So each epitope patch of the target is specified four ways, everything else identical:
    none      no adjacency at all                            contact  'A.. : B1-L'   (state 1)      hot   'A.. :: B1-L' (state 3)
    notouch   'A.. !: B1-L' (state 0)                         cold    'A.. !!: B1-L' (state 4)
Effects (per epitope E in {A, B}, shared origin M so only the spec differs; same-face compact patches from hotspot_eval_v3):
    HOT effect  = hot - contact on epitope E   (recall of E = fraction of its residues within 4.5 A of the binder, n contacted target residues, hit)
    COLD effect = cold - notouch on epitope E  (a coldspot should keep the binder off at least as well as a plain non-contact)
    plus contact - none (does plain adjacency steer at all) and the swap (spec on A -> recall of B).
Wrap guard: a binder that wraps the whole target contacts everything; hit() requires <= 22 contacted target residues, no clash, interface centroid within 8 A.

  python crosschain_hotcold.py build --pack <out.tar.gz> [--targets 1ycr:A 4zqk:A 1ubq:A file:<pdb>:A]
  python crosschain_hotcold.py score --tar <tag.tar.gz> --specs-tar <pack> --out <dir>
generation: eval_checkpoint.sbatch SPECS=<pack> CFGS="2.0" N=32 EXTRA="inference_sampler.step_scale=3 inference_sampler.gamma_0=0.2"
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import build_specs as bs  # noqa: E402
import eval_common as ec  # noqa: E402
import hotspot_eval_v3 as h3  # noqa: E402
from hotspot_eval import _ranges, residue_rasa  # noqa: E402

ARMS = ["none", "contactA", "contactB", "hotA", "hotB", "notouchA", "notouchB", "coldA", "coldB"]
OP = {"contact": ":", "hot": "::", "notouch": "!:", "cold": "!!:"}
FIELDS = ("hit", "recall", "n_contact_res")


def make_arms(tname: str, pair: dict, k: int, n_target: int, binder_len: int, heavy: np.ndarray):
    E = {"A": [i + 1 for i in pair["A"]], "B": [i + 1 for i in pair["B"]]}
    ori = h3.stock_ori(heavy, (pair["cenA"] + pair["cenB"]) / 2)
    base = {"dialect": 2, "input": f"{ec.SPEC_ROOT_TOKEN}/targets/{tname}.pdb", "contig": f"A1-{n_target},/0,{binder_len}-{binder_len}",
            "select_fixed_atoms": True, "is_non_loopy": True, "ori_token": ori}
    inputs, recs = {}, {}
    for arm in ARMS:
        spec = dict(base, extra={"harmonia_eval": {"target": tname, "pair": k, "arm": arm}})
        kind, ep = (arm[:-1], arm[-1]) if arm != "none" else (None, None)
        if kind:
            spec["adj_pairs"] = [f"{_ranges(E[ep])} {OP[kind]} B1-{binder_len}"]
        key = f"{tname}__p{k}_{arm}"
        inputs[key] = spec
        recs[key] = {"target": tname, "pair": k, "arm": arm, "key": key, "kind": kind, "ep": ep, "ori_token": ori,
                     "length": n_target + binder_len, "binder_len": binder_len}
    return inputs, recs


def build(out_dir: str, targets: list[str], binder_len: int = 70, n_pairs: int = 1) -> dict:
    import biotite.structure as struc
    import biotite.structure.io.pdb as pdb

    os.makedirs(os.path.join(out_dir, "targets"), exist_ok=True)
    manifest = {"spec_version": 3, "kind": "crosschain_hotcold", "targets": {}, "arms": {}, "unconditional": {}, "binder_len": binder_len}
    inputs = {}
    for spec in targets:
        name, t = h3._target_from_spec(spec)
        name = name.replace("hot3_", "xhc_")
        chobj = ec.protein_chains(t)[0]
        rasa = residue_rasa(t)
        pairs = h3.same_face_pairs(chobj.cb_leg, rasa >= 0.25, t.coord, n_pairs)
        if not pairs:
            print(f"WARNING {spec}: no same-face pair found, target skipped")
            continue
        t2 = t.copy()
        starts = struc.get_residue_starts(t2)
        t2.res_id = np.repeat(np.arange(1, len(starts) + 1), np.diff(list(starts) + [t2.array_length()]))
        t2.chain_id[:] = "A"
        f = pdb.PDBFile()
        f.set_structure(t2)
        f.write(os.path.join(out_dir, "targets", f"{name}.pdb"))
        for k, pr in enumerate(pairs):
            ins, recs = make_arms(name, pr, k, len(chobj), binder_len, t2.coord)
            inputs.update(ins)
            manifest["arms"].update(recs)
        manifest["targets"][name] = {
            "name": name, "src": spec, "n_res": len(chobj), "rasa": [round(float(x), 3) for x in rasa], "cb": np.round(chobj.cb_leg, 2).tolist(),
            "pairs": [{"A": p["A"], "B": p["B"], "sep": round(p["sep"], 2), "ori_dist": round(p["angle"], 1), "diam": round(p["diam"], 2),
                       "cenA": np.round(p["cenA"], 2).tolist(), "cenB": np.round(p["cenB"], 2).tolist()} for p in pairs]}
    ec.write_json(os.path.join(out_dir, "inputs_template.json"), inputs, indent=1)
    ec.write_json(os.path.join(out_dir, "manifest.json"), manifest, indent=None)
    return manifest


def summarise(rows: list[dict], target: str, pair: int, cfg: float) -> dict:
    sel = [r for r in rows if r["target"] == target and r["pair"] == pair and r["cfg"] == cfg and r.get("ok")]

    def g(arm, E, field):
        return np.array([r[f"{field}_{E}"] for r in sel if r["arm"] == arm], float)
    out = {"n": {a: int(len(g(a, "A", "hit"))) for a in ARMS}, "mean": {}, "effect": {}}
    for a in ARMS:
        for E in "AB":
            if len(g(a, E, "hit")):
                out["mean"][f"{a}|{E}"] = {f: float(g(a, E, f).mean()) for f in FIELDS}
    eff = out["effect"]
    for name, (arm_fn, ref_fn) in {
        "hot_vs_contact": (lambda E: f"hot{E}", lambda E: f"contact{E}"),
        "cold_vs_notouch": (lambda E: f"cold{E}", lambda E: f"notouch{E}"),
        "contact_vs_none": (lambda E: f"contact{E}", lambda E: "none"),
        "notouch_vs_none": (lambda E: f"notouch{E}", lambda E: "none"),
        "hot_vs_none": (lambda E: f"hot{E}", lambda E: "none"),
        "cold_vs_none": (lambda E: f"cold{E}", lambda E: "none"),
    }.items():
        for E in "AB":
            a_, r_ = arm_fn(E), ref_fn(E)
            if len(g(a_, E, "hit")) and len(g(r_, E, "hit")):
                eff[f"{name}|{E}"] = {f: h3.diff_ci(g(a_, E, f), g(r_, E, f)) if f == "hit" else _boot(g(a_, E, f), g(r_, E, f)) for f in FIELDS}
    for kind in ("contact", "hot"):  # swap: spec on A -> what happens on B (and vice versa), relative to the same spec on the other patch
        for E, O in (("A", "B"), ("B", "A")):
            if len(g(f"{kind}{E}", O, "hit")) and len(g(f"{kind}{O}", O, "hit")):
                eff[f"{kind}_swap|{E}->{O}"] = {"recall_other_patch_own_spec_minus_this_spec": _boot(g(f"{kind}{O}", O, "recall"), g(f"{kind}{E}", O, "recall"))}
    return out


def _boot(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    ds = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def cmd_build(a):
    tmp = None if a.work_dir else tempfile.TemporaryDirectory(prefix="harmonia_xhc_")
    work = a.work_dir or tmp.name
    try:
        man = build(work, a.targets, a.binder_len, a.n_pairs)
        bs.pack_dir(work, a.pack)
    finally:
        if tmp:
            tmp.cleanup()
    for t in man["targets"].values():
        for k, p in enumerate(t["pairs"]):
            print(f"{t['name']} pair{k}: {t['n_res']} res, A{[i + 1 for i in p['A']]} B{[i + 1 for i in p['B']]} sep {p['sep']} A")
    print(f"{len(man['arms'])} specs, packed {a.pack}")


def cmd_score(a):
    import score_designs as sd

    manifest = bs.read_manifest_from_tar(a.specs_tar)
    rows = []
    for name, data in sd.iter_tar_designs(a.tar):
        mt = ec.DESIGN_RE.match(os.path.basename(name))
        rec = manifest["arms"].get(mt.group("key"))
        if rec is None:
            continue
        cm = ec.CFG_DIR_RE.search(name.split("/")[-2])
        row = {"ok": False}
        try:
            chains = ec.protein_chains(ec.read_structure(data, name))
            ti = manifest["targets"][rec["target"]]
            if len(chains) >= 2 and len(chains[0]) == ti["n_res"]:
                pr = ti["pairs"][rec["pair"]]
                row = {"ok": True}
                for E in "AB":
                    for k, v in h3.hit_metrics(chains[0], chains[1], pr[E], np.array(pr[f"cen{E}"])).items():
                        row[f"{k}_{E}"] = v
        except Exception as e:  # noqa: BLE001
            row = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        row.update(target=rec["target"], pair=rec["pair"], arm=rec["arm"], cfg=float(cm.group("cfg")) if cm else float("nan"))
        rows.append(row)
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for cfg in sorted({r["cfg"] for r in rows}):
        for t, ti in manifest["targets"].items():
            for k in range(len(ti["pairs"])):
                res[f"{t}|p{k}|cfg{cfg:.1f}"] = summarise(rows, t, k, cfg)
    ec.write_json(os.path.join(a.out, "crosschain_hotcold_scores.json"), {"results": res, "n_designs": len(rows), "n_ok": sum(bool(r["ok"]) for r in rows)}, indent=1)
    import csv
    with open(os.path.join(a.out, "crosschain_hotcold_rows.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} designs, {sum(bool(r['ok']) for r in rows)} scored")
    for key, v in res.items():
        m = v["mean"]
        print(f"{key}: recall(own patch) none {m.get('none|A', {}).get('recall', float('nan')):.2f} | contact {m.get('contactA|A', {}).get('recall', float('nan')):.2f}"
              f" hot {m.get('hotA|A', {}).get('recall', float('nan')):.2f} | notouch {m.get('notouchA|A', {}).get('recall', float('nan')):.2f} cold {m.get('coldA|A', {}).get('recall', float('nan')):.2f}  (patch A)")
        for ek in ("hot_vs_contact|A", "hot_vs_contact|B", "cold_vs_notouch|A", "cold_vs_notouch|B", "contact_vs_none|A", "contact_vs_none|B"):
            e = v["effect"].get(ek)
            if e:
                print(f"    {ek:20s} d_recall {e['recall'][0]:+.2f} [{e['recall'][1]:+.2f},{e['recall'][2]:+.2f}]  d_hit {e['hit'][0]:+.2f}  d_nres {e['n_contact_res'][0]:+.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--targets", nargs="+", default=["1ycr:A", "4zqk:A", "1ubq:A", "file:foundry/models/rfd3/docs/input_pdbs/5o45_cropped.pdb:A"])
    b.add_argument("--pack", required=True)
    b.add_argument("--work-dir", default=None)
    b.add_argument("--binder-len", type=int, default=70)
    b.add_argument("--n-pairs", type=int, default=1)
    s = sub.add_parser("score")
    s.add_argument("--tar", required=True)
    s.add_argument("--specs-tar", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    (cmd_build if a.cmd == "build" else cmd_score)(a)


if __name__ == "__main__":
    main()
