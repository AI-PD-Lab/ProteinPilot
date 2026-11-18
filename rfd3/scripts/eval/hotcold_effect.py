#!/usr/bin/env python
"""Does an adjacency HOTSPOT (ADJ state 3) / COLDSPOT (state 4) change the design, versus the SAME map with plain 1 / 0?

The semantics under test (old HARMONIA phase 9 / v2): HOTSPOT = a contact that is enforced more strongly and should get as much
interface (contact count) as the native fold has; COLDSPOT = a non-contact that is enforced more strongly (kept further apart).
So the effect is measured on exactly the promoted block pairs, arm vs base, same map otherwise:
  hot : n6 = number of CB-CB residue pairs < 6 A between the two blocks (interface size), min CB-CB distance, P(contact: min < 6 A)
  cold: min CB-CB distance between the blocks, P(contact formed: min < 6 A)  (lower P and larger distance = stronger separation)
Base = the existing `dense0_true` arm of the fold pack (every block pair revealed, 1 = contact, 0 = non-contact). Arms per target:
  hotk3 / hotall : the 3 contacts with the largest native n6 / ALL native contact pairs -> 3
  coldk3 / coldall: the 3 nearest near-miss non-contacts (6 <= native min dist <= 10) / ALL non-contacts with min dist <= 12 -> 4
Side effects are reported on the NON-promoted revealed pairs (contact recovery) so a hot arm that wins by wrecking the fold shows up.
The native value of each metric is listed next to it (a hotspot is native-matched in training: it cannot be expected to exceed native).

  python hotcold_effect.py build --fold-pack <specs_v2.tar.gz> --pack <out.tar.gz>
  python hotcold_effect.py score --tar <tag.tar.gz> --specs-tar <out.tar.gz> --out <dir>
generation: eval_checkpoint.sbatch SPECS=<out.tar.gz> CFGS="2.0" N=32 (stock null control: STOCK=1)
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import build_specs as bs  # noqa: E402
import eval_common as ec  # noqa: E402

CONTACT, NONCONTACT, HOT, COLD = 1.0, 0.0, 3.0, 4.0
THR = 6.0
BASE = "dense0_true"
SETS = ["hotk3", "hotall", "coldk3", "coldall"]


def _pairs(tinfo: dict):
    """(hot_candidates sorted by native n6 desc, cold_candidates sorted by native min dist asc) as (b1,b2) with b1<b2."""
    hot = sorted(((tuple(map(int, k.split("-"))), n) for k, n in tinfo["pair_n6"].items() if n > 0), key=lambda x: -x[1])
    mind = {tuple(map(int, k.split("-"))): d for k, d in tinfo["pair_min"].items()}
    pos = {tuple(p) for p in tinfo["pos_pairs"]}
    neg = {tuple(p) for p in tinfo["neg_pairs"]}
    hot = [p for p, _ in hot if p in pos]
    cold = sorted((p for p in neg if THR <= mind.get(p, 99) <= 12.0), key=lambda p: mind[p])
    return hot, cold, mind


def build(fold_pack: str, pack: str) -> dict:
    with tarfile.open(fold_pack, "r:*") as tf:
        man = json.load(tf.extractfile("manifest.json"))
        raw = {n: tf.extractfile(n).read() for n in tf.getnames() if n.startswith("tensors/")}
    out_man = {"spec_version": 3, "kind": "hotcold", "dialect": man["dialect"], "seed": man["seed"], "k_requested": 0, "label_function": man["label_function"],
               "targets": {}, "arms": {}, "unconditional": {}, "sets": {}}
    inputs, tensors = {}, {}
    for tname, ti in man["targets"].items():
        adj = np.load(io.BytesIO(raw[f"tensors/{tname}__{BASE}.adj.npy"]))
        ss = np.load(io.BytesIO(raw[f"tensors/{tname}__{BASE}.ss.npy"]))
        hot, cold, mind = _pairs(ti)
        sets = {"hotk3": hot[:3], "hotall": hot, "coldk3": cold[:3], "coldall": [p for p in cold if mind[p] <= 12.0]}
        out_man["targets"][tname] = {k: ti[k] for k in ("name", "pdb_id", "chain", "fold", "n_res", "blocks", "pair_n6", "pair_min", "pos_pairs", "neg_pairs", "ss")}
        out_man["sets"][tname] = {k: [list(p) for p in v] for k, v in sets.items()}
        arms = {BASE: adj}
        for sname, prs in sets.items():
            if not prs:
                continue
            a = adj.copy()
            for b1, b2 in prs:
                B1, B2 = ti["blocks"][b1], ti["blocks"][b2]
                s1, s2 = slice(B1["start"], B1["stop"]), slice(B2["start"], B2["stop"])
                old, new = (CONTACT, HOT) if sname.startswith("hot") else (NONCONTACT, COLD)
                for r, c in ((s1, s2), (s2, s1)):
                    blk = a[r, c]
                    blk[blk == old] = new
                    a[r, c] = blk
            arms[sname] = a
        for arm, a in arms.items():
            key = f"{tname}__{arm}"
            tensors[f"tensors/{key}.adj.npy"] = a.astype(np.float32)
            tensors[f"tensors/{key}.ss.npy"] = ss
            inputs[key] = {"dialect": 2, "length": str(ti["n_res"]), "extra": {"harmonia_eval": {"target": tname, "arm": arm}},
                           "ss_file": f"{ec.SPEC_ROOT_TOKEN}/tensors/{key}.ss.npy", "adj_file": f"{ec.SPEC_ROOT_TOKEN}/tensors/{key}.adj.npy"}
            out_man["arms"][key] = {"target": tname, "arm": arm, "key": key, "length": ti["n_res"], "n_changed": int(((a == HOT) | (a == COLD)).sum())}
    work = tempfile.mkdtemp(prefix="harmonia_hotcold_")
    os.makedirs(os.path.join(work, "tensors"))
    for n, a in tensors.items():
        np.save(os.path.join(work, n), a)
    ec.write_json(os.path.join(work, "inputs_template.json"), inputs, indent=1)
    ec.write_json(os.path.join(work, "manifest.json"), out_man, indent=None)
    bs.pack_dir(work, pack)
    return out_man


def pair_metrics(cb: np.ndarray, blocks: list, prs: list) -> dict:
    n6, md = [], []
    for b1, b2 in prs:
        B1, B2 = blocks[b1], blocks[b2]
        d = np.linalg.norm(cb[B1["start"]:B1["stop"], None] - cb[None, B2["start"]:B2["stop"]], axis=-1)
        n6.append(int((d < THR).sum()))
        md.append(float(d.min()))
    return {"n6": float(np.mean(n6)), "min_d": float(np.mean(md)), "contact": float(np.mean([m < THR for m in md]))}


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    ds = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def cmd_score(a):
    import score_designs as sd

    man = bs.read_manifest_from_tar(a.specs_tar)
    rows = []
    for name, data in sd.iter_tar_designs(a.tar):
        mt = ec.DESIGN_RE.match(os.path.basename(name))
        rec = man["arms"].get(mt.group("key"))
        if rec is None:
            continue
        cm = ec.CFG_DIR_RE.search(name.split("/")[-2])
        ti = man["targets"][rec["target"]]
        try:
            ch = ec.protein_chains(ec.read_structure(data, name))[0]
            if len(ch) != ti["n_res"]:
                continue
            cb = ch.cb_leg
            row = {"target": rec["target"], "arm": rec["arm"], "cfg": float(cm.group("cfg")) if cm else float("nan")}
            for sname, prs in man["sets"][rec["target"]].items():
                if prs:
                    for k, v in pair_metrics(cb, ti["blocks"], prs).items():
                        row[f"{sname}|{k}"] = v
            posrest = [p for p in ti["pos_pairs"] if p not in man["sets"][rec["target"]]["hotall"]]
            allpos = pair_metrics(cb, ti["blocks"], ti["pos_pairs"])
            row["allpos|contact"], row["allpos|n6"] = allpos["contact"], allpos["n6"]
            row["rest_pos|contact"] = pair_metrics(cb, ti["blocks"], posrest)["contact"] if posrest else float("nan")
            row["rg"] = float(np.sqrt(((cb - cb.mean(0)) ** 2).sum(-1).mean()))
            rows.append(row)
        except Exception:  # noqa: BLE001
            continue
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for cfg in sorted({r["cfg"] for r in rows}):
        for t in man["targets"]:
            ti = man["targets"][t]
            for sname, prs in man["sets"][t].items():
                if not prs:
                    continue
                base = [r for r in rows if r["target"] == t and r["cfg"] == cfg and r["arm"] == BASE and f"{sname}|n6" in r]
                arm = [r for r in rows if r["target"] == t and r["cfg"] == cfg and r["arm"] == sname]
                if not base or not arm:
                    continue
                nat = pair_metrics(np.array([[0.0, 0, 0]]), [], []) if False else None
                entry = {"n_base": len(base), "n_arm": len(arm), "n_pairs": len(prs),
                         "native_n6": float(np.mean([ti["pair_n6"][f"{b1}-{b2}"] for b1, b2 in prs])),
                         "native_min_d": float(np.mean([ti["pair_min"][f"{b1}-{b2}"] for b1, b2 in prs]))}
                for k in ("n6", "min_d", "contact"):
                    A = np.array([r[f"{sname}|{k}"] for r in arm])
                    B = np.array([r[f"{sname}|{k}"] for r in base])
                    entry[k] = {"base": float(B.mean()), "arm": float(A.mean()), "diff": boot_diff(A, B)}
                for k in ("rest_pos|contact", "rg"):
                    A = np.array([r[k] for r in arm if not math.isnan(r[k])])
                    B = np.array([r[k] for r in base if not math.isnan(r[k])])
                    if len(A) and len(B):
                        entry[k] = {"base": float(B.mean()), "arm": float(A.mean()), "diff": boot_diff(A, B)}
                res[f"{t}|{sname}|cfg{cfg:.1f}"] = entry
    ec.write_json(os.path.join(a.out, "hotcold_scores.json"), {"results": res, "n_designs": len(rows)}, indent=1)
    print(f"{len(rows)} designs scored")
    for kind, keys in (("HOT", ("n6", "min_d", "contact")), ("COLD", ("min_d", "contact"))):
        print(f"--- {kind}: effect = arm - base on the promoted pairs [95% CI]; native reference in brackets")
        for key, e in res.items():
            t, s, c = key.split("|")
            if s.startswith(kind.lower()):
                d = e["n6"]["diff"] if kind == "HOT" else e["min_d"]["diff"]
                print(f"{t:18s} {s:7s} {c}: n6 base {e['n6']['base']:.1f} arm {e['n6']['arm']:.1f} (native {e['native_n6']:.1f}) | min_d base {e['min_d']['base']:.2f} arm {e['min_d']['arm']:.2f} "
                      f"(native {e['native_min_d']:.2f}) | P(contact) {e['contact']['base']:.2f}->{e['contact']['arm']:.2f} | diff[{('n6' if kind == 'HOT' else 'min_d')}] {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]"
                      + (f" | rest-contacts {e['rest_pos|contact']['base']:.2f}->{e['rest_pos|contact']['arm']:.2f}" if 'rest_pos|contact' in e else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--fold-pack", default=ec.EVAL_ROOT + "/specs_v2.tar.gz")
    b.add_argument("--pack", required=True)
    s = sub.add_parser("score")
    s.add_argument("--tar", required=True)
    s.add_argument("--specs-tar", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        m = build(a.fold_pack, a.pack)
        for t, sets in m["sets"].items():
            print(t, {k: len(v) for k, v in sets.items()})
        print(len(m["arms"]), "specs ->", a.pack)
    else:
        cmd_score(a)


if __name__ == "__main__":
    main()
