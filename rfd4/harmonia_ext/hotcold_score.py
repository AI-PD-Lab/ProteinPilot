"""HOT / COLD effect on PAIRED designs: two eval tars generated with the same keys / guidance / seeds, one plain (--hot 0 --cold 0) and one
with promoted pairs (--hot k --cold k). Same selection as generate.promoted. Per promoted block pair and design:
  HOT  -> interface size n6 = reconstructed-CB pairs < 6 A between the two blocks (bigger = HOT works)
  COLD -> min reconstructed-CB distance between the two blocks (bigger = COLD works)
Reports the pair-weighted mean paired difference (treated - plain) with a DESIGN-cluster bootstrap 95 % CI (whole designs resampled: the
pairs of one backbone are not independent; red-team N001-2 #1), overall and per target, plus the plain-contact rates. Both tars must come
from the same generation settings except --hot/--cold (meta.json checked; red-team N001-2 #2).

  python harmonia_ext/hotcold_score.py --plain <tar> --treated <tar> [--hot 3 --cold 3] [--manifest <json>]
"""
from __future__ import annotations

import os
import argparse
import ast
import gzip
import io
import json
import re
import tarfile
from collections import defaultdict

import numpy as np

MANIFEST = os.environ.get("HARMONIA_MANIFEST", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rfd3", "scripts", "eval", "kit", "manifest.json"))
NAME = re.compile(r"designs/cfg(?P<cfg>[\d.]+)/(?P<key>.+)_(?P<b>\d+)_model_(?P<m>\d+)\.cif\.gz$")


def lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def promoted(target, arm, hot_k, cold_k):
    n6 = {k: int(v) for k, v in lit(target["pair_n6"]).items()}
    dm = {k: float(v) for k, v in lit(target["pair_min"]).items()}
    key = lambda p: f"{min(p)}-{max(p)}"
    pos = [tuple(p) for p in lit(arm["pos_pairs"])]
    neg = [tuple(p) for p in lit(arm["neg_pairs"])]
    hot = sorted(pos, key=lambda p: -n6.get(key(p), 0))[:hot_k] if hot_k else []
    cold = sorted(neg, key=lambda p: dm.get(key(p), 1e9))[:cold_k] if cold_k else []
    return hot, cold


def cb_coords(data: bytes) -> np.ndarray:
    from biotite.structure import get_residue_starts
    from biotite.structure.io import pdbx

    st = pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(gzip.decompress(data).decode())), model=1)
    st = st[st.hetero == False]  # noqa: E712
    rs = list(get_residue_starts(st)) + [len(st)]
    out = np.full((len(rs) - 1, 3), np.nan)
    for i in range(len(rs) - 1):
        r = st[rs[i]:rs[i + 1]]
        g = {a: r.coord[r.atom_name == a] for a in ("N", "CA", "C")}
        if all(len(v) for v in g.values()):
            n, ca, c = g["N"][0], g["CA"][0], g["C"][0]
            b, cc = ca - n, c - ca
            out[i] = -0.58273431 * np.cross(b, cc) + 0.56802827 * b - 0.54067466 * cc + ca
    return out


def pair_stats(cb, blocks, a, b):
    (a0, a1), (b0, b1) = blocks[a], blocks[b]
    d = np.linalg.norm(cb[a0:a1][:, None] - cb[b0:b1][None], axis=-1)
    return int((d < 6.0).sum()), float(np.nanmin(d))


PAIRING_KEYS = ("model", "manifest", "cfgs", "n", "batch", "keys", "seed", "tsr_k", "cath")   # everything that fixes the noise schedule + spec


def load(tar_path):
    out, meta = {}, None
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            mt = NAME.search(m.name)
            if mt and m.isfile():
                out[(mt["cfg"], mt["key"], int(mt["b"]), int(mt["m"]))] = tf.extractfile(m).read()
            elif m.name.endswith("/meta.json"):
                meta = json.loads(tf.extractfile(m).read())
    return out, meta


def check_paired(mp, mt, kp, kt):
    """Refuse unpaired inputs: different generation settings (other than hot/cold) or different design sets."""
    assert mp is not None and mt is not None, "meta.json missing"
    bad = [k for k in PAIRING_KEYS if mp.get(k) != mt.get(k)]
    assert not bad, f"not paired, meta differs in {bad}"
    assert kp == kt, f"not paired: {len(kp ^ kt)} designs present in only one tar"


def boot(groups, n=2000, seed=0):
    """Pair-weighted mean of the per-design groups of paired differences, CI from resampling whole designs."""
    s = np.array([[np.sum(g), len(g)] for g in groups if len(g)], float)
    if s.size == 0:
        return float("nan"), float("nan"), float("nan")
    b = s[np.random.default_rng(seed).integers(len(s), size=(n, len(s)))].sum(1)
    r = b[:, 0] / b[:, 1]
    return float(s[:, 0].sum() / s[:, 1].sum()), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True)
    ap.add_argument("--treated", required=True)
    ap.add_argument("--hot", type=int, default=3)
    ap.add_argument("--cold", type=int, default=3)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--json-out")
    a = ap.parse_args()
    M = json.load(open(a.manifest))
    (P, mp), (T, mt) = load(a.plain), load(a.treated)
    check_paired(mp, mt, set(P), set(T))
    rows = defaultdict(lambda: defaultdict(list))   # (cfg, kind, target) -> design -> paired diffs
    raw = defaultdict(list)        # (cfg, kind, variant) -> values
    for k in sorted(set(P) & set(T)):
        cfg, key = k[0], k[1]
        arm = M["arms"].get(key)
        if arm is None or str(arm.get("has_adj")) != "True":
            continue
        tgt = M["targets"][arm["target"]]
        blocks = {int(b["id"]): (int(b["start"]), int(b["stop"])) for b in lit(tgt["blocks"])}
        hot, cold = promoted(tgt, arm, a.hot, a.cold)
        cp, ct = cb_coords(P[k]), cb_coords(T[k])
        for kind, pairs, idx in (("HOT_n6", hot, 0), ("COLD_dmin", cold, 1)):
            for p, q in pairs:
                vp, vt = pair_stats(cp, blocks, p, q)[idx], pair_stats(ct, blocks, p, q)[idx]
                rows[(cfg, kind, arm["target"])][k].append(vt - vp)
                rows[(cfg, kind, "ALL")][k].append(vt - vp)
                raw[(cfg, kind, "plain")].append(vp)
                raw[(cfg, kind, "treated")].append(vt)
    res = {}
    for (cfg, kind, tg), v in sorted(rows.items()):
        m, lo, hi = boot(list(v.values()))
        res[f"{cfg}|{kind}|{tg}"] = {"n_pairs": sum(map(len, v.values())), "n_designs": len(v), "diff": m, "ci": [lo, hi]}
    for cfg in sorted({k[0] for k in rows}):
        print(f"== guidance w={cfg}")
        for kind in ("HOT_n6", "COLD_dmin"):
            pl, tr = raw[(cfg, kind, "plain")], raw[(cfg, kind, "treated")]
            if not pl:
                continue
            r = res[f"{cfg}|{kind}|ALL"]
            print(f"  {kind:10s} plain {np.mean(pl):6.2f}  treated {np.mean(tr):6.2f}  paired diff {r['diff']:+.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] pairs={r['n_pairs']} designs={r['n_designs']}")
            for tg in sorted({k[2] for k in rows if k[0] == cfg and k[1] == kind} - {"ALL"}):
                rr = res[f"{cfg}|{kind}|{tg}"]
                print(f"      {tg:18s} {rr['diff']:+.2f} [{rr['ci'][0]:+.2f},{rr['ci'][1]:+.2f}] designs={rr['n_designs']}")
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
