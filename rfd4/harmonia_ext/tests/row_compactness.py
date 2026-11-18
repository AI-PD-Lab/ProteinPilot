"""How compact are the HARMONIA training rows? Per row (R1 harm_pdb dataset config, forced mode 'full'): kind (monomer / interface), tokens,
blocks, shown contact / non-contact block pairs, and per protein chain (>= 40 eligible tokens) the panel scorer's geometry
(eval/design_metrics.geometry_metrics on CA + ideal CB): rg_n04, lr_cb, split, and the universal split flag (split < 0.67 or lr_cb < 0.92
or rg_n04 > 2.15). Question: does an ADJ map in training co-occur with NON-compact structures (interfaces, crops, extended chains)?

  python -m harmonia_ext.tests.row_compactness <n_rows> <out.csv> [workers]
"""
from __future__ import annotations

import csv
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/mir/harmonia_study/scripts/eval")
import design_metrics as dm  # noqa: E402

from harmonia_ext.adj_data import HarmoniaDataset, token_cb  # noqa: E402

T_SPLIT, T_LR, T_RG = 0.67, 0.92, 2.15
CFG = dict(rows="all", split="train", seed=11, forced_mode="full", return_atom_array=True,
           transform={"crop_size": None, "crop_size_range": [0.9, 1.0], "loss_weight_cfg": {"non_standard": 2.5, "sampled_tip_atom": 0.0,
                      "unindexed": 2.5}, "max_crop_size": 256, "p_no_subcrop": 0.2, "path_length_search_cutoff": 64})
_DS = None


def token_ca(oa, ts):
    names = np.asarray(oa.atom_name).astype(str)
    ends = np.append(ts[1:], len(names))
    out = np.full((len(ts), 3), np.nan)
    for k, (a, b) in enumerate(zip(ts, ends)):
        j = np.flatnonzero(names[a:b] == "CA")
        if j.size:
            out[k] = oa.coord[a + j[0]]
    return out


def one(i):
    global _DS
    from atomworks.ml.utils.token import get_token_starts

    if _DS is None:
        _DS = HarmoniaDataset(**CFG)
    try:
        s = _DS[int(i)]
    except Exception as e:  # noqa: BLE001
        return [{"idx": i, "error": type(e).__name__}]
    oa, f, st = s["atom_array"], s["feats"], s["harm_stats"]
    row = _DS.row(int(i)) if hasattr(_DS, "row") else {}
    ts = get_token_starts(oa, add_exclusive_stop=False).astype(np.int64)
    prot = (f["token_type"].numpy() == 0)
    unit = np.asarray(oa.pn_unit_iid)[ts].astype(str)
    ca, cb = token_ca(oa, ts), token_cb(oa, ts)
    out = []
    for u in np.unique(unit[prot]):
        m = prot & (unit == u) & np.isfinite(ca).all(1) & np.isfinite(cb).all(1)
        if m.sum() < 40:
            continue
        g = dm.geometry_metrics(ca[m], cb[m])
        flag = g["split"] < T_SPLIT or g["lr_cb"] < T_LR or g["rg_n04"] > T_RG
        out.append({"idx": i, "error": "", "kind": row.get("kind", "?"), "pdb": row.get("pdb_id", "?"), "chain": u, "n_tok": int(st["n_tokens"]),
                    "n_chain": int(m.sum()), "n_units": int(len(np.unique(unit[prot]))), "n_blocks": st["n_blocks"],
                    "n_contact": st["n_contact_gt"], "n_noncontact": st["n_noncontact_gt"],
                    "rg_n04": round(g["rg_n04"], 4), "lr_cb": round(g["lr_cb"], 4), "split": round(g["split"], 4), "flag": int(flag),
                    "f_cut": int(g["split"] < T_SPLIT), "f_lr": int(g["lr_cb"] < T_LR), "f_rg": int(g["rg_n04"] > T_RG)})
    return out or [{"idx": i, "error": "no_chain_ge40"}]


def main():
    n, path = int(sys.argv[1]), sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    size = len(HarmoniaDataset(**CFG))
    idx = np.random.default_rng(0).choice(size, n, replace=False)
    with Pool(workers) as p:
        rows = [r for rr in p.imap_unordered(one, idx, chunksize=2) for r in rr]
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if not r["error"]]
    print(f"dataset rows {size}, sampled {n}, chains scored {len(ok)}, errors {len(rows) - len(ok)}")
    for kind in ("monomer", "interface"):
        k = [r for r in ok if r["kind"] == kind]
        if k:
            print(f"  {kind:9s} chains={len(k):5d} flagged={np.mean([r['flag'] for r in k]):.3f} [cut {np.mean([r['f_cut'] for r in k]):.3f} "
                  f"lr {np.mean([r['f_lr'] for r in k]):.3f} rg {np.mean([r['f_rg'] for r in k]):.3f}]  "
                  f"median rg_n04 {np.median([r['rg_n04'] for r in k]):.3f} lr_cb {np.median([r['lr_cb'] for r in k]):.3f}")


if __name__ == "__main__":
    main()
