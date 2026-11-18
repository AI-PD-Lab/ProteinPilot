"""CPU smoke of the R2 data changes: compactness gate (non-compact rows -> no SS, no ADJ) and dense-zero noise (loop / same-block pairs -> 0).
Usage: python -m harmonia_ext.tests.smoke_gate [n]"""
import sys
import numpy as np
from harmonia_ext.adj_data import HarmoniaDataset
from harmonia_ext import adj_labels as L
from rfproteina.conditions import C_SEC

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ds = HarmoniaDataset(rows="all", split="train", seed=3, forced_mode="full", gate_compact=True, p_dense_zero=1.0,
                     transform={"crop_size": None, "crop_size_range": [0.9, 1.0], "max_crop_size": 256, "p_no_subcrop": 0.2})
seen = {"gated": 0, "dense": 0}
for i in range(n):
    s = ds[i * 7919]; f = s["feats"]; st = s["harm_stats"]
    a = f["adj_state"].numpy(); sec = np.asarray(f[C_SEC.full_name])
    vals = {int(k): int(v) for k, v in zip(*np.unique(a, return_counts=True))}
    print(f"{s['example_id']:24s} compact={st['compact']} gated={st['gated']} dense={st['dense_zero']} min_lr={st['min_lr_cb']:.2f} "
          f"max_rg={st['max_rg_n04']:.2f} C_SEC>0={int((sec > 0).sum())} adj={vals}", flush=True)
    assert (a == a.T).all()
    if st["gated"]:
        seen["gated"] += 1
        assert (a == L.ADJ_MASKED).all() and (sec == 0).all(), "gated row still conditioned"
    else:
        assert st["compact"] == 1
        if st["n_blocks"] >= 2:
            assert st["dense_zero"] == 1 and (np.diag(a)[st["n_elig"] > 0 and slice(None)] != L.ADJ_MASKED).any(), "dense-zero not applied"
            seen["dense"] += 1
# mixed-length unit check (red-team N003 #1): compact 80-token chain + straight 30-token chain must be REJECTED
from harmonia_ext.adj_data import row_compact


class _OA:
    def get_annotation_categories(self):
        return list(vars(self))


core = np.random.default_rng(0).normal(0, 2, (80, 3)) * 2.5
tail = np.stack([np.arange(30) * 3.8 + 60, np.zeros(30), np.zeros(30)], 1)
ca = np.concatenate([core, tail])
oa = _OA()
oa.coord = np.stack([ca + [-1.45, 0, 0], ca, ca + [0.5, 1.4, 0]], 1).reshape(-1, 3)
oa.atom_name = np.tile(["N", "CA", "C"], 110)
oa.occupancy = np.ones(330)
oa.pn_unit_iid = np.repeat(np.array(["A"] * 80 + ["B"] * 30), 3)
ok, ms = row_compact(oa, np.arange(110) * 3, np.ones(110, bool))
assert not ok and ms[-1].get("short_chain") == 30, (ok, ms)
print(seen, "mixed-length row rejected", "SMOKE_GATE_OK")
