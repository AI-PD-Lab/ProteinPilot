"""CPU smoke of HarmoniaDataset: real PDB rows -> RFD4 pipeline -> C_SEC + adj_state. Usage: python -m harmonia_ext.tests.smoke_data [n]"""
import sys, time, collections
import numpy as np, torch
from harmonia_ext.adj_data import HarmoniaDataset
from harmonia_ext import adj_labels as L
from rfproteina.conditions import C_SEC

n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
modes = collections.Counter()
for forced in (None, "full"):
    ds = HarmoniaDataset(rows="all", split="train", seed=3, forced_mode=forced, transform={"crop_size": 256})
    for i in range(n if forced is None else 4):
        t = time.time(); s = ds[i * 997]; f = s["feats"]
        a = f["adj_state"].numpy(); sec = f.get(C_SEC.full_name)
        st = s["harm_stats"]; modes[st["mode"]] += 1
        assert a.shape[0] == a.shape[1] == f["residue_type"].shape[0]
        assert (a == a.T).all() and (np.diag(a) == L.ADJ_MASKED).all()
        nsec = None if sec is None else int((sec > 0).sum())
        vals = dict(zip(*np.unique(a, return_counts=True)))
        print(f"{s['example_id']:22s} mode={st['mode']:8s} tok={st['n_tokens']} blocks={st['n_blocks']} con={st['n_contact_gt']} non={st['n_noncontact_gt']} "
              f"shown={st['n_shown']} hot={st['n_hot']} cold={st['n_cold']} ss_shown_res={st['n_ss_shown']}/{st['n_res']} C_SEC_tok>0={nsec} "
              f"adj_vals={ {int(k): int(v) for k, v in vals.items()} } {time.time()-t:.1f}s", flush=True)
        if st["mode"] in ("uncond", "adj_only"): assert nsec in (None, 0), nsec
        if st["mode"] in ("ss_only", "full"): assert nsec and nsec > 0
        if st["mode"] in ("uncond", "ss_only"): assert (a == L.ADJ_MASKED).all()
print("modes", dict(modes)); print("SMOKE_DATA_OK")
