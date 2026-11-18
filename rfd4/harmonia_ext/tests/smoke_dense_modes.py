"""CPU smoke of R3 dense_modes: adj_only rows get dense-zero when 'adj_only' is in dense_modes, and never with the R2 default ("full",).
Usage: python -m harmonia_ext.tests.smoke_dense_modes [n]"""
import sys
from harmonia_ext.adj_data import HarmoniaDataset

n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
T = {"crop_size": None, "crop_size_range": [0.9, 1.0], "max_crop_size": 256, "p_no_subcrop": 0.2}
for modes, want in ((("full",), 0), (("adj_only", "full"), 1)):
    ds = HarmoniaDataset(rows="all", split="train", seed=3, forced_mode="adj_only", gate_compact=True, p_dense_zero=1.0, dense_modes=modes, transform=T)
    hit = 0
    for i in range(n):
        st = ds[i * 7919]["harm_stats"]
        if st["gated"] or st["n_blocks"] < 2:
            assert st["dense_zero"] == 0
            continue
        assert st["dense_zero"] == want, (modes, st)
        hit += 1
    assert hit > 0, "no conditioned adj_only row sampled"
    print(f"dense_modes={modes}: {hit} conditioned adj_only rows, dense_zero={want} on all -> OK", flush=True)
print("SMOKE_DENSE_MODES PASSED")
