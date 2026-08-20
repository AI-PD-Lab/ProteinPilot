"""Non-degenerate loop-adjacency cases.

The first loop case asked the loop block (21-40) to contact helix 1 (3-20). Those
blocks are SEQUENCE-ADJACENT -- residue 20 and 21 are covalently bonded -- so the
constraint is satisfied at ~1.3 A no matter what the model does. Both checkpoints
scored exactly 1.000 because the case is incapable of failing. It measured nothing.

Here the loop must reach a helix 32+ residues away in sequence, which requires the
model to actually fold the loop against a distant element, and the strict variant
additionally forbids a contact that a compact 3-helix bundle would otherwise make
by default -- so the negative is not free either.

Layout (L=110):
   3- 25  helix 1
  26- 49  LOOP (the block under test)
  50- 72  helix 2
  73- 76  loop
  77-100  helix 3
 101-110  loop
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
TDIR = OUT / "tensors"
TDIR.mkdir(parents=True, exist_ok=True)

SS_H, SS_E, SS_L, SS_MASK = 0, 1, 2, 3
ADJ_NO, ADJ_YES, ADJ_MASK = 0, 1, 2

L = 110
ss = np.full(L, SS_L, dtype=np.float32)
ss[2:25] = SS_H      # helix 1  (residues 3-25)
ss[49:72] = SS_H     # helix 2  (residues 50-72)
ss[76:100] = SS_H    # helix 3  (residues 77-100)

H1, LOOP, H2, H3 = (3, 25), (26, 49), (50, 72), (77, 100)


def build(pairs):
    adj = np.full((L, L), ADJ_MASK, dtype=np.float32)
    for (a, b, v) in pairs:
        adj[a[0] - 1:a[1], b[0] - 1:b[1]] = v
        adj[b[0] - 1:b[1], a[0] - 1:a[1]] = v
    np.fill_diagonal(adj, ADJ_NO)
    return adj


specs, manifest = {}, {}


def emit(name, pairs, desc, target):
    adj = build(pairs)
    sp, ap = str(TDIR / f"{name}.ss.pt"), str(TDIR / f"{name}.adj.pt")
    torch.save(torch.from_numpy(ss), sp)
    torch.save(torch.from_numpy(adj), ap)
    specs[name] = {"length": f"{L}-{L}", "ss_file": sp, "adj_file": ap}
    manifest[name] = {"mode": "loop_adjacency", "description": desc,
                      "target": {"ss_labels": ss.tolist(), **target}}


# LOOP must contact helix 3 -- 32 residues away in sequence, so nothing about the
# chain topology grants it. Helix1-helix2 contact is included to keep the fold
# well-posed rather than leaving the bundle underdetermined.
emit(
    "loop_adj_distal",
    [(LOOP, H3, ADJ_YES), (H1, H2, ADJ_YES)],
    "Loop block must contact a helix 32 residues away in sequence.",
    {"contact_blocks": [[list(LOOP), list(H3)], [list(H1), list(H2)]]},
)

# Same, plus an explicit non-contact that a compact 3-helix bundle would otherwise
# satisfy by default: helix 1 must stay away from helix 3.
emit(
    "loop_adj_distal_strict_no",
    [(LOOP, H3, ADJ_YES), (H1, H2, ADJ_YES), (H1, H3, ADJ_NO)],
    "As above plus helix1 forbidden from contacting helix3.",
    {"contact_blocks": [[list(LOOP), list(H3)], [list(H1), list(H2)]],
     "non_contact_blocks": [[list(H1), list(H3)]],
     "has_negatives": True},
)

json.dump(specs, open(OUT / "loop_inputs.json", "w"), indent=1)
json.dump(manifest, open(OUT / "loop_manifest.json", "w"), indent=1)
print(f"  wrote {len(specs)} cases -> {OUT/'loop_inputs.json'}")
for k, v in manifest.items():
    t = v["target"]
    print(f"    {k:28s} contacts={t['contact_blocks']}  neg={t.get('non_contact_blocks')}")
print(f"  sequence separation LOOP->H3: {H3[0]-LOOP[1]} residues (previous case: 0, bonded)")
