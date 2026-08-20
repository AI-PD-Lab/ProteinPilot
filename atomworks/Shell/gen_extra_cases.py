"""Benchmark cases for the things phase 6 actually added: loop adjacency,
ligand adjacency, nucleic-acid adjacency.

The existing v2 suite is protein-only H/E, so it cannot see any of it. These use
adj_pairs (chain+residue selections resolved against the assembled output) rather
than hand-authored matrices -- for a ligand or a DNA duplex, writing the tensor by
hand is impractical and silently order-dependent.

Targets:
  1LMB  lambda repressor / operator DNA  -> protein-NA
  3PTB  trypsin / benzamidine (BEN)      -> protein-ligand
  1STP  streptavidin / biotin (BTN)      -> protein-ligand, larger
  synthetic                              -> loop adjacency (H/E to loop block)
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
PDBDIR = OUT / "pdbs"
TDIR = OUT / "tensors"
for d in (PDBDIR, TDIR):
    d.mkdir(parents=True, exist_ok=True)

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb_io
import biotite.database.rcsb as rcsb

SS_H, SS_E, SS_L, SS_MASK = 0, 1, 2, 3
ADJ_NO, ADJ_YES, ADJ_MASK = 0, 1, 2


NA_RES = ["DA", "DT", "DG", "DC", "A", "U", "G", "C"]


def fetch_pdb(code, drop_chains=(), ligand=None, keep_protein_chains=None):
    """Write a cleaned .pdb with NORMALISED chain ids and residue numbering.

    Necessary because the contig grammar is 'ChainIDStart-Stop': 1LMB names its
    DNA strands "1" and "2", which collide with the residue-range syntax, and a
    ligand must be addressed by chain+resid, not by residue NAME. Normalising here
    means the contig strings below are unambiguous.

    Layout produced:  P.. protein   D/E.. nucleic acid strands   L.. ligand
    """
    arr = pdbx.get_structure(pdbx.CIFFile.read(rcsb.fetch(code, "cif", str(PDBDIR))), model=1)
    arr = arr[~struc.filter_solvent(arr)]
    if drop_chains:
        arr = arr[~np.isin(arr.chain_id, list(drop_chains))]

    is_prot = struc.filter_amino_acids(arr)
    is_na = np.isin(arr.res_name, NA_RES)
    keep = is_prot | is_na
    if ligand is not None:
        keep = keep | (arr.res_name == ligand)
    arr = arr[keep]

    is_prot = struc.filter_amino_acids(arr)
    is_na = np.isin(arr.res_name, NA_RES)
    new_chain = np.array(arr.chain_id, dtype="U4")
    if keep_protein_chains is not None:
        arr = arr[~is_prot | np.isin(arr.chain_id, list(keep_protein_chains))]
        is_prot = struc.filter_amino_acids(arr)
        is_na = np.isin(arr.res_name, NA_RES)
        new_chain = np.array(arr.chain_id, dtype="U4")
    new_chain[is_prot] = "P"
    for letter, c in zip("DE", [c for c in np.unique(arr.chain_id[is_na])]):
        new_chain[is_na & (arr.chain_id == c)] = letter
    if ligand is not None:
        new_chain[arr.res_name == ligand] = "L"
    arr.chain_id = new_chain

    # renumber each chain from 1 so contig ranges are predictable
    new_res = np.array(arr.res_id)
    for c in np.unique(arr.chain_id):
        m = arr.chain_id == c
        sub = arr[m]
        starts = struc.get_residue_starts(sub)
        mapping = {int(sub.res_id[s]): i + 1 for i, s in enumerate(starts)}
        new_res[m] = [mapping[int(r)] for r in sub.res_id]
    arr.res_id = new_res

    p = PDBDIR / f"{code}.pdb"
    f = pdb_io.PDBFile()
    f.set_structure(arr)
    f.write(str(p))
    counts = {str(c): int(struc.get_residue_count(arr[arr.chain_id == c])) for c in np.unique(arr.chain_id)}
    print(f"    {code}: chains {counts}")
    return str(p), arr


specs, manifest = {}, {}


def case(name, spec, mode, desc, target):
    specs[name] = spec
    manifest[name] = {"mode": mode, "description": desc, "target": target}


# ---------------------------------------------------------------- protein-NA
# Fix the DNA duplex, design a 70-residue binder, and require contact with one
# groove face while forbidding contact with the far end of the duplex.
p, arr = fetch_pdb("1LMB", drop_chains=("3", "4"))   # DNA only -> chains D, E
case(
    "na_dna_binder",
    {"input": p, "contig": "70-70,/0,D1-20,/0,E1-20", "select_fixed_atoms": True,
     "adj_pairs": ["A20-45 : B5-15"]},
    "na_adjacency",
    "Designed 70mer required to contact one face of a fixed B-DNA duplex.",
    # token layout: A = 1..70 designed, B = 71..90 (strand D), C = 91..110 (strand E)
    {"pair_kinds": ["na-protein"], "contact_blocks": [[[20, 45], [75, 85]]]},
)
case(
    "na_dna_binder_strict_no",
    {"input": p, "contig": "70-70,/0,D1-20,/0,E1-20", "select_fixed_atoms": True,
     "adj_pairs": ["A20-45 : B5-15", "A55-70 !: C5-15"]},
    "na_adjacency_negative",
    "As above plus an explicit NON-contact to the second strand's far end.",
    {"pair_kinds": ["na-protein"], "has_negatives": True,
     "contact_blocks": [[[20, 45], [75, 85]]],
     "non_contact_blocks": [[[55, 70], [95, 105]]]},
)

# ------------------------------------------------------------ protein-ligand
for code, lig, n_des in (("3PTB", "BEN", 60), ("1STP", "BTN", 60)):
    p, arr = fetch_pdb(code, ligand=lig, keep_protein_chains=())
    n_lig_atoms = int((arr.res_name == lig).sum())
    print(f"    {code} {lig}: {n_lig_atoms} atoms -> {n_lig_atoms} tokens once atomised")
    case(
        f"ligand_{code.lower()}_binder",
        {"input": p, "contig": f"{n_des}-{n_des},/0,L1", "select_fixed_atoms": True,
         "adj_pairs": [f"A15-35 : B1-1"]},
        "ligand_adjacency",
        f"Designed {n_des}mer required to contact the fixed {lig} ligand of {code}.",
        # the ligand is ATOMISED: n_lig_atoms tokens starting right after the
        # designed chain, so the block spans 61 .. 60+n_lig_atoms.
        {"pair_kinds": ["ligand-protein"],
         "contact_blocks": [[[15, 35], [n_des + 1, n_des + n_lig_atoms]]]},
    )

# --------------------------------------------------------------- loop adjacency
# Pure de novo: helix - loop - helix, with the LOOP required to contact helix 1.
# Loop pairs need >=3 close pairs by the label rule, so this is a real test of
# whether loop conditioning is honoured rather than incidentally satisfied.
L = 60
ss = np.full(L, SS_L, dtype=np.float32)
ss[2:20] = SS_H
ss[40:58] = SS_H
adj = np.full((L, L), ADJ_MASK, dtype=np.float32)
def set_pair(a, b, v):
    adj[a[0]-1:a[1], b[0]-1:b[1]] = v
    adj[b[0]-1:b[1], a[0]-1:a[1]] = v
set_pair((3, 20), (41, 58), ADJ_YES)      # helix-helix packing
set_pair((21, 40), (3, 20), ADJ_YES)      # LOOP must contact helix 1
np.fill_diagonal(adj, ADJ_NO)
ss_p, adj_p = str(TDIR / "loop_adj.ss.pt"), str(TDIR / "loop_adj.adj.pt")
torch.save(torch.from_numpy(ss), ss_p)
torch.save(torch.from_numpy(adj), adj_p)
case(
    "loop_adjacency",
    {"length": f"{L}-{L}", "ss_file": ss_p, "adj_file": adj_p},
    "loop_adjacency",
    "Helix-loop-helix where the LOOP block must contact helix 1.",
    {"ss_labels": ss.tolist(),
     "contact_blocks": [((3, 20), (41, 58)), ((21, 40), (3, 20))]},
)

json.dump(specs, open(OUT / "extra_inputs.json", "w"), indent=1)
json.dump(manifest, open(OUT / "extra_manifest.json", "w"), indent=1)
print(f"  wrote {len(specs)} cases -> {OUT/'extra_inputs.json'}")
for k, v in manifest.items():
    print(f"    {k:28s} {v['mode']}")
