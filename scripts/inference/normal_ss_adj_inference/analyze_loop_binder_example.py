#!/usr/bin/env python3
"""Analyze loop-aware fixed-target/fixed-framework binder outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from pathlib import Path

import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdbx import CIFFile, get_structure


def load_cif(path: Path):
    with gzip.open(path, "rt") as handle:
        return get_structure(CIFFile.read(handle), model=1)


def residue_coords_and_sse(atom_array):
    protein = atom_array[struc.filter_amino_acids(atom_array)]
    starts = struc.get_residue_starts(protein)
    coords = []
    chains = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else protein.array_length()
        res = protein[start:end]
        cb = res.coord[res.atom_name == "CB"]
        ca = res.coord[res.atom_name == "CA"]
        if len(cb):
            coords.append(cb[0])
        elif len(ca):
            coords.append(ca[0])
        else:
            coords.append(np.full(3, np.nan))
        chains.append(str(res.chain_id[0]))
    return np.asarray(coords, dtype=float), np.asarray(chains), struc.annotate_sse(protein)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=6.0)
    args = parser.parse_args()

    rows = []
    for cif in sorted((args.out_root / "designs").glob("*.cif.gz")):
        coords, chains, sse = residue_coords_and_sse(load_cif(cif))
        if len(coords) < 42:
            continue
        target = slice(3, 13)   # flat residues 4-13
        loop = slice(24, 34)    # flat residues 25-34
        fw1 = slice(16, 24)
        fw2 = slice(34, 42)
        d = np.linalg.norm(coords[target, None, :] - coords[None, loop, :], axis=-1)
        c = d <= args.threshold
        loop_sse = sse[loop]
        fw_break_1 = np.linalg.norm(coords[23] - coords[24])
        fw_break_2 = np.linalg.norm(coords[33] - coords[34])
        rows.append(
            {
                "file": cif.name,
                "n_res": len(coords),
                "loop_frac_in_loop_span": float(np.mean(loop_sse == "c")),
                "non_loop_frac_in_loop_span": float(np.mean(loop_sse != "c")),
                "interface_min_cb_ca_dist": float(np.min(d)),
                "adjacency_recovered": float(np.min(d) <= args.threshold),
                "n_contacts_6A": int(np.sum(c)),
                "target_res_contacted_6A": int(np.sum(np.any(c, axis=1))),
                "loop_res_contacted_6A": int(np.sum(np.any(c, axis=0))),
                "anchor_gap_1_ca_dist": float(fw_break_1),
                "anchor_gap_2_ca_dist": float(fw_break_2),
                "framework_sse_frac": float(np.mean(np.concatenate([sse[fw1], sse[fw2]]) == "b")),
            }
        )

    analysis_dir = args.out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "loop_binder_metrics.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def mean(key: str) -> float:
        vals = np.asarray([float(r[key]) for r in rows if float(r[key]) == float(r[key])], dtype=float)
        return float(vals.mean()) if len(vals) else math.nan

    print(f"designs_analyzed={len(rows)}")
    print(f"csv={csv_path}")
    for key in [
        "adjacency_recovered",
        "interface_min_cb_ca_dist",
        "n_contacts_6A",
        "target_res_contacted_6A",
        "loop_res_contacted_6A",
        "loop_frac_in_loop_span",
        "framework_sse_frac",
        "anchor_gap_1_ca_dist",
        "anchor_gap_2_ca_dist",
    ]:
        print(f"{key}_mean={mean(key):.3f}")


if __name__ == "__main__":
    main()
