#!/usr/bin/env python3
"""Analyze fixed-target beta binder adjacency-conditioned outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdbx import CIFFile, get_structure


def load_cif(path: Path):
    with gzip.open(path, "rt") as handle:
        return get_structure(CIFFile.read(handle), model=1)


def residue_cb_ca_coords(protein):
    starts = struc.get_residue_starts(protein)
    coords = []
    chain_ids = []
    res_ids = []
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
        chain_ids.append(str(res.chain_id[0]))
        res_ids.append(int(res.res_id[0]))
    return np.asarray(coords, dtype=float), np.asarray(chain_ids), np.asarray(res_ids)


def chain_spans(chain_ids):
    spans = []
    if len(chain_ids) == 0:
        return spans
    start = 0
    for i in range(1, len(chain_ids)):
        if chain_ids[i] != chain_ids[start]:
            spans.append((chain_ids[start], start, i))
            start = i
    spans.append((chain_ids[start], start, len(chain_ids)))
    return spans


def min_block_distance(coords, a0, a1, b0, b1):
    a = coords[a0:a1]
    b = coords[b0:b1]
    if len(a) == 0 or len(b) == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return math.nan
    return float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=6.0)
    args = parser.parse_args()

    designs = sorted((args.out_root / "designs").glob("*.cif.gz"))
    rows = []
    for cif in designs:
        atom_array = load_cif(cif)
        protein = atom_array[struc.filter_amino_acids(atom_array)]
        if protein.array_length() == 0:
            continue
        sse = struc.annotate_sse(protein)
        coords, chain_ids, res_ids = residue_cb_ca_coords(protein)
        spans = chain_spans(chain_ids)
        if len(spans) < 2:
            continue
        target_chain, t0, t1 = spans[0]
        binder_chain, b0, b1 = spans[1]
        # The generator used a 16-residue target and a 24-residue binder.
        target_site = (t0 + 2, min(t0 + 14, t1))
        binder_strand = (b0 + 4, min(b0 + 16, b1))
        dmin = min_block_distance(coords, *target_site, *binder_strand)
        binder_sse = sse[binder_strand[0] : binder_strand[1]]
        target_sse = sse[target_site[0] : target_site[1]]
        binder_beta_frac = float(np.mean(binder_sse == "b")) if len(binder_sse) else math.nan
        target_beta_frac = float(np.mean(target_sse == "b")) if len(target_sse) else math.nan
        rows.append(
            {
                "file": cif.name,
                "n_res": len(sse),
                "target_chain": target_chain,
                "binder_chain": binder_chain,
                "target_len": t1 - t0,
                "binder_len": b1 - b0,
                "target_site_min_res": target_site[0] + 1,
                "target_site_max_res": target_site[1],
                "binder_strand_min_res": binder_strand[0] + 1,
                "binder_strand_max_res": binder_strand[1],
                "target_beta_frac": target_beta_frac,
                "binder_beta_frac": binder_beta_frac,
                "interface_min_cb_ca_dist": dmin,
                "adjacency_recovered": float(dmin <= args.threshold) if dmin == dmin else math.nan,
            }
        )

    analysis_dir = args.out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "beta_binder_metrics.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def mean(key):
        vals = np.asarray([float(r[key]) for r in rows if r[key] == r[key]], dtype=float)
        return float(vals.mean()) if len(vals) else math.nan

    print(f"designs_analyzed={len(rows)}")
    print(f"csv={csv_path}")
    print(f"adjacency_recovery_mean={mean('adjacency_recovered'):.3f}")
    print(f"interface_min_cb_ca_dist_mean={mean('interface_min_cb_ca_dist'):.3f}")
    print(f"binder_beta_frac_mean={mean('binder_beta_frac'):.3f}")
    print(f"target_beta_frac_mean={mean('target_beta_frac'):.3f}")


if __name__ == "__main__":
    main()
