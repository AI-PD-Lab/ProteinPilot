#!/usr/bin/env python3
"""Convert curated loop-source records into RFD3 training metadata.

The source builder identifies relevant PDBs and chains. This stage joins those
records to AtomWorks pn-unit metadata, keeps pn units on requested chains,
deduplicates by the existing PDB training cluster, and emits rows accepted by
GenericDFParser.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", required=True)
    parser.add_argument("--pn_units_parquet", required=True)
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--max_resolution", type=float, default=3.0)
    return parser.parse_args()


def _chain_matches(chain_order: object, pn_unit_iid: object) -> bool:
    if not isinstance(chain_order, str) or not chain_order.strip():
        return True
    allowed = {chain.strip() for chain in chain_order.split(",") if chain.strip()}
    pn_chain = str(pn_unit_iid).split("_", 1)[0]
    return pn_chain in allowed


def main() -> int:
    args = parse_args()
    source = pd.read_csv(args.source_csv, dtype={"pdb_id": str})
    source["pdb_id"] = source["pdb_id"].str.lower()

    pn_units = pd.read_parquet(
        args.pn_units_parquet,
        columns=[
            "pdb_id",
            "assembly_id",
            "q_pn_unit_iid",
            "q_pn_unit_type",
            "q_pn_unit_is_polymer",
            "cluster",
            "resolution",
        ],
    )
    pn_units["pdb_id"] = pn_units["pdb_id"].astype(str).str.lower()

    merged = pn_units.merge(
        source[["pdb_id", "source", "chain_order", "cif_path"]],
        on="pdb_id",
        how="inner",
    )
    merged = merged[
        merged.apply(
            lambda row: _chain_matches(row["chain_order"], row["q_pn_unit_iid"]),
            axis=1,
        )
    ].copy()
    merged = merged[
        merged["q_pn_unit_is_polymer"].astype(bool)
        & merged["q_pn_unit_type"].isin([0, 5, 6])
        & merged["cluster"].notna()
        & merged["resolution"].le(args.max_resolution)
    ].copy()
    if merged.empty:
        raise RuntimeError("No protein pn units matched the curated loop records")

    merged.sort_values(["resolution", "pdb_id"], inplace=True, na_position="last")
    merged.drop_duplicates(subset=["cluster"], keep="first", inplace=True)

    pdb_dir = Path(args.pdb_dir).resolve()

    def choose_path(row: pd.Series) -> str | None:
        mirror_path = pdb_dir / row.pdb_id[1:3] / f"{row.pdb_id}.cif.gz"
        if mirror_path.exists():
            return str(mirror_path)
        source_path = Path(str(row.cif_path))
        return str(source_path.resolve()) if source_path.exists() else None

    merged["cif_path"] = merged.apply(choose_path, axis=1)
    merged = merged[merged["cif_path"].notna()].copy()
    merged["assembly_id"] = merged["assembly_id"].astype(str)
    merged["example_id"] = merged.apply(
        lambda row: (
            f"{{['ss_adj_loop', '{row.source}']}}{{{row.pdb_id}}}"
            f"{{{row.assembly_id}}}{{['{row.q_pn_unit_iid}']}}"
        ),
        axis=1,
    )

    columns = [
        "example_id",
        "cif_path",
        "assembly_id",
        "pdb_id",
        "source",
        "chain_order",
        "q_pn_unit_iid",
        "cluster",
        "resolution",
    ]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged[columns].to_csv(out_csv, index=False)
    print(f"Wrote {len(merged)} cluster-deduplicated loop examples to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
