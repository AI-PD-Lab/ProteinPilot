#!/usr/bin/env python3
"""Create leakage-free SS/ADJ loop splits and filtered standard metadata."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop_csv", required=True)
    parser.add_argument("--loop_train_csv", required=True)
    parser.add_argument("--loop_val_csv", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--filter_parquet",
        action="append",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        default=[],
        help="Stream-filter a standard metadata parquet to exclude validation PDBs.",
    )
    return parser.parse_args()


def _stable_key(seed: int, source: str, pdb_id: str) -> str:
    value = f"{seed}:{source}:{pdb_id}".encode("ascii")
    return hashlib.sha256(value).hexdigest()


def split_loop_metadata(
    loop_csv: Path,
    train_csv: Path,
    val_csv: Path,
    val_fraction: float,
    seed: int,
) -> set[str]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")

    data = pd.read_csv(loop_csv, dtype={"pdb_id": str, "cluster": str})
    data["pdb_id"] = data["pdb_id"].str.lower()
    if data["cluster"].duplicated().any():
        raise RuntimeError("Expected cluster-deduplicated loop metadata")

    val_pdbs: set[str] = set()
    for source, source_rows in data.groupby("source", sort=True):
        pdb_sizes = source_rows.groupby("pdb_id").size()
        ordered_pdbs = sorted(
            pdb_sizes.index,
            key=lambda pdb_id: _stable_key(seed, str(source), str(pdb_id)),
        )
        target_rows = max(1, round(len(source_rows) * val_fraction))
        selected_rows = 0
        for pdb_id in ordered_pdbs:
            if selected_rows >= target_rows:
                break
            val_pdbs.add(str(pdb_id))
            selected_rows += int(pdb_sizes[pdb_id])

    is_val = data["pdb_id"].isin(val_pdbs)
    train = data[~is_val].copy()
    val = data[is_val].copy()

    if set(train["pdb_id"]) & set(val["pdb_id"]):
        raise RuntimeError("PDB leakage between loop train and validation splits")
    if set(train["cluster"]) & set(val["cluster"]):
        raise RuntimeError("Cluster leakage between loop train and validation splits")

    train_csv.parent.mkdir(parents=True, exist_ok=True)
    val_csv.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_csv, index=False)
    val.to_csv(val_csv, index=False)

    print(f"Loop train: {len(train)} rows, {train.pdb_id.nunique()} PDBs")
    print(f"Loop val:   {len(val)} rows, {val.pdb_id.nunique()} PDBs")
    print("Validation source counts:")
    print(val.groupby("source").size().to_string())
    return val_pdbs


def filter_parquet(input_path: Path, output_path: Path, excluded_pdbs: set[str]) -> None:
    source = pq.ParquetFile(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    excluded = pa.array(sorted(excluded_pdbs), type=pa.string())
    writer = pq.ParquetWriter(output_path, source.schema_arrow, compression="zstd")
    input_rows = 0
    output_rows = 0
    try:
        for batch in source.iter_batches(batch_size=65_536):
            table = pa.Table.from_batches([batch], schema=source.schema_arrow)
            pdb_ids = pc.utf8_lower(pc.cast(table["pdb_id"], pa.string()))
            keep = pc.invert(pc.is_in(pdb_ids, value_set=excluded))
            filtered = table.filter(keep)
            writer.write_table(filtered)
            input_rows += len(table)
            output_rows += len(filtered)
    finally:
        writer.close()

    print(
        f"Filtered {input_path.name}: {input_rows} -> {output_rows} rows "
        f"({input_rows - output_rows} held-out rows removed)"
    )


def main() -> int:
    args = parse_args()
    val_pdbs = split_loop_metadata(
        Path(args.loop_csv),
        Path(args.loop_train_csv),
        Path(args.loop_val_csv),
        args.val_fraction,
        args.seed,
    )
    for input_name, output_name in args.filter_parquet:
        filter_parquet(Path(input_name), Path(output_name), val_pdbs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
