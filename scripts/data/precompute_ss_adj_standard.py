#!/usr/bin/env python3
"""
SLURM array job: precompute SS/ADJ for the standard PDB mirror (Pipeline 1).

Each task processes a shard of the PDB metadata CSV, computing per-structure
SS labels and block-adjacency matrices, and writes them directly to LMDB.

Loop rows/cols are always 0.0 (RFD1-compatible, loop_adj_active=False).

Submission example:
    sbatch --array=0-999 scripts/precompute_ss_adj_standard.py \\
        --metadata /data/pdb_metadata.csv \\
        --cif_dir  /data/pdb_mirror \\
        --lmdb_out /data/ss_adj_standard.lmdb \\
        --n_tasks  1000

Environment variables used by SLURM:
    SLURM_ARRAY_TASK_ID   — 0-indexed shard index
    SLURM_ARRAY_TASK_COUNT — total number of tasks
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import lmdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute_ss_adj_std")

LMDB_MAP_SIZE = 500 * 1024**3  # 500 GB
COMMIT_EVERY = 500


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metadata", required=True, help="CSV with 'pdb_id' and 'cif_path' columns"
    )
    p.add_argument(
        "--cif_dir",
        default=None,
        help="Base directory prepended to relative cif_path values",
    )
    p.add_argument("--lmdb_out", required=True, help="Output LMDB directory")
    p.add_argument(
        "--n_tasks",
        type=int,
        default=None,
        help="Override total task count (else uses SLURM_ARRAY_TASK_COUNT)",
    )
    p.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Override task index (else uses SLURM_ARRAY_TASK_ID)",
    )
    p.add_argument(
        "--chain_col",
        default=None,
        help="CSV column with comma-separated chain IDs to process (optional)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-compute and overwrite existing LMDB entries",
    )
    p.add_argument(
        "--cb_threshold",
        type=float,
        default=6.0,
        help="Reconstructed Cβ distance threshold for H/E block contacts (RFD1 default 6.0 Å)",
    )
    return p.parse_args()


def _load_structure(cif_path: Path):
    import biotite.structure.io.pdbx as pdbx_io

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if str(cif_path).endswith(".gz"):
            import gzip
            import io

            with gzip.open(cif_path, "rb") as f:
                raw = f.read()
            cif_file = pdbx_io.CIFFile.read(
                io.StringIO(raw.decode("utf-8", errors="replace"))
            )
        else:
            cif_file = pdbx_io.CIFFile.read(str(cif_path))
        return pdbx_io.get_structure(cif_file, model=1, altloc="occupancy")


def _process_one(pdb_id, cif_path, chain_order, cb_threshold, lmdb_env, overwrite):
    from foundry.data.ss_adj import SS_MASKED, get_ss_adj, pack_adj, pack_ss

    with lmdb_env.begin() as txn:
        if not overwrite and txn.get(f"{pdb_id}_ss".encode()) is not None:
            return "skipped"

    try:
        struct = _load_structure(Path(cif_path))
    except Exception as e:
        return f"load_error: {e}"

    try:
        ss, adj = get_ss_adj(
            struct,
            chain_order=chain_order,
            loop_adj=False,
            cb_threshold=cb_threshold,
        )
    except Exception as e:
        return f"compute_error: {e}"

    # Sanity checks — SS_MASKED must never appear in stored files
    if SS_MASKED in ss:
        return "error: SS_MASKED value in ss array (should never happen)"

    ss_bytes = pack_ss(ss)
    adj_bytes = pack_adj(adj)

    return {"ss": ss_bytes, "adj": adj_bytes, "loop_adj_active": b"\x00"}


def main():
    args = _parse_args()

    # Determine shard index
    task_id = (
        args.task_id
        if args.task_id is not None
        else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    )
    n_tasks = (
        args.n_tasks
        if args.n_tasks is not None
        else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    )

    logger.info(f"Task {task_id}/{n_tasks}")

    # Load metadata CSV
    import pandas as pd

    df = pd.read_csv(args.metadata)
    logger.info(f"Metadata: {len(df)} rows total")

    # Shard
    shard = df.iloc[task_id::n_tasks].copy()
    logger.info(f"Processing {len(shard)} structures in this shard")

    # Open LMDB (writeable, single-process writer per task)
    lmdb_path = Path(args.lmdb_out)
    lmdb_path.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(lmdb_path),
        map_size=LMDB_MAP_SIZE,
        writemap=False,
        sync=True,
    )

    n_written = n_skipped = n_error = 0
    txn = env.begin(write=True)
    written_since_commit = 0

    for row in shard.itertuples():
        pdb_id = str(row.pdb_id)
        cif_path = row.cif_path
        if args.cif_dir:
            cif_path = str(Path(args.cif_dir) / cif_path)

        chain_order = None
        if args.chain_col and hasattr(row, args.chain_col):
            raw_chains = getattr(row, args.chain_col)
            if isinstance(raw_chains, str) and raw_chains.strip():
                chain_order = [c.strip() for c in raw_chains.split(",")]

        result = _process_one(
            pdb_id, cif_path, chain_order, args.cb_threshold, env, args.overwrite
        )

        if result == "skipped":
            n_skipped += 1
            continue
        elif isinstance(result, str):
            logger.warning(f"{pdb_id}: {result}")
            n_error += 1
            continue

        txn.put(f"{pdb_id}_ss".encode(), result["ss"])
        txn.put(f"{pdb_id}_adj".encode(), result["adj"])
        txn.put(f"{pdb_id}_loop_adj_active".encode(), result["loop_adj_active"])
        n_written += 1
        written_since_commit += 1

        if written_since_commit >= COMMIT_EVERY:
            txn.commit()
            txn = env.begin(write=True)
            written_since_commit = 0

    txn.commit()
    env.sync()
    env.close()

    logger.info(f"Done — written: {n_written}  skipped: {n_skipped}  errors: {n_error}")
    return 0 if n_error == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
