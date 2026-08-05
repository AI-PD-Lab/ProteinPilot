#!/usr/bin/env python3
"""Build a tiny local metadata fixture for SS/ADJ loop-routing smoke tests.

This is not training data. It reuses locally mirrored PDB entries only to verify
that the loop-aware transform and mixed sampler can complete a training step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--pdb_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--n_examples", type=int, default=32)
    parser.add_argument("--dataset_name", default="ss_adj_loop_smoke")
    parser.add_argument("--source", default="local_smoke_only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = pd.read_parquet(
        args.metadata,
        columns=["pdb_id", "assembly_id", "q_pn_unit_iid"],
    ).drop_duplicates(subset=["pdb_id", "assembly_id"])

    records = []
    pdb_dir = Path(args.pdb_dir).resolve()
    for row in source.itertuples(index=False):
        pdb_id = str(row.pdb_id).lower()
        cif_path = pdb_dir / pdb_id[1:3] / f"{pdb_id}.cif.gz"
        if not cif_path.exists():
            continue
        records.append(
            {
                "example_id": f"{{['{args.dataset_name}']}}{{{pdb_id}}}{{{row.assembly_id}}}{{['{row.q_pn_unit_iid}']}}",
                "cif_path": str(cif_path),
                "assembly_id": str(row.assembly_id),
                "pdb_id": pdb_id,
                "q_pn_unit_iid": str(row.q_pn_unit_iid),
                "source": args.source,
                "chain_order": "",
            }
        )
        if len(records) >= args.n_examples:
            break

    if not records:
        raise RuntimeError("No locally mirrored structures matched the metadata")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Wrote {len(records)} smoke-only examples to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
