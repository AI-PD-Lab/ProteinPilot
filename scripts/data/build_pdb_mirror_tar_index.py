"""build_pdb_mirror_tar_index.py — single-pass streaming scan of a PDB mirror tar.

Produces pdb_mirror_tar_index.parquet: (pdb_id, name, offset_data, size) for
O(1) random-access reads by TarBackedPDBParser, without ever extracting
individual files to disk (avoids the per-user file-count quota that a plain
directory-tree copy would hit at full-PDB scale on hnvme/anvme).

Mirrors scripts/afdb_pipeline/build_afdb_tar_index_and_manifest.py's approach
in Training_Triad, applied to the PDB mirror instead of the AFDB swissprot tar.

Usage:
    python build_pdb_mirror_tar_index.py \
        --tar_path /hnvme/workspace/v181bb10-rfd3_ssadj_test/pdb_mirror.tar \
        --out_path /hnvme/workspace/v181bb10-rfd3_ssadj_test/pdb_mirror_tar_index.parquet
"""

from __future__ import annotations

import argparse
import logging
import re

import pandas as pd
import tarfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_PDB_ID_RE = re.compile(r"([0-9a-zA-Z]{4})\.cif\.gz$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tar_path", required=True)
    p.add_argument("--out_path", required=True)
    args = p.parse_args()

    records: list[dict] = []
    n_total = 0

    log.info("Streaming through tar: %s", args.tar_path)
    with tarfile.open(args.tar_path, "r:") as tf:
        for member in tf:
            if not member.isfile():
                continue
            fname = member.name.split("/")[-1]
            m = _PDB_ID_RE.match(fname)
            if not m:
                continue
            pdb_id = m.group(1).lower()
            n_total += 1
            records.append(
                {
                    "pdb_id": pdb_id,
                    "name": member.name,
                    "offset_data": member.offset_data,
                    "size": member.size,
                }
            )
            if n_total % 50000 == 0:
                log.info("  %d entries indexed so far", n_total)

    log.info("Scan complete: %d entries", n_total)
    idx_df = pd.DataFrame(records)
    idx_df.to_parquet(args.out_path, index=False)
    log.info("Wrote index: %s (%d rows)", args.out_path, len(idx_df))


if __name__ == "__main__":
    main()
