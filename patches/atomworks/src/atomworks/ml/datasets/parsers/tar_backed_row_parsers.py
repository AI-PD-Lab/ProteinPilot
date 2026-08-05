"""Tar-backed MetadataRowParsers for the PDB mirror.

Avoids the inode explosion of extracting ~255K individual .cif.gz files by
reading directly out of a single tar archive via a precomputed offset index
(scripts/build_pdb_mirror_tar_index.py in the RFD3-Training repo), following
the same "no extraction to disk" pattern Training_Triad's AFDB pipeline uses
for its swissprot tar (scripts/afdb_pipeline/build_afdb_tar_index_and_manifest.py).

atomworks.io.parser.parse() accepts io.StringIO directly (see its `filename`
type hint) -- no changes to the core parsing pipeline are needed. Only
MetadataRowParser.required_schema needs relaxing, since the base class
otherwise enforces `path: Path` via isinstance and would reject a buffer.
"""

from __future__ import annotations

import gzip
import io
import os
import tarfile
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from atomworks.ml.datasets.parsers import MetadataRowParser

_INDEX_CACHE: dict[str, dict[str, tuple[int, int]]] = {}
_TAR_HANDLE_CACHE: dict[tuple[str, int], tarfile.TarFile] = {}


def _load_index(index_path: str) -> dict[str, tuple[int, int]]:
    """Load and cache the (pdb_id -> (offset_data, size)) index, once per process."""
    if index_path not in _INDEX_CACHE:
        df = pd.read_parquet(index_path, columns=["pdb_id", "offset_data", "size"])
        _INDEX_CACHE[index_path] = dict(
            zip(df["pdb_id"], zip(df["offset_data"], df["size"], strict=True), strict=True)
        )
    return _INDEX_CACHE[index_path]


def _get_tar(tar_path: str) -> tarfile.TarFile:
    """Lazily open (and cache) a tar handle, reopening per process/worker PID."""
    key = (tar_path, os.getpid())
    tf = _TAR_HANDLE_CACHE.get(key)
    if tf is None:
        tf = tarfile.open(tar_path, "r:")
        _TAR_HANDLE_CACHE[key] = tf
    return tf


def _read_cif_text(tar_path: str, index_path: str, pdb_id: str) -> io.StringIO:
    """O(1) random-access read of one gzipped CIF member's decompressed text.

    Seeks directly into the (uncompressed) tar's underlying file object at the
    precomputed byte offset -- no tarfile member scan, no disk extraction.
    """
    entry = _load_index(index_path).get(pdb_id)
    if entry is None:
        raise FileNotFoundError(f"pdb_id {pdb_id!r} not found in tar index {index_path}")
    offset_data, size = entry
    fileobj = _get_tar(tar_path).fileobj
    fileobj.seek(offset_data)
    raw = fileobj.read(size)
    text = gzip.decompress(raw).decode("utf-8")
    return io.StringIO(text)


class _TarBackedSchemaMixin:
    # Relax the base schema: "path" may be a real Path OR an in-memory StringIO
    # buffer (atomworks.io.parser.parse() accepts either -- see its type hint).
    required_schema: ClassVar[dict[str, type]] = {
        "example_id": str,
        "path": (Path, io.StringIO),
        "extra_info": dict,
    }


class TarBackedPNUnitsDFParser(_TarBackedSchemaMixin, MetadataRowParser):
    """Tar-backed equivalent of atomworks.ml.datasets.parsers.PNUnitsDFParser."""

    def __init__(self, tar_path: str, index_path: str, **_legacy_unused_kwargs):
        # **_legacy_unused_kwargs absorbs base_dir/file_extension/path_template etc.
        # in case a Hydra override merges into rather than replaces the parent
        # dataset_parser node (recursive dict merge is the default); this class
        # never uses those, only tar_path/index_path.
        self.tar_path = tar_path
        self.index_path = index_path

    def _parse(self, row: pd.Series) -> dict[str, Any]:
        pdb_id = str(row["pdb_id"]).lower()
        buf = _read_cif_text(self.tar_path, self.index_path, pdb_id)
        return {
            "example_id": row["example_id"],
            "path": buf,
            "pdb_id": pdb_id,
            "assembly_id": row["assembly_id"],
            "query_pn_unit_iids": [row["q_pn_unit_iid"]],
            "extra_info": row.to_dict(),
        }


class TarBackedInterfacesDFParser(_TarBackedSchemaMixin, MetadataRowParser):
    """Tar-backed equivalent of atomworks.ml.datasets.parsers.InterfacesDFParser."""

    def __init__(self, tar_path: str, index_path: str, **_legacy_unused_kwargs):
        self.tar_path = tar_path
        self.index_path = index_path

    def _parse(self, row: pd.Series) -> dict[str, Any]:
        pdb_id = str(row["pdb_id"]).lower()
        buf = _read_cif_text(self.tar_path, self.index_path, pdb_id)
        return {
            "example_id": row["example_id"],
            "path": buf,
            "pdb_id": pdb_id,
            "assembly_id": row["assembly_id"],
            "query_pn_unit_iids": [row["pn_unit_1_iid"], row["pn_unit_2_iid"]],
            "extra_info": row.to_dict(),
        }
