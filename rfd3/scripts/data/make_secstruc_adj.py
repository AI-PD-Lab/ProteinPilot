#!/usr/bin/env python3
"""
make_secstruc_adj.py — RFD1-compatible SS/ADJ tensor generator.

Reads a PDB/mmCIF file and writes per-residue secondary structure and
block-adjacency tensors using exact RFD1 H/E contact semantics and loadable by the RFD1 fold-conditioned
checkpoint without modification.

Ground truth tensor formats (verified against RFD1 example files):
    SS  tensor: [L] float32, values {0.=helix, 1.=strand, 2.=loop}
    ADJ tensor: [L,L] float32, binary {0.,1.}, symmetric, diagonal=0.

Without --loop_adj (default):
    Loop rows/cols are all 0.0  ← RFD1-compatible
With --loop_adj:
    Loop-mediated contacts included (Pipeline 2, antibody/enzyme active sites)

Usage:
    # Standard (RFD1-compatible)
    python scripts/make_secstruc_adj.py input.pdb --out_dir /tmp

    # Loop-aware (Pipeline 2)
    python scripts/make_secstruc_adj.py input.pdb --out_dir /tmp --loop_adj

    # Specific chains
    python scripts/make_secstruc_adj.py input.cif --out_dir /tmp --chains A B

    # Save as .pt (PyTorch) instead of .npy
    python scripts/make_secstruc_adj.py input.pdb --out_dir /tmp --format pt

    # Print summary only, no file output
    python scripts/make_secstruc_adj.py input.pdb --dry_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate RFD1-compatible SS + ADJ tensors from a structure file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[0],
    )
    p.add_argument("input", help="Path to PDB or mmCIF input file")
    p.add_argument(
        "--out_dir",
        default=".",
        help="Output directory (default: current directory)",
    )
    p.add_argument(
        "--prefix",
        default=None,
        help="Output file prefix. Defaults to input file stem.",
    )
    p.add_argument(
        "--chains",
        nargs="+",
        default=None,
        metavar="CHAIN_ID",
        help="Chain IDs to include (in order). Default: all chains in appearance order.",
    )
    p.add_argument(
        "--loop_adj",
        action="store_true",
        default=False,
        help="Include loop-mediated contacts (Pipeline 2). Without this flag, loop rows/cols=0 (RFD1-compatible).",
    )
    p.add_argument(
        "--cb_threshold",
        type=float,
        default=6.0,
        metavar="Å",
        help="Reconstructed Cβ distance threshold for H/E block contacts (RFD1 default: 6.0 Å)",
    )
    p.add_argument(
        "--ca_threshold_loop",
        type=float,
        default=6.0,
        metavar="Å",
        help="Cα distance threshold for loop contacts (--loop_adj only, default: 6.0 Å)",
    )
    p.add_argument(
        "--min_loop_pairs",
        type=int,
        default=3,
        metavar="N",
        help="Minimum Cα pairs within threshold to form a loop contact (--loop_adj only, default: 3)",
    )
    p.add_argument(
        "--format",
        choices=["npy", "pt"],
        default="pt",
        help="Output format: 'npy' (numpy .npy) or 'pt' (PyTorch .pt, default)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print summary statistics only, do not write files.",
    )
    return p.parse_args()


def _load_structure(path: str):
    """Load a PDB or mmCIF file into a biotite AtomArray."""
    import gzip
    import warnings

    import biotite.structure.io.pdb as pdb_io
    import biotite.structure.io.pdbx as pdbx_io
    from biotite.structure import AtomArray

    path = Path(path)
    ext = path.suffix.lower()
    is_gz = ext == ".gz"
    if is_gz:
        ext = Path(path.stem).suffix.lower()

    opener = (lambda: gzip.open(path, "rt")) if is_gz else (lambda: open(path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ext in (".cif", ".mmcif"):
            with opener() as fh:
                cif_file = pdbx_io.CIFFile.read(fh)
            struct = pdbx_io.get_structure(cif_file, model=1, altloc="occupancy")
        elif ext in (".pdb", ".ent"):
            with opener() as fh:
                pdb_file = pdb_io.PDBFile.read(fh)
            struct = pdb_io.get_structure(pdb_file, model=1, altloc="occupancy")
        else:
            raise ValueError(
                f"Unrecognized file extension: {ext} (expected .pdb, .cif, .mmcif)"
            )

    if not isinstance(struct, AtomArray):
        raise ValueError(f"Expected AtomArray, got {type(struct)}")
    return struct


def _save(arr: np.ndarray, path: Path, fmt: str) -> None:
    if fmt == "npy":
        np.save(str(path), arr)
    else:
        import torch

        torch.save(torch.from_numpy(arr), str(path))


def main() -> int:
    args = _parse_args()

    from foundry.data.ss_adj import SS_HELIX, SS_LOOP, SS_STRAND, get_ss_adj

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    prefix = args.prefix or input_path.stem
    ext = ".pt" if args.format == "pt" else ".npy"

    print(f"Loading structure: {input_path}")
    struct = _load_structure(str(input_path))

    n_atoms = struct.array_length()
    import biotite.structure as struc

    n_residues = len(struc.get_residue_starts(struct))
    chains = np.unique(struct.chain_id).tolist()
    print(f"  {n_atoms} atoms, {n_residues} residues, chains: {chains}")

    if args.chains:
        # get_adj_matrix_* (unlike get_ss_labels) does not itself apply
        # chain_order filtering to the coordinate structure, so pre-filter
        # here to keep SS labels and ADJ coordinates over the same residues.
        parts = []
        for chain_id in args.chains:
            mask = struct.chain_id == chain_id
            if not mask.any():
                print(f"ERROR: chain '{chain_id}' not found in structure", file=sys.stderr)
                return 1
            parts.append(struct[mask])
        struct = struc.concatenate(parts)

    print(f"Computing SS labels (chain_order={args.chains}) ...")
    adj_kwargs = {"cb_threshold": args.cb_threshold}
    if args.loop_adj:
        adj_kwargs.update(
            ca_threshold_loop=args.ca_threshold_loop,
            min_loop_contact_pairs=args.min_loop_pairs,
        )
    ss, adj = get_ss_adj(
        struct,
        chain_order=None,
        loop_adj=args.loop_adj,
        **adj_kwargs,
    )

    L = len(ss)
    n_helix = int((ss == SS_HELIX).sum())
    n_strand = int((ss == SS_STRAND).sum())
    n_loop = int((ss == SS_LOOP).sum())
    n_adj = int(adj.sum() / 2)  # symmetric, count each pair once

    print(f"\n  SS tensor:  shape={ss.shape}, dtype={ss.dtype}")
    print(f"  Helix:  {n_helix:5d} / {L}  ({100*n_helix/L:.1f}%)")
    print(f"  Strand: {n_strand:5d} / {L}  ({100*n_strand/L:.1f}%)")
    print(f"  Loop:   {n_loop:5d} / {L}  ({100*n_loop/L:.1f}%)")
    print(f"\n  ADJ tensor: shape={adj.shape}, dtype={adj.dtype}")
    print(f"  Contact pairs: {n_adj}")
    print(
        f"  Loop rows/cols zeroed: {'NO (loop_adj=True)' if args.loop_adj else 'YES (RFD1-compatible)'}"
    )

    # Verify RFD1 constraints (when not loop_adj)
    if not args.loop_adj:
        loop_mask = ss == SS_LOOP
        loop_adj_nonzero = adj[loop_mask].any() or adj[:, loop_mask].any()
        if loop_adj_nonzero:
            print(
                "  WARNING: loop rows/cols contain non-zero entries (unexpected)",
                file=sys.stderr,
            )
        else:
            print("  ✓  Loop rows/cols all 0.0 (RFD1-compatible)")

    diag_nonzero = np.diag(adj).any()
    sym_ok = np.allclose(adj, adj.T)
    if diag_nonzero:
        print("  WARNING: diagonal is non-zero", file=sys.stderr)
    if not sym_ok:
        print("  WARNING: ADJ matrix is not symmetric", file=sys.stderr)
    if not diag_nonzero and sym_ok:
        print("  ✓  Diagonal=0, symmetric")

    if args.dry_run:
        print("\nDry run — no files written.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    ss_path = out_dir / f"{prefix}_ss{ext}"
    adj_path = out_dir / f"{prefix}_adj{ext}"

    _save(ss, ss_path, args.format)
    _save(adj, adj_path, args.format)

    print(f"\nWrote: {ss_path}")
    print(f"Wrote: {adj_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
