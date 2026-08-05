#!/usr/bin/env python3
"""Create a fixed beta-strand target plus adjacency-conditioned beta binder input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

SS_HELIX = 0
SS_STRAND = 1
SS_LOOP = 2
SS_MASKED = 3
ADJ_NO = 0
ADJ_YES = 1
ADJ_MASKED = 2

AA3 = "ALA"


def pdb_atom_line(serial: int, atom: str, res_id: int, x: float, y: float, z: float) -> str:
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:^4s} {AA3:>3s} A{res_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2s}\n"
    )


def write_target_pdb(path: Path, n_res: int = 16) -> None:
    """Write a simple extended beta-like target chain A."""
    lines = []
    serial = 1
    for i in range(n_res):
        res_id = i + 1
        x = i * 3.35
        y = 0.55 if i % 2 else -0.55
        z = 0.0
        atoms = {
            "N": (x - 1.25, y, z),
            "CA": (x, y, z),
            "C": (x + 1.25, y, z),
            "O": (x + 1.95, y, z - 0.55),
            "CB": (x, -y * 2.2, z + 1.15),
        }
        for atom, coord in atoms.items():
            lines.append(pdb_atom_line(serial, atom, res_id, *coord))
            serial += 1
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))


def save(path: Path, tensor: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    return str(path.resolve())


def contact(adj: torch.Tensor, a: tuple[int, int], b: tuple[int, int]) -> None:
    a0, a1 = a[0] - 1, a[1]
    b0, b1 = b[0] - 1, b[1]
    adj[a0:a1, b0:b1] = ADJ_YES
    adj[b0:b1, a0:a1] = ADJ_YES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = args.out_dir / "tensors"
    target_pdb = args.out_dir / "fixed_beta_target_A.pdb"
    write_target_pdb(target_pdb)

    target_len = 16
    binder_len = 24
    total_len = target_len + binder_len

    ss = torch.full((total_len,), SS_MASKED, dtype=torch.long)
    ss[0:target_len] = SS_STRAND
    # Binder chain is positions 17-40 in the flattened token order.  Its central
    # 12 residues are conditioned as a strand; termini remain free/masked.
    ss[20:32] = SS_STRAND

    adj = torch.full((total_len, total_len), ADJ_MASKED, dtype=torch.float32)
    # Force the target beta face to be adjacent to the designed binder strand.
    contact(adj, (3, 14), (21, 32))
    adj.fill_diagonal_(ADJ_NO)

    specs = {
        "fixed_target_beta_binder_adj": {
            "dialect": 2,
            "input": str(target_pdb.resolve()),
            "contig": "A1-16,/0,24-24",
            "length": "40-40",
            "select_fixed_atoms": True,
            "ss_file": save(tensor_dir / "fixed_target_beta_binder.ss.pt", ss),
            "adj_file": save(tensor_dir / "fixed_target_beta_binder.adj.pt", adj),
            "extra": {
                "target_chain": "A",
                "target_residue_span_flat": [3, 14],
                "binder_residue_span_flat": [21, 32],
                "target_len": target_len,
                "binder_len": binder_len,
                "description": "Fixed beta-strand target plus adjacency-conditioned beta-strand binder.",
            },
        }
    }

    out_json = args.out_dir / "beta_binder_adj_input.json"
    out_json.write_text(json.dumps(specs, indent=2) + "\n")
    print(out_json)


if __name__ == "__main__":
    main()
