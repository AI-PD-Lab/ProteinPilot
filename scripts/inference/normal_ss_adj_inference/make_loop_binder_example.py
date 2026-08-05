#!/usr/bin/env python3
"""Create a fixed-target/fixed-framework loop-binder SS/ADJ inference input."""

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


def atom_line(serial: int, atom: str, chain: str, res_id: int, x: float, y: float, z: float) -> str:
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:^4s} {AA3:>3s} {chain}{res_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2s}\n"
    )


def add_res(lines: list[str], serial: int, chain: str, res_id: int, x: float, y: float, z: float) -> int:
    atoms = {
        "N": (x - 1.25, y, z),
        "CA": (x, y, z),
        "C": (x + 1.25, y, z),
        "O": (x + 1.95, y, z - 0.55),
        "CB": (x, y + 1.25, z + 1.15),
    }
    for atom, coord in atoms.items():
        lines.append(atom_line(serial, atom, chain, res_id, *coord))
        serial += 1
    return serial


def write_input_pdb(path: Path) -> None:
    """Toy PDB with target A1-16 and two fixed framework segments B1-8/B19-26."""
    lines = []
    serial = 1
    # Target beta-like strand along x-axis.
    for i in range(16):
        serial = add_res(lines, serial, "A", i + 1, i * 3.35, -0.6 if i % 2 else 0.6, 0.0)
    lines.append("TER\n")
    # Fixed framework pieces for binder chain B.  There is an intentional gap B9-B18
    # that the contig will regenerate as a loop between these anchors.
    for i in range(8):
        serial = add_res(lines, serial, "B", i + 1, i * 3.35 + 6.0, 12.0, 0.0)
    for i in range(8):
        serial = add_res(lines, serial, "B", i + 19, i * 3.35 + 6.0, 12.0, 7.0)
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

    input_pdb = args.out_dir / "fixed_target_framework_loop_gap.pdb"
    write_input_pdb(input_pdb)

    # Flattened contig order: A1-16, /0, B1-8, 10 designed loop residues, B19-26.
    # Total 42 residues.  The loop segment is positions 25-34 in flattened coords.
    total_len = 42
    ss = torch.full((total_len,), SS_MASKED, dtype=torch.long)
    ss[0:16] = SS_STRAND      # target face
    ss[16:24] = SS_STRAND     # fixed framework anchor 1
    ss[24:34] = SS_LOOP       # generated loop to test loop-aware adjacency
    ss[34:42] = SS_STRAND     # fixed framework anchor 2

    adj = torch.full((total_len, total_len), ADJ_MASKED, dtype=torch.float32)
    # Condition target beta face A4-A13 against generated loop positions 25-34.
    contact(adj, (4, 13), (25, 34))
    adj.fill_diagonal_(ADJ_NO)

    specs = {
        "fixed_target_loop_binder_adj": {
            "dialect": 2,
            "input": str(input_pdb.resolve()),
            "contig": "A1-16,/0,B1-8,10-10,B19-26",
            "length": "42-42",
            "select_fixed_atoms": True,
            "ss_file": save(args.out_dir / "tensors" / "loop_binder.ss.pt", ss),
            "adj_file": save(args.out_dir / "tensors" / "loop_binder.adj.pt", adj),
            "extra": {
                "description": "Fixed target/framework with generated loop adjacency-conditioned to target.",
                "target_span_flat": [4, 13],
                "loop_span_flat": [25, 34],
                "framework_spans_flat": [[17, 24], [35, 42]],
            },
        }
    }
    out_json = args.out_dir / "loop_binder_adj_input.json"
    out_json.write_text(json.dumps(specs, indent=2) + "\n")
    print(out_json)


if __name__ == "__main__":
    main()
