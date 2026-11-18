#!/usr/bin/env python3
"""Create a small, normal RFdiffusion3 SS/ADJ inference input JSON."""

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


def contact(adj: torch.Tensor, a: tuple[int, int], b: tuple[int, int]) -> None:
    a0, a1 = a[0] - 1, a[1]
    b0, b1 = b[0] - 1, b[1]
    adj[a0:a1, b0:b1] = ADJ_YES
    adj[b0:b1, a0:a1] = ADJ_YES


def save(path: Path, tensor: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    tensor_dir = args.out_dir / "tensors"
    specs = {}

    specs["global_beta_fraction"] = {
        "dialect": 2,
        "length": "120",
        "ss_ratio": 0.25,
    }

    length = 100
    ss = torch.full((length,), SS_LOOP, dtype=torch.long)
    ss[4:27] = SS_HELIX
    ss[37:60] = SS_HELIX
    ss[70:93] = SS_HELIX
    adj = torch.full((length, length), ADJ_NO, dtype=torch.float32)
    contact(adj, (5, 27), (38, 60))
    contact(adj, (38, 60), (71, 93))
    contact(adj, (5, 27), (71, 93))
    adj.fill_diagonal_(ADJ_NO)
    specs["helix_bundle_full_ss_adj"] = {
        "dialect": 2,
        "length": str(length),
        "ss_file": save(tensor_dir / "helix_bundle.ss.pt", ss),
        "adj_file": save(tensor_dir / "helix_bundle.adj.pt", adj),
    }

    length = 96
    partial_ss = torch.full((length,), SS_MASKED, dtype=torch.long)
    partial_ss[7:18] = SS_STRAND
    partial_ss[31:42] = SS_STRAND
    partial_adj = torch.full((length, length), ADJ_MASKED, dtype=torch.float32)
    contact(partial_adj, (8, 18), (32, 42))
    partial_adj.fill_diagonal_(ADJ_NO)
    specs["partial_beta_binding_site"] = {
        "dialect": 2,
        "length": str(length),
        "ss_file": save(tensor_dir / "partial_beta_site.ss.pt", partial_ss),
        "adj_file": save(tensor_dir / "partial_beta_site.adj.pt", partial_adj),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "ss_adj_example_inputs.json"
    out_json.write_text(json.dumps(specs, indent=2) + "\n")
    print(out_json)


if __name__ == "__main__":
    main()
