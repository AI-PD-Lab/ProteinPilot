#!/usr/bin/env python3
"""Generate expanded RFdiffusion3 SS/ADJ conditioning benchmark inputs.

This v2 benchmark includes the original de novo modes plus the externally tested
fixed-target beta-binder and loop-binder examples, and additional combined modes
that were missing from the first pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

SS_HELIX = 0
SS_STRAND = 1
SS_LOOP = 2
SS_MASKED = 3
ADJ_NO = 0
ADJ_YES = 1
ADJ_MASKED = 2
AA3 = "ALA"


def _length_value(length: Any) -> int:
    if length is None:
        return 0
    text = str(length)
    if "-" in text:
        a, b = text.split("-", 1)
        return int(b if b else a)
    return int(text)


def _segments(length: int, spans: list[tuple[int, int, int]]) -> torch.Tensor:
    labels = torch.full((length,), SS_LOOP, dtype=torch.long)
    for start, end, label in spans:
        labels[start - 1 : end] = label
    return labels


def _adj(length: int, fill: int = ADJ_NO) -> torch.Tensor:
    adj = torch.full((length, length), fill, dtype=torch.float32)
    adj.fill_diagonal_(ADJ_NO)
    return adj


def _contact(adj: torch.Tensor, a: tuple[int, int], b: tuple[int, int]) -> None:
    a0, a1 = a[0] - 1, a[1]
    b0, b1 = b[0] - 1, b[1]
    adj[a0:a1, b0:b1] = ADJ_YES
    adj[b0:b1, a0:a1] = ADJ_YES


def _no_contact(adj: torch.Tensor, a: tuple[int, int], b: tuple[int, int]) -> None:
    a0, a1 = a[0] - 1, a[1]
    b0, b1 = b[0] - 1, b[1]
    adj[a0:a1, b0:b1] = ADJ_NO
    adj[b0:b1, a0:a1] = ADJ_NO


def _mask_rows_cols(adj: torch.Tensor, start: int, end: int) -> None:
    sl = slice(start - 1, end)
    adj[sl, :] = ADJ_MASKED
    adj[:, sl] = ADJ_MASKED
    adj.fill_diagonal_(ADJ_NO)


def _save_tensor(path: Path, tensor: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    return str(path.resolve())


def _case(
    specs: dict[str, dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    key: str,
    spec: dict[str, Any],
    mode: str,
    description: str,
    target: dict[str, Any] | None = None,
) -> None:
    spec = {"dialect": 2, **spec}
    spec.setdefault("extra", {})
    spec["extra"].update({"mode": mode, "description": description})
    specs[key] = spec
    manifest[key] = {
        "mode": mode,
        "description": description,
        "length": _length_value(spec.get("length", 0)),
        "target": target or {},
    }


def _pdb_atom_line(serial: int, atom: str, chain: str, res_id: int, x: float, y: float, z: float) -> str:
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:^4s} {AA3:>3s} {chain}{res_id:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{20.00:6.2f}          {element:>2s}\n"
    )


def _add_res(lines: list[str], serial: int, chain: str, res_id: int, x: float, y: float, z: float, cb_flip: float = 1.0) -> int:
    atoms = {
        "N": (x - 1.25, y, z),
        "CA": (x, y, z),
        "C": (x + 1.25, y, z),
        "O": (x + 1.95, y, z - 0.55),
        "CB": (x, y + cb_flip * 1.25, z + 1.15),
    }
    for atom, coord in atoms.items():
        lines.append(_pdb_atom_line(serial, atom, chain, res_id, *coord))
        serial += 1
    return serial


def _write_beta_target(path: Path, n_res: int = 16) -> None:
    lines: list[str] = []
    serial = 1
    for i in range(n_res):
        serial = _add_res(lines, serial, "A", i + 1, i * 3.35, 0.55 if i % 2 else -0.55, 0.0, cb_flip=-2.0)
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))


def _write_loop_target_framework(path: Path) -> None:
    lines: list[str] = []
    serial = 1
    for i in range(16):
        serial = _add_res(lines, serial, "A", i + 1, i * 3.35, -0.6 if i % 2 else 0.6, 0.0)
    lines.append("TER\n")
    for i in range(8):
        serial = _add_res(lines, serial, "B", i + 1, i * 3.35 + 6.0, 12.0, 0.0)
    for i in range(8):
        serial = _add_res(lines, serial, "B", i + 19, i * 3.35 + 6.0, 12.0, 7.0)
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))


def _add_fixed_target_beta_binder_cases(specs: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]], out_root: Path) -> None:
    input_dir = out_root / "fixed_target_inputs"
    tensor_dir = out_root / "tensors"
    input_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = input_dir / "fixed_beta_target_A.pdb"
    _write_beta_target(pdb_path)

    target_len = 16
    binder_len = 24
    total_len = target_len + binder_len
    ss = torch.full((total_len,), SS_MASKED, dtype=torch.long)
    ss[0:target_len] = SS_STRAND
    ss[20:32] = SS_STRAND
    adj = _adj(total_len, ADJ_MASKED)
    _contact(adj, (3, 14), (21, 32))
    adj.fill_diagonal_(ADJ_NO)

    strict_adj = adj.clone()
    # Tell the model that binder flanks should not satisfy the target interface.
    _no_contact(strict_adj, (3, 14), (17, 20))
    _no_contact(strict_adj, (3, 14), (33, 40))
    strict_adj.fill_diagonal_(ADJ_NO)

    ss_path = _save_tensor(tensor_dir / "fixed_target_beta_binder.ss.pt", ss)
    adj_path = _save_tensor(tensor_dir / "fixed_target_beta_binder.adj.pt", adj)
    strict_adj_path = _save_tensor(tensor_dir / "fixed_target_beta_binder.strict.adj.pt", strict_adj)
    target = {"ss_labels": ss.tolist(), "contact_blocks": [((3, 14), (21, 32))], "fixed_target": True}

    base_spec = {
        "input": str(pdb_path.resolve()),
        "contig": "A1-16,/0,24-24",
        "length": "40-40",
        "select_fixed_atoms": True,
        "ss_file": ss_path,
        "adj_file": adj_path,
    }
    _case(specs, manifest, "binder_beta_adj", base_spec, "fixed_target_beta_binder", "Fixed beta target plus generated beta binder with local SS+ADJ", target)
    _case(specs, manifest, "binder_beta_adj_ssratio070", {**base_spec, "ss_ratio": 0.70}, "fixed_target_beta_binder_ssratio", "Beta binder local SS+ADJ plus global beta-rich ss_ratio=0.70", {**target, "ss_ratio": 0.70})
    _case(specs, manifest, "binder_beta_adj_ssratio090", {**base_spec, "ss_ratio": 0.90}, "fixed_target_beta_binder_ssratio", "Beta binder local SS+ADJ plus global beta-rich ss_ratio=0.90", {**target, "ss_ratio": 0.90})
    _case(specs, manifest, "binder_beta_adj_strict_no", {**base_spec, "adj_file": strict_adj_path}, "fixed_target_beta_binder_strict_no", "Beta binder with explicit ADJ_NO for target-to-binder-flank negatives", target)


def _write_cdr_framework(path: Path) -> None:
    """Epitope chain A (10 res) + VH framework anchors chain B (B1-8, B21-28, gap at B9-B20 for CDR loop)."""
    lines: list[str] = []
    serial = 1
    for i in range(10):
        serial = _add_res(lines, serial, "A", i + 1, i * 3.35, 0.5 if i % 2 else -0.5, 0.0)
    lines.append("TER\n")
    for i in range(8):
        serial = _add_res(lines, serial, "B", i + 1, i * 3.35 + 5.0, 9.0, 0.0)
    for i in range(8):
        serial = _add_res(lines, serial, "B", i + 21, i * 3.35 + 5.0, 9.0, 5.5)
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))


def _write_multi_patch_target(path: Path, n_res: int = 20) -> None:
    """Slightly curved 20-residue target chain A with two epitope patches."""
    lines: list[str] = []
    serial = 1
    for i in range(n_res):
        serial = _add_res(lines, serial, "A", i + 1, i * 3.35, 0.5 * math.sin(i * 0.35), 0.0)
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))


def _add_antibody_cdr_cases(specs: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]], out_root: Path) -> None:
    """CDR-H3-like loop against a short epitope patch. Tests loop-aware ADJ in a realistic antibody geometry."""
    input_dir = out_root / "fixed_target_inputs"
    tensor_dir = out_root / "tensors"
    input_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = input_dir / "cdr_framework.pdb"
    _write_cdr_framework(pdb_path)

    # Flat tensor order: A1-10 | B1-8 | CDR loop 12 res | B21-28
    # Indices:           0-9   | 10-17 | 18-29          | 30-37
    total_len = 38
    ss = torch.full((total_len,), SS_MASKED, dtype=torch.long)
    ss[0:10] = SS_STRAND      # target epitope (fixed)
    ss[10:18] = SS_STRAND     # VH anchor 1 (fixed)
    ss[18:30] = SS_LOOP       # CDR-H3 loop (generated)
    ss[30:38] = SS_STRAND     # VH anchor 2 (fixed)

    adj = _adj(total_len, ADJ_MASKED)
    _contact(adj, (2, 9), (19, 30))   # epitope core <-> CDR loop
    adj.fill_diagonal_(ADJ_NO)

    strict_adj = adj.clone()
    _no_contact(strict_adj, (2, 9), (11, 18))  # epitope vs anchor 1: no contact
    _no_contact(strict_adj, (2, 9), (31, 38))  # epitope vs anchor 2: no contact
    strict_adj.fill_diagonal_(ADJ_NO)

    ss_path = _save_tensor(tensor_dir / "antibody_cdr.ss.pt", ss)
    adj_path = _save_tensor(tensor_dir / "antibody_cdr.adj.pt", adj)
    strict_adj_path = _save_tensor(tensor_dir / "antibody_cdr.strict.adj.pt", strict_adj)

    target = {"ss_labels": ss.tolist(), "contact_blocks": [((2, 9), (19, 30))], "loop_span": [19, 30], "fixed_target": True}
    base_spec = {
        "input": str(pdb_path.resolve()),
        "contig": "A1-10,/0,B1-8,12-12,B21-28",
        "length": "38-38",
        "select_fixed_atoms": True,
        "ss_file": ss_path,
        "adj_file": adj_path,
    }
    _case(specs, manifest, "antibody_cdr_loop", base_spec, "antibody_cdr_loop", "CDR-H3-like generated loop against epitope patch with loop-ADJ", target)
    _case(specs, manifest, "antibody_cdr_loop_strict_no", {**base_spec, "adj_file": strict_adj_path}, "antibody_cdr_loop_strict_no", "CDR-H3 loop with ADJ_NO between epitope and framework anchors", target)


def _add_multi_patch_binder_case(specs: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]], out_root: Path) -> None:
    """Binder that must satisfy TWO separate contact patches on a fixed target. Tests multi-interface ADJ."""
    input_dir = out_root / "fixed_target_inputs"
    tensor_dir = out_root / "tensors"
    input_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = input_dir / "multi_patch_target.pdb"
    _write_multi_patch_target(pdb_path)

    # Flat order: A1-20 (target, fixed) | generated binder 28 res
    # Indices:    0-19                  | 20-47
    total_len = 48
    ss = torch.full((total_len,), SS_MASKED, dtype=torch.long)
    ss[0:20] = SS_STRAND    # target conditioned as extended/strand-like
    ss[22:28] = SS_STRAND   # binder strand segment 1 (contacts patch 1)
    ss[35:41] = SS_STRAND   # binder strand segment 2 (contacts patch 2)

    adj = _adj(total_len, ADJ_MASKED)
    _contact(adj, (2, 7), (23, 28))    # target patch 1 <-> binder seg 1
    _contact(adj, (14, 19), (36, 41))  # target patch 2 <-> binder seg 2
    adj.fill_diagonal_(ADJ_NO)

    ss_path = _save_tensor(tensor_dir / "multi_patch_binder.ss.pt", ss)
    adj_path = _save_tensor(tensor_dir / "multi_patch_binder.adj.pt", adj)
    target = {
        "ss_labels": ss.tolist(),
        "contact_blocks": [((2, 7), (23, 28)), ((14, 19), (36, 41))],
        "fixed_target": True,
        "description": "Two separate epitope patches on a 20-residue target",
    }
    _case(
        specs, manifest, "multi_patch_binder",
        {"input": str(pdb_path.resolve()), "contig": "A1-20,/0,28-28", "length": "48-48",
         "select_fixed_atoms": True, "ss_file": ss_path, "adj_file": adj_path},
        "multi_patch_binder",
        "Binder must satisfy two separate epitope contact patches (multi-interface ADJ)",
        target,
    )


def _add_multi_insert_case(specs: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]], out_root: Path) -> None:
    """Two simultaneous masked insertion windows in a single design spec."""
    tensor_dir = out_root / "tensors"
    length = 80
    ss = _segments(length, [
        (5, 20, SS_HELIX),
        (21, 28, SS_MASKED),   # insertion point 1
        (29, 45, SS_STRAND),
        (46, 53, SS_MASKED),   # insertion point 2
        (54, 72, SS_HELIX),
    ])
    adj = _adj(length, ADJ_MASKED)
    _contact(adj, (5, 20), (29, 45))   # helix 1 contacts strand
    _contact(adj, (29, 45), (54, 72))  # strand contacts helix 2
    adj.fill_diagonal_(ADJ_NO)
    ss_path = _save_tensor(tensor_dir / "multi_insert.ss.pt", ss)
    adj_path = _save_tensor(tensor_dir / "multi_insert.adj.pt", adj)
    target = {"ss_labels": ss.tolist(), "contact_blocks": [((5, 20), (29, 45)), ((29, 45), (54, 72))]}
    _case(
        specs, manifest, "multi_insert_masked",
        {"length": str(length), "ss_file": ss_path, "adj_file": adj_path, "sampled_insertion": "3-6"},
        "scaffold_multi_insert",
        "Two simultaneous masked insertion windows with sampled growth",
        target,
    )


def _add_large_topology_case(specs: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]], out_root: Path) -> None:
    """4-repeat Rossmann-like fold at 175 residues — tests whether SS+ADJ conditioning scales to larger designs."""
    tensor_dir = out_root / "tensors"
    length = 175
    ss = _segments(length, [
        (5, 14, SS_STRAND), (20, 41, SS_HELIX),
        (47, 56, SS_STRAND), (62, 83, SS_HELIX),
        (89, 98, SS_STRAND), (104, 125, SS_HELIX),
        (131, 140, SS_STRAND), (146, 167, SS_HELIX),
    ])
    contacts = [
        # parallel beta sheet
        ((5, 14), (47, 56)), ((47, 56), (89, 98)), ((89, 98), (131, 140)),
        # helix packing
        ((20, 41), (62, 83)), ((62, 83), (104, 125)), ((104, 125), (146, 167)),
    ]
    adj = _adj(length, ADJ_NO)
    for a, b in contacts:
        _contact(adj, a, b)
    ss_path = _save_tensor(tensor_dir / "large_rossmann.ss.pt", ss)
    adj_path = _save_tensor(tensor_dir / "large_rossmann.adj.pt", adj)
    target = {"ss_labels": ss.tolist(), "contact_blocks": contacts}
    _case(
        specs, manifest, "large_rossmann",
        {"length": str(length), "ss_file": ss_path, "adj_file": adj_path},
        "large_full_ss_adj",
        "Full SS+ADJ on a 175-residue 4-repeat Rossmann fold — scale test",
        target,
    )


def _add_conflict_cases(specs: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]], out_root: Path) -> None:
    """SS file says helix, ss_ratio says max beta — tests which conditioning channel dominates."""
    tensor_dir = out_root / "tensors"
    length = 100
    ss = _segments(length, [(5, 27, SS_HELIX), (38, 60, SS_HELIX), (71, 93, SS_HELIX)])
    ss_path = str((tensor_dir / "helix_bundle.ss.pt").resolve())  # reuse existing helix_bundle tensor
    adj_masked = _adj(length, ADJ_MASKED)
    adj_masked_path = _save_tensor(tensor_dir / "helix_bundle_conflict_adj_masked.pt", adj_masked)
    target = {"ss_labels": ss.tolist(), "contact_blocks": [], "ss_ratio": 1.0}
    _case(
        specs, manifest, "conflict_helix_ss_beta_ratio",
        {"length": str(length), "ss_file": ss_path, "adj_file": str(adj_masked_path), "ss_ratio": 1.0},
        "conflict_ss_ratio",
        "Helix SS file vs ss_ratio=1.0 (all beta): which channel dominates?",
        target,
    )
    _case(
        specs, manifest, "conflict_helix_ss_zero_ratio",
        {"length": str(length), "ss_file": ss_path, "adj_file": str(adj_masked_path), "ss_ratio": 0.0},
        "conflict_ss_ratio",
        "Helix SS file vs ss_ratio=0.0 (no beta): consistent or conflicting?",
        {**target, "ss_ratio": 0.0},
    )


def _add_fixed_target_loop_binder_cases(specs: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]], out_root: Path) -> None:
    input_dir = out_root / "fixed_target_inputs"
    tensor_dir = out_root / "tensors"
    input_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = input_dir / "fixed_target_framework_loop_gap.pdb"
    _write_loop_target_framework(pdb_path)

    total_len = 42
    ss = torch.full((total_len,), SS_MASKED, dtype=torch.long)
    ss[0:16] = SS_STRAND
    ss[16:24] = SS_STRAND
    ss[24:34] = SS_LOOP
    ss[34:42] = SS_STRAND
    adj = _adj(total_len, ADJ_MASKED)
    _contact(adj, (4, 13), (25, 34))
    adj.fill_diagonal_(ADJ_NO)

    strict_adj = adj.clone()
    _no_contact(strict_adj, (4, 13), (17, 24))
    _no_contact(strict_adj, (4, 13), (35, 42))
    strict_adj.fill_diagonal_(ADJ_NO)

    ss_path = _save_tensor(tensor_dir / "loop_binder.ss.pt", ss)
    adj_path = _save_tensor(tensor_dir / "loop_binder.adj.pt", adj)
    strict_adj_path = _save_tensor(tensor_dir / "loop_binder.strict.adj.pt", strict_adj)
    target = {"ss_labels": ss.tolist(), "contact_blocks": [((4, 13), (25, 34))], "loop_span": [25, 34], "fixed_target": True}
    base_spec = {
        "input": str(pdb_path.resolve()),
        "contig": "A1-16,/0,B1-8,10-10,B19-26",
        "length": "42-42",
        "select_fixed_atoms": True,
        "ss_file": ss_path,
        "adj_file": adj_path,
    }
    _case(specs, manifest, "binder_loop_adj", base_spec, "fixed_target_loop_binder", "Fixed target/framework with generated loop ADJ-conditioned to target", target)
    _case(specs, manifest, "binder_loop_adj_strict_no", {**base_spec, "adj_file": strict_adj_path}, "fixed_target_loop_binder_strict_no", "Loop binder with explicit ADJ_NO between target and fixed framework anchors", target)


def _write_helix_target(path: Path, n_res: int = 16) -> None:
    """Ideal alpha-helical target chain A."""
    lines: list[str] = []
    serial = 1
    for i in range(n_res):
        theta = math.radians(i * 100)
        x = 2.3 * math.cos(theta)
        y = 2.3 * math.sin(theta)
        z = i * 1.5
        serial = _add_res(lines, serial, "A", i + 1, x, y, z)
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))


def _write_helix_loop_framework(path: Path) -> None:
    """Helical target chain A (16 res) + two framework anchors chain B with CDR-like gap."""
    lines: list[str] = []
    serial = 1
    for i in range(16):
        theta = math.radians(i * 100)
        x = 2.3 * math.cos(theta)
        y = 2.3 * math.sin(theta)
        z = i * 1.5
        serial = _add_res(lines, serial, "A", i + 1, x, y, z)
    lines.append("TER\n")
    for i in range(6):
        serial = _add_res(lines, serial, "B", i + 1, i * 3.35 + 4.0, 12.0, 3.0)
    for i in range(6):
        serial = _add_res(lines, serial, "B", i + 21, i * 3.35 + 4.0, 12.0, 8.5)
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))


def _write_mixed_surface_target(path: Path) -> None:
    """Small beta strand (8 res) + helix (10 res) surface as target chain A (18 res)."""
    lines: list[str] = []
    serial = 1
    for i in range(8):
        serial = _add_res(lines, serial, "A", i + 1, i * 3.35, 0.4 if i % 2 else -0.4, 0.0)
    for i in range(10):
        theta = math.radians(i * 100)
        x = 8 * 3.35 + 5.0 + 2.3 * math.cos(theta)
        y = 2.3 * math.sin(theta)
        z = i * 1.5
        serial = _add_res(lines, serial, "A", i + 9, x, y, z)
    lines.append("TER\nEND\n")
    path.write_text("".join(lines))


def _add_helix_target_binder_cases(specs: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]], out_root: Path) -> None:
    """Helix target bound by helix, beta, and loop binders — tests cross-topology interface conditioning."""
    input_dir = out_root / "fixed_target_inputs"
    tensor_dir = out_root / "tensors"
    input_dir.mkdir(parents=True, exist_ok=True)

    target_len = 16
    binder_len = 20
    total_len = target_len + binder_len  # 36

    helix_pdb = input_dir / "fixed_helix_target_A.pdb"
    _write_helix_target(helix_pdb)

    def _base_spec(ss_path: str, adj_path: str) -> dict[str, Any]:
        return {
            "input": str(helix_pdb.resolve()),
            "contig": f"A1-{target_len},/0,{binder_len}-{binder_len}",
            "length": f"{total_len}-{total_len}",
            "select_fixed_atoms": True,
            "ss_file": ss_path,
            "adj_file": adj_path,
        }

    # ── Helix binder ─────────────────────────────────────────────────────────
    ss = torch.full((total_len,), SS_MASKED, dtype=torch.long)
    ss[0:target_len] = SS_HELIX
    ss[20:30] = SS_HELIX           # binder core: flat 20-29 (positions 5-14 of 20-res binder)
    adj = _adj(total_len, ADJ_MASKED)
    _contact(adj, (3, 14), (21, 30))   # target face <-> binder helix core
    adj.fill_diagonal_(ADJ_NO)
    strict_adj = adj.clone()
    _no_contact(strict_adj, (3, 14), (17, 20))   # no contact vs binder N-flank
    _no_contact(strict_adj, (3, 14), (31, 37))   # no contact vs binder C-flank
    strict_adj.fill_diagonal_(ADJ_NO)
    ss_path = _save_tensor(tensor_dir / "helix_target_helix_binder.ss.pt", ss)
    adj_path = _save_tensor(tensor_dir / "helix_target_helix_binder.adj.pt", adj)
    strict_path = _save_tensor(tensor_dir / "helix_target_helix_binder.strict.adj.pt", strict_adj)
    tgt = {"ss_labels": ss.tolist(), "contact_blocks": [((3, 14), (21, 30))], "fixed_target": True}
    _case(specs, manifest, "helix_target_helix_binder", _base_spec(ss_path, adj_path), "helix_target_helix_binder", "Helix binder docking onto helical target", tgt)
    _case(specs, manifest, "helix_target_helix_binder_strict_no", _base_spec(ss_path, strict_path), "helix_target_helix_binder_strict_no", "Helix-helix interface with explicit ADJ_NO at binder flanks", tgt)

    # ── Beta binder vs helix target ───────────────────────────────────────────
    ss_b = torch.full((total_len,), SS_MASKED, dtype=torch.long)
    ss_b[0:target_len] = SS_HELIX
    ss_b[20:28] = SS_STRAND        # binder central 8 residues as strand
    adj_b = _adj(total_len, ADJ_MASKED)
    _contact(adj_b, (3, 14), (21, 28))
    adj_b.fill_diagonal_(ADJ_NO)
    ss_b_path = _save_tensor(tensor_dir / "helix_target_beta_binder.ss.pt", ss_b)
    adj_b_path = _save_tensor(tensor_dir / "helix_target_beta_binder.adj.pt", adj_b)
    tgt_b = {"ss_labels": ss_b.tolist(), "contact_blocks": [((3, 14), (21, 28))], "fixed_target": True}
    _case(specs, manifest, "helix_target_beta_binder", _base_spec(ss_b_path, adj_b_path), "helix_target_beta_binder", "Beta strand binder docking onto helical target", tgt_b)

    # ── Loop binder vs helix target ───────────────────────────────────────────
    loop_pdb = input_dir / "fixed_helix_target_loop_framework.pdb"
    _write_helix_loop_framework(loop_pdb)
    # Flat: A1-16 | B1-6 | loop 10 | B21-26  = 16+6+10+6 = 38
    loop_total = 38
    ss_l = torch.full((loop_total,), SS_MASKED, dtype=torch.long)
    ss_l[0:16] = SS_HELIX          # target
    ss_l[16:22] = SS_STRAND        # framework anchor 1
    ss_l[22:32] = SS_LOOP          # generated CDR-like loop
    ss_l[32:38] = SS_STRAND        # framework anchor 2
    adj_l = _adj(loop_total, ADJ_MASKED)
    _contact(adj_l, (4, 13), (23, 32))    # target face <-> loop
    adj_l.fill_diagonal_(ADJ_NO)
    ss_l_path = _save_tensor(tensor_dir / "helix_target_loop_binder.ss.pt", ss_l)
    adj_l_path = _save_tensor(tensor_dir / "helix_target_loop_binder.adj.pt", adj_l)
    tgt_l = {"ss_labels": ss_l.tolist(), "contact_blocks": [((4, 13), (23, 32))], "loop_span": [23, 32], "fixed_target": True}
    _case(
        specs, manifest, "helix_target_loop_binder",
        {"input": str(loop_pdb.resolve()), "contig": "A1-16,/0,B1-6,10-10,B21-26",
         "length": "38-38", "select_fixed_atoms": True, "ss_file": ss_l_path, "adj_file": adj_l_path},
        "helix_target_loop_binder",
        "CDR-like loop binder docking onto helical target",
        tgt_l,
    )


def _add_helix_binder_cases(specs: dict[str, dict[str, Any]], manifest: dict[str, dict[str, Any]], out_root: Path) -> None:
    """Helical binder targeting beta-strand and mixed-topology surfaces."""
    input_dir = out_root / "fixed_target_inputs"
    tensor_dir = out_root / "tensors"
    input_dir.mkdir(parents=True, exist_ok=True)

    binder_len = 20

    # ── Helix binder vs beta target (reuse existing beta target PDB) ──────────
    beta_pdb = input_dir / "fixed_beta_target_A.pdb"
    if not beta_pdb.exists():
        _write_beta_target(beta_pdb)
    target_len_b = 16
    total_b = target_len_b + binder_len   # 36
    ss_hb = torch.full((total_b,), SS_MASKED, dtype=torch.long)
    ss_hb[0:target_len_b] = SS_STRAND    # beta target
    ss_hb[20:30] = SS_HELIX              # helix binder core
    adj_hb = _adj(total_b, ADJ_MASKED)
    _contact(adj_hb, (3, 14), (21, 30))
    adj_hb.fill_diagonal_(ADJ_NO)
    strict_hb = adj_hb.clone()
    _no_contact(strict_hb, (3, 14), (17, 20))
    _no_contact(strict_hb, (3, 14), (31, 37))
    strict_hb.fill_diagonal_(ADJ_NO)
    ss_hb_path = _save_tensor(tensor_dir / "helix_binder_beta_target.ss.pt", ss_hb)
    adj_hb_path = _save_tensor(tensor_dir / "helix_binder_beta_target.adj.pt", adj_hb)
    strict_hb_path = _save_tensor(tensor_dir / "helix_binder_beta_target.strict.adj.pt", strict_hb)
    tgt_hb = {"ss_labels": ss_hb.tolist(), "contact_blocks": [((3, 14), (21, 30))], "fixed_target": True}
    base_hb = {"input": str(beta_pdb.resolve()), "contig": f"A1-{target_len_b},/0,{binder_len}-{binder_len}",
               "length": f"{total_b}-{total_b}", "select_fixed_atoms": True,
               "ss_file": ss_hb_path, "adj_file": adj_hb_path}
    _case(specs, manifest, "helix_binder_beta_target", base_hb, "helix_binder_beta_target", "Helical binder docking onto beta-strand target", tgt_hb)
    _case(specs, manifest, "helix_binder_beta_target_strict_no", {**base_hb, "adj_file": strict_hb_path}, "helix_binder_beta_target_strict_no", "Helix-on-beta binder with explicit ADJ_NO at binder flanks", tgt_hb)

    # ── Helix binder vs mixed (strand + helix) surface ────────────────────────
    mixed_pdb = input_dir / "fixed_mixed_surface_target.pdb"
    _write_mixed_surface_target(mixed_pdb)
    target_len_m = 18   # 8 strand + 10 helix
    total_m = target_len_m + binder_len   # 38
    ss_hm = torch.full((total_m,), SS_MASKED, dtype=torch.long)
    ss_hm[0:8] = SS_STRAND               # target strand face
    ss_hm[8:18] = SS_HELIX               # target helix face
    ss_hm[22:32] = SS_HELIX              # helix binder core
    adj_hm = _adj(total_m, ADJ_MASKED)
    # Binder helix contacts the target helix face (helix-helix interface)
    _contact(adj_hm, (9, 18), (23, 32))
    adj_hm.fill_diagonal_(ADJ_NO)
    # Also condition that binder does NOT contact the target strand face
    _no_contact(adj_hm, (1, 8), (19, 39))
    adj_hm.fill_diagonal_(ADJ_NO)
    ss_hm_path = _save_tensor(tensor_dir / "helix_binder_mixed_target.ss.pt", ss_hm)
    adj_hm_path = _save_tensor(tensor_dir / "helix_binder_mixed_target.adj.pt", adj_hm)
    tgt_hm = {"ss_labels": ss_hm.tolist(), "contact_blocks": [((9, 18), (23, 32))], "fixed_target": True}
    _case(
        specs, manifest, "helix_binder_mixed_target",
        {"input": str(mixed_pdb.resolve()), "contig": f"A1-{target_len_m},/0,{binder_len}-{binder_len}",
         "length": f"{total_m}-{total_m}", "select_fixed_atoms": True,
         "ss_file": ss_hm_path, "adj_file": adj_hm_path},
        "helix_binder_mixed_target",
        "Helical binder targeting the helix face of a mixed strand+helix surface",
        tgt_hm,
    )


def build_benchmark(out_root: Path) -> tuple[Path, Path]:
    tensor_dir = out_root / "tensors"
    specs: dict[str, dict[str, Any]] = {}
    manifest: dict[str, dict[str, Any]] = {}

    for length in (100, 150):
        _case(specs, manifest, f"baseline_L{length}", {"length": str(length)}, "unconditional", f"Unconditioned de novo {length}mer")

    for target in (0.0, 0.25, 0.5, 0.75, 1.0):
        _case(specs, manifest, f"ssratio_{int(target * 100):03d}", {"length": "120", "ss_ratio": target}, "ss_ratio", f"Global ss_ratio target {target:.2f}", {"ss_ratio": target})

    for key, value, ss_ratio in (("nonloopy_true", True, None), ("nonloopy_false", False, None), ("nonloopy_beta", True, 0.75)):
        spec: dict[str, Any] = {"length": "120", "is_non_loopy": value}
        if ss_ratio is not None:
            spec["ss_ratio"] = ss_ratio
        _case(specs, manifest, key, spec, "is_non_loopy", f"is_non_loopy={value}" + (f", ss_ratio={ss_ratio}" if ss_ratio is not None else ""), {"is_non_loopy": value, "ss_ratio": ss_ratio})

    inline_cases = {
        "select_3helix": ({"A5-28": "H", "A40-63": "H", "A75-98": "H"}, [(5, 28, SS_HELIX), (40, 63, SS_HELIX), (75, 98, SS_HELIX)], "Three separated helices via select_ss"),
        "select_betasheet": ({"A5-14": "E", "A24-33": "E", "A43-52": "E", "A62-71": "E", "A81-90": "E"}, [(5, 14, SS_STRAND), (24, 33, SS_STRAND), (43, 52, SS_STRAND), (62, 71, SS_STRAND), (81, 90, SS_STRAND)], "Five strand segments via select_ss"),
        "select_alpha_beta": ({"A5-16": "E", "A25-45": "H", "A55-66": "E", "A74-94": "H", "A102-113": "E"}, [(5, 16, SS_STRAND), (25, 45, SS_HELIX), (55, 66, SS_STRAND), (74, 94, SS_HELIX), (102, 113, SS_STRAND)], "Alternating alpha/beta via select_ss"),
        "select_loop_linker": ({"A5-30": "H", "A45-75": "L", "A90-115": "H"}, [(5, 30, SS_HELIX), (45, 75, SS_LOOP), (90, 115, SS_HELIX)], "Explicit long loop separating two helices"),
    }
    for key, (select_ss, target_ss, description) in inline_cases.items():
        _case(specs, manifest, key, {"length": "120", "select_ss": select_ss}, "select_ss", description, {"ss_segments": target_ss})

    file_cases: list[tuple[str, int, torch.Tensor, list[tuple[tuple[int, int], tuple[int, int]]], str]] = [
        ("helix_bundle", 100, _segments(100, [(5, 27, SS_HELIX), (38, 60, SS_HELIX), (71, 93, SS_HELIX)]), [((5, 27), (38, 60)), ((38, 60), (71, 93)), ((5, 27), (71, 93))], "Three-helix bundle contact graph"),
        ("beta_sandwich", 110, _segments(110, [(5, 14, SS_STRAND), (24, 33, SS_STRAND), (43, 52, SS_STRAND), (62, 71, SS_STRAND), (81, 90, SS_STRAND), (98, 107, SS_STRAND)]), [((5, 14), (24, 33)), ((24, 33), (43, 52)), ((62, 71), (81, 90)), ((81, 90), (98, 107)), ((5, 14), (98, 107))], "Two beta-sheet faces with a sandwich contact"),
        ("rossmann_like", 120, _segments(120, [(5, 13, SS_STRAND), (20, 38, SS_HELIX), (45, 53, SS_STRAND), (60, 78, SS_HELIX), (85, 93, SS_STRAND), (100, 118, SS_HELIX)]), [((5, 13), (45, 53)), ((45, 53), (85, 93)), ((20, 38), (60, 78)), ((60, 78), (100, 118))], "Rossmann-like alternating beta-alpha topology"),
    ]

    for base, length, ss, contacts, description in file_cases:
        full_adj = _adj(length, ADJ_NO)
        for a, b in contacts:
            _contact(full_adj, a, b)
        target = {"ss_labels": ss.tolist(), "contact_blocks": contacts}
        ss_path = _save_tensor(tensor_dir / f"{base}.ss.pt", ss)
        adj_path = _save_tensor(tensor_dir / f"{base}.adj.pt", full_adj)

        _case(specs, manifest, f"full_{base}", {"length": str(length), "ss_file": ss_path, "adj_file": adj_path}, "full_ss_adj", f"Full SS+ADJ: {description}", target)
        if base in {"beta_sandwich", "rossmann_like"}:
            _case(specs, manifest, f"full_{base}_ssratio080", {"length": str(length), "ss_file": ss_path, "adj_file": adj_path, "ss_ratio": 0.80}, "full_ss_adj_ssratio", f"Full SS+ADJ plus global beta-rich ss_ratio=0.80: {description}", {**target, "ss_ratio": 0.80})

        ss_only_adj = _adj(length, ADJ_MASKED)
        ss_only_adj_path = _save_tensor(tensor_dir / f"{base}.ss_only_adj_masked.pt", ss_only_adj)
        _case(specs, manifest, f"ssonly_{base}", {"length": str(length), "ss_file": ss_path, "adj_file": ss_only_adj_path}, "ss_only", f"SS only, ADJ fully masked: {description}", {"ss_labels": ss.tolist(), "contact_blocks": []})

        masked_ss = torch.full((length,), SS_MASKED, dtype=torch.long)
        masked_ss_path = _save_tensor(tensor_dir / f"{base}.adj_only_ss_masked.pt", masked_ss)
        _case(specs, manifest, f"adjonly_{base}", {"length": str(length), "ss_file": masked_ss_path, "adj_file": adj_path}, "adj_only", f"ADJ only, SS fully masked: {description}", {"ss_labels": masked_ss.tolist(), "contact_blocks": contacts})

        sparse_adj = _adj(length, ADJ_MASKED)
        for a, b in contacts[:1]:
            _contact(sparse_adj, a, b)
        sparse_adj_path = _save_tensor(tensor_dir / f"{base}.sparse_adj_only.pt", sparse_adj)
        _case(specs, manifest, f"sparse_adjonly_{base}", {"length": str(length), "ss_file": masked_ss_path, "adj_file": sparse_adj_path}, "sparse_adj_only", f"Sparse ADJ-only, only one contact block exposed: {description}", {"ss_labels": masked_ss.tolist(), "contact_blocks": contacts[:1]})

        partial_ss = torch.full((length,), SS_MASKED, dtype=torch.long)
        partial_adj = _adj(length, ADJ_MASKED)
        for a, b in contacts[:2]:
            partial_ss[a[0] - 1 : a[1]] = ss[a[0] - 1 : a[1]]
            partial_ss[b[0] - 1 : b[1]] = ss[b[0] - 1 : b[1]]
            _contact(partial_adj, a, b)
        partial_adj.fill_diagonal_(ADJ_NO)
        partial_ss_path = _save_tensor(tensor_dir / f"{base}.interaction_partial.ss.pt", partial_ss)
        partial_adj_path = _save_tensor(tensor_dir / f"{base}.interaction_partial.adj.pt", partial_adj)
        _case(specs, manifest, f"partial_site_{base}", {"length": str(length), "ss_file": partial_ss_path, "adj_file": partial_adj_path}, "interaction_site_partial", f"Only selected interacting SS blocks exposed: {description}", {"ss_labels": partial_ss.tolist(), "contact_blocks": contacts[:2]})
        if base == "beta_sandwich":
            _case(specs, manifest, f"partial_site_{base}_ssratio080", {"length": str(length), "ss_file": partial_ss_path, "adj_file": partial_adj_path, "ss_ratio": 0.80}, "interaction_site_partial_ssratio", "Partial beta interface SS+ADJ plus global ss_ratio=0.80", {"ss_labels": partial_ss.tolist(), "contact_blocks": contacts[:2], "ss_ratio": 0.80})
            strict_partial = partial_adj.clone()
            for a, b in contacts[2:]:
                _no_contact(strict_partial, a, b)
            strict_path = _save_tensor(tensor_dir / f"{base}.interaction_partial_strict_no.adj.pt", strict_partial)
            _case(specs, manifest, f"partial_site_{base}_strict_no", {"length": str(length), "ss_file": partial_ss_path, "adj_file": strict_path}, "interaction_site_partial_strict_no", "Partial beta interface with selected explicit non-contact negatives", {"ss_labels": partial_ss.tolist(), "contact_blocks": contacts[:2]})

        remodel_ss = ss.clone()
        remodel_adj = full_adj.clone()
        windows = [(max(1, length // 3 - 4), min(length, length // 3 + 5)), (max(1, 2 * length // 3 - 4), min(length, 2 * length // 3 + 5))]
        for start, end in windows:
            remodel_ss[start - 1 : end] = SS_MASKED
            _mask_rows_cols(remodel_adj, start, end)
        remodel_ss_path = _save_tensor(tensor_dir / f"{base}.remodel.ss.pt", remodel_ss)
        remodel_adj_path = _save_tensor(tensor_dir / f"{base}.remodel.adj.pt", remodel_adj)
        _case(specs, manifest, f"remodel_{base}", {"length": str(length), "ss_file": remodel_ss_path, "adj_file": remodel_adj_path}, "remodeling_partial", f"Full fold with two masked remodeling windows: {description}", {"ss_labels": remodel_ss.tolist(), "contact_blocks": contacts, "masked_windows": windows})

    # Expansion/insertion modes: exercise sampled_N/C, sampled_insertion, and mask_loops.
    exp_len = 64
    exp_ss = _segments(exp_len, [(5, 18, SS_STRAND), (19, 24, SS_MASKED), (25, 38, SS_STRAND), (45, 58, SS_HELIX)])
    exp_adj = _adj(exp_len, ADJ_MASKED)
    _contact(exp_adj, (5, 18), (25, 38))
    exp_ss_path = _save_tensor(tensor_dir / "expansion_masked.ss.pt", exp_ss)
    exp_adj_path = _save_tensor(tensor_dir / "expansion_masked.adj.pt", exp_adj)
    _case(specs, manifest, "expansion_masked_insert", {"length": str(exp_len), "ss_file": exp_ss_path, "adj_file": exp_adj_path, "sampled_insertion": "4-8"}, "scaffold_expansion_masked", "Expand an explicit SS=3 masked insertion segment between beta strands", {"ss_labels": exp_ss.tolist(), "contact_blocks": [((5, 18), (25, 38))]})
    _case(specs, manifest, "expansion_terminal_nc", {"length": str(exp_len), "ss_file": exp_ss_path, "adj_file": exp_adj_path, "sampled_N": "3-6", "sampled_C": "3-6"}, "scaffold_expansion_terminal", "Sample masked N/C terminal additions around a conditioned fold", {"ss_labels": exp_ss.tolist(), "contact_blocks": [((5, 18), (25, 38))]})
    loop_exp_ss = _segments(72, [(5, 18, SS_STRAND), (19, 28, SS_LOOP), (29, 42, SS_STRAND), (49, 66, SS_HELIX)])
    loop_exp_adj = _adj(72, ADJ_MASKED)
    _contact(loop_exp_adj, (5, 18), (29, 42))
    loop_exp_ss_path = _save_tensor(tensor_dir / "expansion_loop.ss.pt", loop_exp_ss)
    loop_exp_adj_path = _save_tensor(tensor_dir / "expansion_loop.adj.pt", loop_exp_adj)
    _case(specs, manifest, "expansion_mask_loops", {"length": "72", "ss_file": loop_exp_ss_path, "adj_file": loop_exp_adj_path, "mask_loops": True, "sampled_insertion": "3-6"}, "scaffold_expansion_loop", "Expand existing loop segments with mask_loops=True", {"ss_labels": loop_exp_ss.tolist(), "contact_blocks": [((5, 18), (29, 42))]})

    _add_fixed_target_beta_binder_cases(specs, manifest, out_root)
    _add_fixed_target_loop_binder_cases(specs, manifest, out_root)
    _add_antibody_cdr_cases(specs, manifest, out_root)
    _add_multi_patch_binder_case(specs, manifest, out_root)
    _add_multi_insert_case(specs, manifest, out_root)
    _add_large_topology_case(specs, manifest, out_root)
    _add_conflict_cases(specs, manifest, out_root)
    _add_helix_target_binder_cases(specs, manifest, out_root)
    _add_helix_binder_cases(specs, manifest, out_root)

    input_json = out_root / "ss_adj_conditioning_inputs.json"
    manifest_json = out_root / "ss_adj_conditioning_manifest.json"
    out_root.mkdir(parents=True, exist_ok=True)
    input_json.write_text(json.dumps(specs, indent=2) + "\n")
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
    return input_json, manifest_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True, type=Path)
    args = parser.parse_args()
    input_json, manifest_json = build_benchmark(args.out_root)
    n_cases = len(json.loads(input_json.read_text()))
    print(f"input_json={input_json}")
    print(f"manifest_json={manifest_json}")
    print(f"n_cases={n_cases}")


if __name__ == "__main__":
    main()
