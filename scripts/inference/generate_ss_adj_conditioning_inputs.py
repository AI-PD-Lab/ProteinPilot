#!/usr/bin/env python3
"""Generate RFdiffusion3 SS/ADJ conditioning benchmark inputs."""

from __future__ import annotations

import argparse
import json
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


def _mask_rows_cols(adj: torch.Tensor, start: int, end: int) -> None:
    sl = slice(start - 1, end)
    adj[sl, :] = ADJ_MASKED
    adj[:, sl] = ADJ_MASKED
    adj.fill_diagonal_(ADJ_NO)


def _save_tensor(path: Path, tensor: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)
    return str(path)


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
        "length": int(spec.get("length", 0)),
        "target": target or {},
    }


def build_benchmark(out_root: Path) -> tuple[Path, Path]:
    tensor_dir = out_root / "tensors"
    specs: dict[str, dict[str, Any]] = {}
    manifest: dict[str, dict[str, Any]] = {}

    # 1. Unconditional baseline.
    for length in (100, 150):
        _case(
            specs,
            manifest,
            f"baseline_L{length}",
            {"length": str(length)},
            "unconditional",
            f"Unconditioned de novo {length}mer",
        )

    # 2. Global ss_ratio sweep.
    for target in (0.0, 0.25, 0.5, 0.75, 1.0):
        _case(
            specs,
            manifest,
            f"ssratio_{int(target * 100):03d}",
            {"length": "120", "ss_ratio": target},
            "ss_ratio",
            f"Global ss_ratio target {target:.2f}",
            {"ss_ratio": target},
        )

    # 3. Global topology flag.
    for key, value, ss_ratio in (
        ("nonloopy_true", True, None),
        ("nonloopy_false", False, None),
        ("nonloopy_beta", True, 0.75),
    ):
        spec: dict[str, Any] = {"length": "120", "is_non_loopy": value}
        if ss_ratio is not None:
            spec["ss_ratio"] = ss_ratio
        _case(
            specs,
            manifest,
            key,
            spec,
            "is_non_loopy",
            f"is_non_loopy={value}" + (f", ss_ratio={ss_ratio}" if ss_ratio is not None else ""),
            {"is_non_loopy": value, "ss_ratio": ss_ratio},
        )

    # 4. Inline SS selection via select_ss.
    inline_cases = {
        "select_3helix": {
            "select_ss": {"A5-28": "H", "A40-63": "H", "A75-98": "H"},
            "target_ss": [(5, 28, SS_HELIX), (40, 63, SS_HELIX), (75, 98, SS_HELIX)],
            "description": "Three separated helices via select_ss",
        },
        "select_betasheet": {
            "select_ss": {"A5-14": "E", "A24-33": "E", "A43-52": "E", "A62-71": "E", "A81-90": "E"},
            "target_ss": [(5, 14, SS_STRAND), (24, 33, SS_STRAND), (43, 52, SS_STRAND), (62, 71, SS_STRAND), (81, 90, SS_STRAND)],
            "description": "Five strand segments via select_ss",
        },
        "select_alpha_beta": {
            "select_ss": {"A5-16": "E", "A25-45": "H", "A55-66": "E", "A74-94": "H", "A102-113": "E"},
            "target_ss": [(5, 16, SS_STRAND), (25, 45, SS_HELIX), (55, 66, SS_STRAND), (74, 94, SS_HELIX), (102, 113, SS_STRAND)],
            "description": "Alternating alpha/beta via select_ss",
        },
        "select_loop_linker": {
            "select_ss": {"A5-30": "H", "A45-75": "L", "A90-115": "H"},
            "target_ss": [(5, 30, SS_HELIX), (45, 75, SS_LOOP), (90, 115, SS_HELIX)],
            "description": "Explicit long loop separating two helices",
        },
    }
    for key, data in inline_cases.items():
        _case(
            specs,
            manifest,
            key,
            {"length": "120", "select_ss": data["select_ss"]},
            "select_ss",
            data["description"],
            {"ss_segments": data["target_ss"]},
        )

    # 5-8. File-based SS/ADJ cases.
    file_cases: list[tuple[str, int, torch.Tensor, list[tuple[tuple[int, int], tuple[int, int]]], str]] = []
    file_cases.append(
        (
            "helix_bundle",
            100,
            _segments(100, [(5, 27, SS_HELIX), (38, 60, SS_HELIX), (71, 93, SS_HELIX)]),
            [((5, 27), (38, 60)), ((38, 60), (71, 93)), ((5, 27), (71, 93))],
            "Three-helix bundle contact graph",
        )
    )
    file_cases.append(
        (
            "beta_sandwich",
            110,
            _segments(110, [(5, 14, SS_STRAND), (24, 33, SS_STRAND), (43, 52, SS_STRAND), (62, 71, SS_STRAND), (81, 90, SS_STRAND), (98, 107, SS_STRAND)]),
            [((5, 14), (24, 33)), ((24, 33), (43, 52)), ((62, 71), (81, 90)), ((81, 90), (98, 107)), ((5, 14), (98, 107))],
            "Two beta-sheet faces with a sandwich contact",
        )
    )
    file_cases.append(
        (
            "rossmann_like",
            120,
            _segments(120, [(5, 13, SS_STRAND), (20, 38, SS_HELIX), (45, 53, SS_STRAND), (60, 78, SS_HELIX), (85, 93, SS_STRAND), (100, 118, SS_HELIX)]),
            [((5, 13), (45, 53)), ((45, 53), (85, 93)), ((20, 38), (60, 78)), ((60, 78), (100, 118))],
            "Rossmann-like alternating beta-alpha topology",
        )
    )

    for base, length, ss, contacts, description in file_cases:
        full_adj = _adj(length, ADJ_NO)
        for a, b in contacts:
            _contact(full_adj, a, b)
        target = {"ss_labels": ss.tolist(), "contact_blocks": contacts}

        ss_path = _save_tensor(tensor_dir / f"{base}.ss.pt", ss)
        adj_path = _save_tensor(tensor_dir / f"{base}.adj.pt", full_adj)
        _case(
            specs,
            manifest,
            f"full_{base}",
            {"length": str(length), "ss_file": ss_path, "adj_file": adj_path},
            "full_ss_adj",
            f"Full SS+ADJ: {description}",
            target,
        )

        ss_only_adj = _adj(length, ADJ_MASKED)
        ss_only_adj.fill_diagonal_(ADJ_NO)
        ss_only_adj_path = _save_tensor(tensor_dir / f"{base}.ss_only_adj_masked.pt", ss_only_adj)
        _case(
            specs,
            manifest,
            f"ssonly_{base}",
            {"length": str(length), "ss_file": ss_path, "adj_file": ss_only_adj_path},
            "ss_only",
            f"SS only, ADJ fully masked: {description}",
            {"ss_labels": ss.tolist(), "contact_blocks": []},
        )

        masked_ss = torch.full((length,), SS_MASKED, dtype=torch.long)
        masked_ss_path = _save_tensor(tensor_dir / f"{base}.adj_only_ss_masked.pt", masked_ss)
        _case(
            specs,
            manifest,
            f"adjonly_{base}",
            {"length": str(length), "ss_file": masked_ss_path, "adj_file": adj_path},
            "adj_only",
            f"ADJ only, SS fully masked: {description}",
            {"ss_labels": masked_ss.tolist(), "contact_blocks": contacts},
        )

        partial_ss = torch.full((length,), SS_MASKED, dtype=torch.long)
        partial_adj = _adj(length, ADJ_MASKED)
        for a, b in contacts[:2]:
            partial_ss[a[0] - 1 : a[1]] = ss[a[0] - 1 : a[1]]
            partial_ss[b[0] - 1 : b[1]] = ss[b[0] - 1 : b[1]]
            _contact(partial_adj, a, b)
        partial_adj.fill_diagonal_(ADJ_NO)
        partial_ss_path = _save_tensor(tensor_dir / f"{base}.interaction_partial.ss.pt", partial_ss)
        partial_adj_path = _save_tensor(tensor_dir / f"{base}.interaction_partial.adj.pt", partial_adj)
        _case(
            specs,
            manifest,
            f"partial_site_{base}",
            {"length": str(length), "ss_file": partial_ss_path, "adj_file": partial_adj_path},
            "interaction_site_partial",
            f"Only selected interacting SS blocks exposed: {description}",
            {"ss_labels": partial_ss.tolist(), "contact_blocks": contacts[:2]},
        )

        remodel_ss = ss.clone()
        remodel_adj = full_adj.clone()
        # Mask boundary/remodeling windows while leaving the surrounding fold specified.
        windows = [(max(1, length // 3 - 4), min(length, length // 3 + 5)), (max(1, 2 * length // 3 - 4), min(length, 2 * length // 3 + 5))]
        for start, end in windows:
            remodel_ss[start - 1 : end] = SS_MASKED
            _mask_rows_cols(remodel_adj, start, end)
        remodel_ss_path = _save_tensor(tensor_dir / f"{base}.remodel.ss.pt", remodel_ss)
        remodel_adj_path = _save_tensor(tensor_dir / f"{base}.remodel.adj.pt", remodel_adj)
        _case(
            specs,
            manifest,
            f"remodel_{base}",
            {"length": str(length), "ss_file": remodel_ss_path, "adj_file": remodel_adj_path},
            "remodeling_partial",
            f"Full fold with two masked remodeling windows: {description}",
            {"ss_labels": remodel_ss.tolist(), "contact_blocks": contacts, "masked_windows": windows},
        )

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
