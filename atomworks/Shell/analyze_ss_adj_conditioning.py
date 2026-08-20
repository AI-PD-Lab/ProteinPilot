#!/usr/bin/env python3
"""Analyze RFdiffusion3 SS/ADJ conditioning benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdbx import CIFFile, get_structure

SS_TO_SSE = {0: "a", 1: "b", 2: "c"}


def _load_cif(path: Path):
    with gzip.open(path, "rt") as handle:
        return get_structure(CIFFile.read(handle), model=1)


def _case_key_from_name(name: str, manifest: dict[str, Any]) -> str | None:
    stem = name.removesuffix(".cif.gz").removesuffix(".cif")
    match = re.match(r"(.+)_\d+_model_\d+$", stem)
    if match and match.group(1) in manifest:
        return match.group(1)
    for key in sorted(manifest, key=len, reverse=True):
        if stem.startswith(key + "_"):
            return key
    return None


def _protein_residue_arrays(atom_array):
    protein = atom_array[struc.filter_amino_acids(atom_array)]
    if protein.array_length() == 0:
        return None, None, None
    residue_starts = struc.get_residue_starts(protein)
    sse = struc.annotate_sse(protein)
    coords = []
    for i, start in enumerate(residue_starts):
        end = residue_starts[i + 1] if i + 1 < len(residue_starts) else protein.array_length()
        res = protein[start:end]
        cb = res.coord[res.atom_name == "CB"]
        ca = res.coord[res.atom_name == "CA"]
        if len(cb):
            coords.append(cb[0])
        elif len(ca):
            coords.append(ca[0])
        else:
            coords.append(np.full(3, np.nan))
    return protein, sse, np.asarray(coords, dtype=float)


def _ss_accuracy(sse: np.ndarray, target_labels: list[int] | None) -> tuple[float, int]:
    if target_labels is None:
        return math.nan, 0
    n = min(len(sse), len(target_labels))
    if n == 0:
        return math.nan, 0
    target = np.asarray(target_labels[:n], dtype=int)
    mask = np.isin(target, [0, 1, 2])
    if not np.any(mask):
        return math.nan, 0
    expected = np.asarray([SS_TO_SSE[int(v)] for v in target[mask]])
    return float(np.mean(sse[:n][mask] == expected)), int(np.sum(mask))


def _ss_segments_to_labels(length: int, segments: list[list[int]] | None) -> list[int] | None:
    if not segments:
        return None
    labels = np.full(length, 3, dtype=int)
    for start, end, label in segments:
        labels[start - 1 : end] = int(label)
    return labels.tolist()


def _adj_recovery(coords: np.ndarray, contact_blocks: list[list[list[int]]] | None, threshold: float) -> tuple[float, int, float]:
    if coords is None or not contact_blocks:
        return math.nan, 0, math.nan
    recovered = []
    min_dists = []
    n_res = coords.shape[0]
    for first, second in contact_blocks:
        a0, a1 = max(0, first[0] - 1), min(n_res, first[1])
        b0, b1 = max(0, second[0] - 1), min(n_res, second[1])
        if a0 >= a1 or b0 >= b1:
            continue
        block_a = coords[a0:a1]
        block_b = coords[b0:b1]
        if not np.isfinite(block_a).all() or not np.isfinite(block_b).all():
            continue
        d = np.linalg.norm(block_a[:, None, :] - block_b[None, :, :], axis=-1)
        min_dist = float(np.min(d))
        min_dists.append(min_dist)
        recovered.append(min_dist <= threshold)
    if not recovered:
        return math.nan, 0, math.nan
    return float(np.mean(recovered)), len(recovered), float(np.mean(min_dists))



# --------------------------------------------------------------------------
# Typed adjacency metrics (protein / ligand / nucleic acid)
# --------------------------------------------------------------------------
# The original _adj_recovery is protein-only (C-beta, one coordinate per residue)
# and scores POSITIVE pairs only. Two gaps that matter:
#   * ligand and NA pairs are invisible, yet they are ~56% of the effective
#     training distribution once cluster sampling is accounted for;
#   * there is no number at all for over-association, i.e. explicitly
#     non-adjacent blocks that the model put in contact anyway.

_NA_RES = {"A","U","G","C","DA","DT","DG","DC","I","N"}


def _residue_groups(atom_array):
    """Per-residue heavy atoms, contact coordinate, and molecule class.

    Returns (heavy_atoms, contact_coord, kind) where heavy_atoms is a list of
    (n_i, 3) arrays, contact_coord is (L, 3) (C-beta for protein, falling back to
    C-alpha), and kind is a list of "protein" | "na" | "ligand".
    """
    n = atom_array.array_length()
    if n == 0:
        return [], np.zeros((0, 3)), []
    starts = struc.get_residue_starts(atom_array)
    heavy, contact, kind = [], [], []
    is_h = atom_array.element == "H"
    for i, st in enumerate(starts):
        en = starts[i + 1] if i + 1 < len(starts) else n
        res = atom_array[st:en]
        hv = res.coord[~is_h[st:en]]
        heavy.append(hv if len(hv) else res.coord)
        rn = str(res.res_name[0])
        if struc.filter_amino_acids(res).any():
            k = "protein"
        elif rn in _NA_RES:
            k = "na"
        else:
            k = "ligand"
        kind.append(k)
        cb = res.coord[res.atom_name == "CB"]
        ca = res.coord[res.atom_name == "CA"]
        if k == "protein" and len(cb):
            contact.append(cb[0])
        elif k == "protein" and len(ca):
            contact.append(ca[0])
        else:
            contact.append(hv.mean(axis=0) if len(hv) else np.full(3, np.nan))
    return heavy, np.asarray(contact, dtype=float), kind


def _pair_kind(k1: str, k2: str) -> str:
    return "-".join(sorted((k1, k2)))


# cutoff per pair kind; mirrors foundry.data.ss_adj.DEFAULT_CONTACT_RULES
_PAIR_CUTOFF = {
    "protein-protein": 6.0,   # C-beta, RFD1 rule
    "ligand-protein": 4.5,    # heavy atom, van der Waals
    "na-protein": 5.0,
    "na-na": 5.0,
    "ligand-na": 4.5,
    "ligand-ligand": 4.5,
}


def _block_min_dist(blk_a, blk_b, heavy, contact, kind) -> tuple[float, str]:
    """Min distance for a block pair, using the atoms its pair kind calls for."""
    ka = kind[blk_a[0]] if blk_a else "protein"
    kb = kind[blk_b[0]] if blk_b else "protein"
    pk = _pair_kind(ka, kb)
    if pk == "protein-protein":
        a = contact[blk_a]
        b = contact[blk_b]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return math.nan, pk
    else:
        a = np.concatenate([heavy[i] for i in blk_a], axis=0)
        b = np.concatenate([heavy[i] for i in blk_b], axis=0)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(np.min(d)), pk


def _expand(block, n_res):
    lo, hi = max(0, block[0] - 1), min(n_res, block[1])
    return list(range(lo, hi)) if lo < hi else []


def _typed_adj_metrics(atom_array, contact_blocks, non_contact_blocks):
    """Per-pair-kind recovery, plus the non-adjacency violation rate.

    recovery       : conditioned contacts actually realised (higher better)
    violation rate : explicitly NON-adjacent pairs the model put in contact
                     (lower better). This is the number that was missing -- there
                     was no measurement of over-association at all.
    """
    heavy, contact, kind = _residue_groups(atom_array)
    n_res = len(kind)
    out: dict[str, float] = {}
    if n_res == 0:
        return out

    for label, blocks, want_contact in (
        ("recovery", contact_blocks or [], True),
        ("violation", non_contact_blocks or [], False),
    ):
        per_kind: dict[str, list[bool]] = {}
        for first, second in blocks:
            ia, ib = _expand(first, n_res), _expand(second, n_res)
            if not ia or not ib:
                continue
            d, pk = _block_min_dist(ia, ib, heavy, contact, kind)
            if not math.isfinite(d):
                continue
            cut = _PAIR_CUTOFF.get(pk, 6.0)
            in_contact = d <= cut
            per_kind.setdefault(pk, []).append(
                in_contact if want_contact else in_contact
            )
        allv = [v for vs in per_kind.values() for v in vs]
        if allv:
            out[f"adj_{label}_all"] = float(np.mean(allv))
            out[f"n_{label}_pairs"] = len(allv)
        for pk, vs in sorted(per_kind.items()):
            out[f"adj_{label}_{pk}"] = float(np.mean(vs))
            out[f"n_{label}_{pk}"] = len(vs)
    return out


def _quality(protein, coords: np.ndarray, sse: np.ndarray) -> dict[str, float]:
    ca = []
    residue_starts = struc.get_residue_starts(protein)
    for i, start in enumerate(residue_starts):
        end = residue_starts[i + 1] if i + 1 < len(residue_starts) else protein.array_length()
        res = protein[start:end]
        c = res.coord[res.atom_name == "CA"]
        if len(c):
            ca.append(c[0])
    ca = np.asarray(ca, dtype=float)
    n_res = len(sse)
    helix = int(np.sum(sse == "a"))
    sheet = int(np.sum(sse == "b"))
    loop = int(np.sum(sse == "c"))
    structured = helix + sheet
    ss_ratio = float(sheet / structured) if structured else math.nan
    if len(ca) >= 2:
        diffs = np.linalg.norm(np.diff(ca, axis=0), axis=-1)
        n_chainbreaks = int(np.sum(diffs > 4.2))
        max_ca_dev = float(np.max(np.abs(diffs - 3.8)))
    else:
        n_chainbreaks = 0
        max_ca_dev = math.nan
    clashes = 0
    if len(ca) >= 4:
        for i in range(len(ca) - 3):
            d = np.linalg.norm(ca[i + 3 :] - ca[i], axis=-1)
            clashes += int(np.sum(d < 3.5))
    rg = float(np.sqrt(np.mean(np.sum((ca - ca.mean(0)) ** 2, axis=1)))) if len(ca) else math.nan
    return {
        "n_res": n_res,
        "helix_frac": helix / n_res if n_res else math.nan,
        "sheet_frac": sheet / n_res if n_res else math.nan,
        "loop_frac": loop / n_res if n_res else math.nan,
        "ss_ratio_achieved": ss_ratio,
        "n_chainbreaks": n_chainbreaks,
        "ca_clashes": clashes,
        "max_ca_deviation": max_ca_dev,
        "radius_of_gyration": rg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--adj-threshold", type=float, default=6.0)
    args = parser.parse_args()

    manifest_path = args.out_root / "ss_adj_conditioning_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    design_dir = args.out_root / "designs"
    rows = []

    for cif in sorted(design_dir.glob("*.cif.gz")):
        key = _case_key_from_name(cif.name, manifest)
        if key is None:
            continue
        target = manifest[key].get("target", {})
        atom_array = _load_cif(cif)
        protein, sse, coords = _protein_residue_arrays(atom_array)
        if protein is None:
            continue
        target_labels = target.get("ss_labels")
        if target_labels is None:
            target_labels = _ss_segments_to_labels(manifest[key].get("length", len(sse)), target.get("ss_segments"))
        ss_acc, n_ss = _ss_accuracy(sse, target_labels)
        adj_rec, n_contacts, adj_min_dist = _adj_recovery(coords, target.get("contact_blocks"), args.adj_threshold)
        # typed metrics: per-pair-kind recovery + non-adjacency violation rate.
        # Degrades to {} on manifests without non_contact_blocks, so old runs work.
        typed = _typed_adj_metrics(
            atom_array,
            target.get("contact_blocks"),
            target.get("non_contact_blocks"),
        )
        quality = _quality(protein, coords, sse)
        target_ss_ratio = target.get("ss_ratio")
        ss_ratio_error = math.nan
        if target_ss_ratio is not None and quality["ss_ratio_achieved"] == quality["ss_ratio_achieved"]:
            ss_ratio_error = abs(float(target_ss_ratio) - quality["ss_ratio_achieved"])
        rows.append(
            {
                "case": key,
                "mode": manifest[key]["mode"],
                "file": cif.name,
                "ss_accuracy": ss_acc,
                "n_ss_conditioned": n_ss,
                "adj_block_recovery": adj_rec,
                "n_adj_blocks": n_contacts,
                "adj_mean_min_distance": adj_min_dist,
                **typed,
                "target_ss_ratio": target_ss_ratio if target_ss_ratio is not None else "",
                "ss_ratio_error": ss_ratio_error,
                **quality,
            }
        )

    summary_dir = args.out_root / "analysis"
    summary_dir.mkdir(parents=True, exist_ok=True)
    per_design_csv = summary_dir / "per_design_metrics.csv"
    if rows:
        with per_design_csv.open("w", newline="") as handle:
            # Union over ALL rows, not row 0: typed metrics are per-pair-kind, so a
            # ligand case or one with explicit negatives emits keys the others lack.
            # Row-0 fieldnames raised 'dict contains fields not in fieldnames' and
            # killed the entire summary after the designs had already been made.
            _fields = list(dict.fromkeys(k for _r in rows for k in _r))
            writer = csv.DictWriter(handle, fieldnames=_fields, restval="")
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["case"])].append(row)

    summary_rows = []
    metrics = [
        "ss_accuracy",
        "adj_block_recovery",
        "adj_mean_min_distance",
        "ss_ratio_achieved",
        "ss_ratio_error",
        "helix_frac",
        "sheet_frac",
        "loop_frac",
        "n_chainbreaks",
        "ca_clashes",
        "max_ca_deviation",
        "radius_of_gyration",
    ]
    # Typed metrics are discovered, not hardcoded: which per-pair-kind keys exist
    # depends on the manifest (a ligand case emits adj_recovery_ligand-protein, a
    # strict_no case emits adj_violation_*). A fixed list silently dropped every
    # one of them from the summary while they sat correctly in per_design_metrics.
    metrics += sorted(
        {
            k
            for r in rows
            for k in r
            if (k.startswith(("adj_recovery_", "adj_violation_")) or k in ("adj_recovery_all", "adj_violation_all"))
            and k not in metrics
        }
    )
    for (mode, case), group_rows in sorted(grouped.items()):
        out = {"mode": mode, "case": case, "n": len(group_rows)}
        for metric in metrics:
            # .get, not [...]: typed keys are per-case, so a row in this group may
            # legitimately lack the metric entirely rather than hold a blank.
            vals = np.asarray(
                [
                    float(r.get(metric, ""))
                    for r in group_rows
                    if r.get(metric, "") != "" and float(r.get(metric, "nan")) == float(r.get(metric, "nan"))
                ],
                dtype=float,
            )
            out[f"{metric}_mean"] = float(vals.mean()) if len(vals) else math.nan
            out[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else math.nan
        summary_rows.append(out)

    summary_csv = summary_dir / "summary_by_case.csv"
    if summary_rows:
        with summary_csv.open("w", newline="") as handle:
            # Union over ALL rows, not row 0: typed metrics are per-pair-kind, so a
            # ligand case or one with explicit negatives emits keys the others lack.
            # Row-0 fieldnames raised 'dict contains fields not in fieldnames' and
            # killed the entire summary after the designs had already been made.
            _fields = list(dict.fromkeys(k for _r in summary_rows for k in _r))
            writer = csv.DictWriter(handle, fieldnames=_fields, restval="")
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"designs_analyzed={len(rows)}")
    print(f"per_design_csv={per_design_csv}")
    print(f"summary_csv={summary_csv}")
    print("mode,case,n,ss_acc,adj_recovery,ssratio,chainbreaks,clashes")
    for row in summary_rows:
        print(
            f"{row['mode']},{row['case']},{row['n']},"
            f"{row['ss_accuracy_mean']:.3f},{row['adj_block_recovery_mean']:.3f},"
            f"{row['ss_ratio_achieved_mean']:.3f},{row['n_chainbreaks_mean']:.2f},"
            f"{row['ca_clashes_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
