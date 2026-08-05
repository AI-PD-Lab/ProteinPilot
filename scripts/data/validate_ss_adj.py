#!/usr/bin/env python3
"""
validate_ss_adj.py — Validate SS/ADJ LMDB content and spot-check tensors.

Usage:
    # Check counts and spot-check 100 random entries
    python scripts/validate_ss_adj.py /data/ss_adj.lmdb

    # Only check ADJ entries (faster)
    python scripts/validate_ss_adj.py /data/ss_adj.lmdb --check adj

    # Spot-check specific PDB IDs
    python scripts/validate_ss_adj.py /data/ss_adj.lmdb --pdbs 1abc 2xyz

    # Save a per-entry stats CSV
    python scripts/validate_ss_adj.py /data/ss_adj.lmdb --csv stats.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(description="Validate SS/ADJ LMDB entries.")
    p.add_argument("lmdb_path", help="Path to the SS/ADJ LMDB directory")
    p.add_argument(
        "--check",
        choices=["all", "ss", "adj"],
        default="all",
        help="Which keys to check (default: all)",
    )
    p.add_argument(
        "--n_spot",
        type=int,
        default=100,
        help="Number of random entries to spot-check (0 = none, -1 = all)",
    )
    p.add_argument(
        "--pdbs",
        nargs="+",
        metavar="PDB_ID",
        help="Specific PDB IDs to check (overrides --n_spot)",
    )
    p.add_argument(
        "--csv",
        metavar="PATH",
        help="Optional output CSV with per-entry statistics",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-entry details",
    )
    return p.parse_args()


def _iter_lmdb_keys(env):
    """Yield all string keys in an LMDB environment."""
    with env.begin() as txn:
        cursor = txn.cursor()
        for key_bytes, _ in cursor.iternext_dup() if False else cursor.iternext():
            yield key_bytes.decode("utf-8", errors="replace")


def _get(env, key: str) -> bytes | None:
    with env.begin() as txn:
        return txn.get(key.encode("utf-8"))


def _check_ss(pdb_id: str, env, verbose: bool) -> dict:
    from foundry.data.ss_adj import SS_HELIX, SS_LOOP, SS_STRAND, unpack_ss

    raw = _get(env, f"{pdb_id}_ss")
    if raw is None:
        return {"pdb_id": pdb_id, "has_ss": False, "error": "key missing"}

    try:
        ss = unpack_ss(raw)
    except Exception as e:
        return {"pdb_id": pdb_id, "has_ss": True, "error": str(e)}

    L = len(ss)
    unique_vals = set(np.unique(ss).tolist())
    valid_vals = {SS_HELIX, SS_STRAND, SS_LOOP}  # 0., 1., 2.  — 3. must NOT appear
    bad_vals = unique_vals - valid_vals

    result = {
        "pdb_id": pdb_id,
        "has_ss": True,
        "error": None,
        "L": L,
        "n_helix": int((ss == SS_HELIX).sum()),
        "n_strand": int((ss == SS_STRAND).sum()),
        "n_loop": int((ss == SS_LOOP).sum()),
        "bad_values": sorted(bad_vals),
    }

    if verbose:
        print(
            f"  {pdb_id}_ss: L={L}, H={result['n_helix']}, E={result['n_strand']}, "
            f"L={result['n_loop']}"
            + (f" WARN bad_values={bad_vals}" if bad_vals else "")
        )

    return result


def _check_adj(pdb_id: str, env, verbose: bool) -> dict:
    from foundry.data.ss_adj import unpack_adj

    raw = _get(env, f"{pdb_id}_adj")
    if raw is None:
        return {"pdb_id": pdb_id, "has_adj": False, "error": "key missing"}

    try:
        adj = unpack_adj(raw)
    except Exception as e:
        return {"pdb_id": pdb_id, "has_adj": True, "error": str(e)}

    L = adj.shape[0]
    is_square = adj.ndim == 2 and adj.shape[0] == adj.shape[1]
    is_symmetric = bool(np.allclose(adj, adj.T))
    diag_max = float(np.max(np.abs(np.diag(adj))))
    val_set = set(np.unique(adj).tolist())
    bad_vals = val_set - {0.0, 1.0}
    n_contacts = int(adj.sum()) // 2

    result = {
        "pdb_id": pdb_id,
        "has_adj": True,
        "error": None,
        "L": L,
        "is_square": is_square,
        "is_symmetric": is_symmetric,
        "diag_max": diag_max,
        "n_contacts": n_contacts,
        "bad_values": sorted(bad_vals),
    }

    if verbose:
        status = "OK" if (is_square and is_symmetric and diag_max < 1e-6 and not bad_vals) else "WARN"
        print(
            f"  {pdb_id}_adj: L={L}, contacts={n_contacts}, "
            f"sym={is_symmetric}, diag_max={diag_max:.3f} [{status}]"
            + (f" bad_values={bad_vals}" if bad_vals else "")
        )

    return result


def _check_loop_active(pdb_id: str, env, verbose: bool) -> dict:
    raw = _get(env, f"{pdb_id}_loop_adj_active")
    if raw is None:
        return {"pdb_id": pdb_id, "has_loop_flag": False}
    val = bool(raw == b"\x01")
    if verbose:
        print(f"  {pdb_id}_loop_adj_active: {val}")
    return {"pdb_id": pdb_id, "has_loop_flag": True, "loop_adj_active": val}


def main() -> int:
    args = _parse_args()

    try:
        import lmdb
    except ImportError:
        print("ERROR: lmdb not installed. Run: pip install lmdb", file=sys.stderr)
        return 1

    lmdb_path = Path(args.lmdb_path)
    if not lmdb_path.exists():
        print(f"ERROR: LMDB path not found: {lmdb_path}", file=sys.stderr)
        return 1

    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )

    # ─── Discover unique PDB IDs ───────────────────────────────────────────
    print(f"Scanning LMDB: {lmdb_path} ...")
    all_keys = list(_iter_lmdb_keys(env))
    suffix_map: dict[str, set] = {}
    for key in all_keys:
        for sfx in ("_ss", "_adj", "_loop_adj_active"):
            if key.endswith(sfx):
                pdb_id = key[: -len(sfx)]
                suffix_map.setdefault(pdb_id, set()).add(sfx)
                break

    n_total = len(suffix_map)
    print(f"Found {len(all_keys)} keys across {n_total} unique PDB IDs")

    ss_count   = sum(1 for v in suffix_map.values() if "_ss" in v)
    adj_count  = sum(1 for v in suffix_map.values() if "_adj" in v)
    loop_count = sum(1 for v in suffix_map.values() if "_loop_adj_active" in v)
    print(f"  {ss_count} with _ss")
    print(f"  {adj_count} with _adj")
    print(f"  {loop_count} with _loop_adj_active")

    # ─── Select which PDB IDs to spot-check ──────────────────────────────
    if args.pdbs:
        check_ids = args.pdbs
    elif args.n_spot == 0:
        check_ids = []
    elif args.n_spot == -1:
        check_ids = list(suffix_map.keys())
    else:
        rng = np.random.default_rng(42)
        all_ids = list(suffix_map.keys())
        n = min(args.n_spot, len(all_ids))
        check_ids = rng.choice(all_ids, size=n, replace=False).tolist()

    # ─── Spot-check ────────────────────────────────────────────────────────
    stats = []
    n_ss_errors = n_adj_errors = 0
    print(f"\nSpot-checking {len(check_ids)} entries ...")

    for pdb_id in check_ids:
        entry = {"pdb_id": pdb_id}

        if args.check in ("all", "ss") and "_ss" in suffix_map.get(pdb_id, set()):
            ss_res = _check_ss(pdb_id, env, args.verbose)
            entry.update({k: v for k, v in ss_res.items() if k != "pdb_id"})
            if ss_res.get("error") or ss_res.get("bad_values"):
                n_ss_errors += 1

        if args.check in ("all", "adj") and "_adj" in suffix_map.get(pdb_id, set()):
            adj_res = _check_adj(pdb_id, env, args.verbose)
            entry.update(
                {f"adj_{k}": v for k, v in adj_res.items() if k not in ("pdb_id", "L")}
            )
            if adj_res.get("error") or not adj_res.get("is_symmetric", True) or adj_res.get("bad_values"):
                n_adj_errors += 1

        if "_loop_adj_active" in suffix_map.get(pdb_id, set()):
            loop_res = _check_loop_active(pdb_id, env, args.verbose)
            entry["loop_adj_active"] = loop_res.get("loop_adj_active")

        stats.append(entry)

    env.close()

    # ─── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Spot-checked: {len(stats)}")
    print(f"SS errors:    {n_ss_errors}")
    print(f"ADJ errors:   {n_adj_errors}")

    if stats and args.check in ("all", "ss"):
        lengths = [s["L"] for s in stats if "L" in s]
        if lengths:
            print(f"Length range: {min(lengths)} – {max(lengths)}")

    if n_ss_errors + n_adj_errors == 0 and len(check_ids) > 0:
        print("✓  All spot-checked entries passed validation")
    elif len(check_ids) == 0:
        print("(No entries checked — use --n_spot to spot-check)")

    # ─── CSV output ────────────────────────────────────────────────────────
    if args.csv and stats:
        import csv

        csv_path = Path(args.csv)
        all_keys_set = set()
        for s in stats:
            all_keys_set.update(s.keys())
        fieldnames = ["pdb_id"] + sorted(all_keys_set - {"pdb_id"})

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(stats)
        print(f"\nWrote CSV: {csv_path}")

    return 1 if (n_ss_errors + n_adj_errors) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
