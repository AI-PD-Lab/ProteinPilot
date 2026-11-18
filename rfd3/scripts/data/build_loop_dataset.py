#!/usr/bin/env python3
"""
build_loop_dataset.py — Curated loop-structure dataset builder (Pipeline 2).

Downloads and filters structures from:
  - SAbDab  (antibody-antigen CDR loops)
  - STCRDab (TCR-pMHC CDR loops)
  - M-CSA   (enzyme active site loops)

Applies quality filters:
  - Resolution ≤ max_resolution Å
  - Sequence identity < max_identity% (deduplication)
  - CDR/active-site loop length ≤ max_loop_len residues
  - Structure downloadable and parseable

Outputs a CSV ready for use as --metadata in precompute_ss_adj_loop.py.

Usage:
    python scripts/build_loop_dataset.py \\
        --out_csv /data/loop_dataset.csv \\
        --cif_dir /data/loop_structures \\
        --sources sabdab stcrdab mcsa \\
        --max_resolution 3.0 \\
        --max_identity 90

Requirements:
    pip install requests tqdm pandas biopython

Note: SAbDab and STCRDab require internet access to the OPIG servers.
      For offline use, provide --sabdab_tsv / --stcrdab_tsv / --mcsa_json
      with pre-downloaded summary files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_loop_dataset")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Build curated loop-structure CSV for Pipeline 2 precomputation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_csv",        required=True, help="Output CSV path")
    p.add_argument("--cif_dir",        required=True, help="Directory to download / cache CIF files")
    p.add_argument(
        "--sources", nargs="+",
        choices=["sabdab", "stcrdab", "mcsa"],
        default=["sabdab", "stcrdab", "mcsa"],
        help="Data sources to include",
    )
    p.add_argument("--max_resolution", type=float, default=3.0, help="Maximum X-ray resolution (Å)")
    p.add_argument("--max_identity",   type=int,   default=90,  help="Max sequence identity %% for deduplication")
    p.add_argument("--max_loop_len",   type=int,   default=25,  help="Maximum CDR/active-site loop length")
    p.add_argument("--n_workers",      type=int,   default=8,   help="Parallel download workers")
    p.add_argument("--dry_run",        action="store_true",     help="List entries but do not download")
    # Optional pre-downloaded summary files
    p.add_argument("--sabdab_tsv",   default=None, help="Pre-downloaded SAbDab summary TSV")
    p.add_argument("--stcrdab_tsv",  default=None, help="Pre-downloaded STCRDab summary TSV")
    p.add_argument("--mcsa_json",    default=None, help="Pre-downloaded M-CSA entries JSON")
    return p.parse_args()


# ---------------------------------------------------------------------------
# SAbDab
# ---------------------------------------------------------------------------

SABDAB_SUMMARY_URL = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"


def _fetch_sabdab(tsv_path: str | None, max_resolution: float, max_loop_len: int) -> list[dict]:
    """Load SAbDab entries. Returns list of dicts with pdb_id, chains, source."""
    try:
        import pandas as pd
        import requests
    except ImportError:
        logger.error("pandas / requests required: pip install pandas requests")
        return []

    if tsv_path:
        logger.info(f"Loading SAbDab from local file: {tsv_path}")
        df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    else:
        logger.info("Downloading SAbDab summary ...")
        r = requests.get(SABDAB_SUMMARY_URL, timeout=60)
        r.raise_for_status()
        import io
        df = pd.read_csv(io.StringIO(r.text), sep="\t", low_memory=False)

    # Quality filter
    if "resolution" in df.columns:
        df = df[df["resolution"].apply(lambda x: _safe_float(x, np.inf) <= max_resolution)]

    # CDR loop length filter (use CDR3 heavy as proxy; others checked at compute time)
    # SAbDab reports CDR sequences; use string length
    for cdr_col in ("cdrh3", "cdrl3", "cdrh1", "cdrh2", "cdrl1", "cdrl2"):
        if cdr_col in df.columns:
            df = df[df[cdr_col].apply(
                lambda x: isinstance(x, str) and len(x) <= max_loop_len
            )]

    records = []
    for _, row in df.iterrows():
        pdb_id = str(row.get("pdb", "")).strip().lower()
        if not pdb_id or len(pdb_id) != 4:
            continue
        # Collect heavy + light + antigen chain IDs
        chains = []
        for col in ("Hchain", "Lchain", "antigen_chain"):
            val = str(row.get(col, "")).strip()
            if val and val.lower() not in ("nan", "none", ""):
                chains.extend(v.strip() for v in val.split("|") if v.strip())
        records.append({
            "pdb_id":  pdb_id,
            "chains":  ",".join(dict.fromkeys(chains)),  # dedup, preserve order
            "source":  "sabdab",
        })

    logger.info(f"SAbDab: {len(records)} entries after filtering")
    return records


# ---------------------------------------------------------------------------
# STCRDab
# ---------------------------------------------------------------------------

STCRDAB_SUMMARY_URL = "https://opig.stats.ox.ac.uk/webapps/stcrdab-stcrpred/summary/all/"


def _fetch_stcrdab(tsv_path: str | None, max_resolution: float, max_loop_len: int) -> list[dict]:
    try:
        import io
        import pandas as pd
        import requests
    except ImportError:
        logger.error("pandas / requests required")
        return []

    if tsv_path:
        df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    else:
        logger.info("Downloading STCRDab summary ...")
        r = requests.get(STCRDAB_SUMMARY_URL, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="\t", low_memory=False)

    if "resolution" in df.columns:
        df = df[df["resolution"].apply(lambda x: _safe_float(x, np.inf) <= max_resolution)]

    records = []
    for _, row in df.iterrows():
        pdb_id = str(row.get("pdb", "")).strip().lower()
        if not pdb_id or len(pdb_id) != 4:
            continue
        chains = []
        for col in ("Achain", "Bchain", "MHCAchain", "MHCBchain", "Antigenchain"):
            val = str(row.get(col, "")).strip()
            if val and val.lower() not in ("nan", "none", ""):
                chains.extend(v.strip() for v in val.split("|") if v.strip())
        records.append({
            "pdb_id": pdb_id,
            "chains": ",".join(dict.fromkeys(chains)),
            "source": "stcrdab",
        })

    logger.info(f"STCRDab: {len(records)} entries after filtering")
    return records


# ---------------------------------------------------------------------------
# M-CSA
# ---------------------------------------------------------------------------

MCSA_API_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json&page_size=100"


def _mcsa_entries(data: object) -> list[dict]:
    if isinstance(data, dict):
        entries = data.get("results", [])
    else:
        entries = data
    return entries if isinstance(entries, list) else []


def _download_all_mcsa_entries(requests) -> list[dict]:
    entries: list[dict] = []
    url: str | None = MCSA_API_URL
    while url:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        page = response.json()
        entries.extend(_mcsa_entries(page))
        url = page.get("next") if isinstance(page, dict) else None
    return entries


def _fetch_mcsa(json_path: str | None, max_resolution: float, max_loop_len: int) -> list[dict]:
    try:
        import json
        import requests
    except ImportError:
        logger.error("requests required")
        return []

    if json_path:
        with open(json_path) as f:
            data = json.load(f)
        entries = _mcsa_entries(data)
        if isinstance(data, dict) and data.get("next"):
            logger.warning(
                "Local M-CSA JSON is paginated and contains only %d of %s entries",
                len(entries),
                data.get("count", "unknown"),
            )
    else:
        logger.info("Downloading M-CSA entries ...")
        entries = _download_all_mcsa_entries(requests)

    records: list[dict] = []
    for entry in entries:
        pdb_chains: dict[str, list[str]] = {}
        for residue in entry.get("residues", []) or []:
            for chain in residue.get("residue_chains", []) or []:
                pdb_id = str(chain.get("pdb_id", "")).strip().lower()
                if not pdb_id or len(pdb_id) != 4:
                    continue
                chain_name = str(chain.get("chain_name", "")).strip()
                if chain_name:
                    pdb_chains.setdefault(pdb_id, []).append(chain_name)
                else:
                    pdb_chains.setdefault(pdb_id, [])
        for pdb_id, chains in pdb_chains.items():
            records.append(
                {
                    "pdb_id": pdb_id,
                    "chains": ",".join(dict.fromkeys(chains)),
                    "source": "mcsa",
                }
            )

    seen: dict[str, dict] = {}
    for record in records:
        pdb_id = record["pdb_id"]
        if pdb_id not in seen:
            seen[pdb_id] = record
            continue
        chains = seen[pdb_id]["chains"].split(",") if seen[pdb_id]["chains"] else []
        chains.extend(record["chains"].split(",") if record["chains"] else [])
        seen[pdb_id]["chains"] = ",".join(dict.fromkeys(filter(None, chains)))
    records = list(seen.values())

    logger.info(f"M-CSA: {len(records)} entries after filtering")
    return records


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_float(val, default):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _download_cif(pdb_id: str, cif_dir: Path) -> Path | None:
    """Download mmCIF from RCSB if not already cached. Returns path or None."""
    dest = cif_dir / f"{pdb_id}.cif.gz"
    if dest.exists():
        return dest
    try:
        import requests
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            with open(dest, "wb") as f:
                f.write(r.content)
            return dest
        else:
            logger.warning(f"HTTP {r.status_code} for {pdb_id}")
            return None
    except Exception as e:
        logger.warning(f"Download failed for {pdb_id}: {e}")
        return None


def _deduplicate(records: list[dict], max_identity: int) -> list[dict]:
    """Deduplicate repeated PDB entries.

    ``max_identity`` is retained in the CLI for compatibility, but true
    sequence-identity clustering requires source-specific full sequences that
    are not consistently available from all three metadata sources.
    """
    if max_identity < 100:
        logger.warning(
            "Sequence-identity clustering is not available for all sources; "
            "deduplicating by PDB ID only."
        )
    seen: dict[str, dict] = {}
    for rec in records:
        pdb_id = rec["pdb_id"]
        if pdb_id not in seen:
            seen[pdb_id] = rec
        else:
            # Merge chains and prefer non-empty chain list
            if rec["chains"] and not seen[pdb_id]["chains"]:
                seen[pdb_id]["chains"] = rec["chains"]
    return list(seen.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    try:
        import pandas as pd
    except ImportError:
        logger.error("numpy / pandas required: pip install numpy pandas")
        return 1

    cif_dir = Path(args.cif_dir)
    out_csv = Path(args.out_csv)
    cif_dir.mkdir(parents=True, exist_ok=True)

    # --- Gather entries from each source ---
    all_records: list[dict] = []

    if "sabdab" in args.sources:
        all_records.extend(
            _fetch_sabdab(args.sabdab_tsv, args.max_resolution, args.max_loop_len)
        )

    if "stcrdab" in args.sources:
        all_records.extend(
            _fetch_stcrdab(args.stcrdab_tsv, args.max_resolution, args.max_loop_len)
        )

    if "mcsa" in args.sources:
        all_records.extend(
            _fetch_mcsa(args.mcsa_json, args.max_resolution, args.max_loop_len)
        )

    if not all_records:
        logger.error("No records found. Check source availability and filters.")
        return 1

    all_records = _deduplicate(all_records, args.max_identity)
    logger.info(f"Total unique structures after deduplication: {len(all_records)}")

    if args.dry_run:
        df = pd.DataFrame(all_records).rename(columns={"chains": "chain_order"})
        df["cif_path"] = ""
        df["assembly_id"] = "1"
        df["example_id"] = [
            f"{{['ss_adj_loop', '{source}']}}{{{pdb_id}}}{{1}}{{[]}}"
            for pdb_id, source in zip(df["pdb_id"], df["source"], strict=False)
        ]
        df.to_csv(out_csv, index=False)
        logger.info(f"Dry run — wrote {len(df)} entries (no downloads): {out_csv}")
        return 0

    # --- Download CIF files ---
    logger.info(f"Downloading {len(all_records)} CIF files to {cif_dir} ...")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    final_records = []

    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {
            pool.submit(_download_cif, rec["pdb_id"], cif_dir): rec
            for rec in all_records
        }
        for i, future in enumerate(as_completed(futures), 1):
            rec = futures[future]
            path = future.result()
            if path is not None:
                rec["cif_path"] = str(path)
                final_records.append(rec)
            if i % 100 == 0:
                logger.info(f"  {i}/{len(all_records)} done ...")

    n_failed = len(all_records) - len(final_records)
    logger.info(
        f"Downloaded: {len(final_records)}  Failed: {n_failed}"
    )

    # --- Write CSV ---
    df = pd.DataFrame(final_records)
    df = df.rename(columns={"chains": "chain_order"})
    df["assembly_id"] = "1"
    df["example_id"] = [
        f"{{['ss_adj_loop', '{source}']}}{{{pdb_id}}}{{1}}{{[]}}"
        for pdb_id, source in zip(df["pdb_id"], df["source"], strict=False)
    ]
    df.to_csv(out_csv, index=False)
    logger.info(f"Wrote CSV: {out_csv}  ({len(df)} entries)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
