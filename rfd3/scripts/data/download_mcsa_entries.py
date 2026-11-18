#!/usr/bin/env python3
"""Download a complete, reproducible snapshot of the paginated M-CSA API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


DEFAULT_URL = (
    "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/"
    "?format=json&page_size=100"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries: list[dict] = []
    url: str | None = args.url
    page_count = 0

    while url:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        page = response.json()
        page_entries = page.get("results", [])
        if not isinstance(page_entries, list):
            raise RuntimeError(f"Unexpected M-CSA response on page {page_count + 1}")
        entries.extend(page_entries)
        page_count += 1
        print(f"Downloaded page {page_count}: {len(entries)} entries")
        url = page.get("next")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "count": len(entries),
                "next": None,
                "previous": None,
                "results": entries,
            }
        )
    )
    print(f"Wrote {len(entries)} M-CSA entries to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
