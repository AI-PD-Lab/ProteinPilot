#!/usr/bin/env python3
"""Download the HARMONIA RFD3 SS/ADJ checkpoint (epoch 670) from Hugging Face.

Usage:
    python checkpoints/download.py [--dest checkpoints/epoch-0670_localattn.ckpt]

Requires `pip install huggingface_hub` and, if the HF repo is private,
either `huggingface-cli login` or an `HF_TOKEN` environment variable.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO_ID = "Jetzu77/Harmonia"
HF_FILENAME = "epoch-0670_localattn.ckpt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent / HF_FILENAME,
        help="Where to place the downloaded checkpoint.",
    )
    args = parser.parse_args()

    cached_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    if Path(cached_path) != args.dest:
        shutil.copy2(cached_path, args.dest)

    print(f"Checkpoint ready at {args.dest}")


if __name__ == "__main__":
    main()
