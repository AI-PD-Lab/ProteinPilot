#!/usr/bin/env python3
"""Publish the epoch-670 checkpoint + model card to Hugging Face.

Requires `pip install huggingface_hub` and either `huggingface-cli login`
or an `HF_TOKEN` environment variable with write access to the target repo.

Usage:
    python checkpoints/upload.py /work/gerstung/RFD3-adj-ckpt/epoch-0670_localattn.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

HF_REPO_ID = "Jetzu77/Harmonia"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to epoch-0670_localattn.ckpt")
    parser.add_argument(
        "--private", action="store_true", default=True, help="Create repo as private (default)."
    )
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, private=args.private, exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(args.checkpoint),
        path_in_repo="epoch-0670_localattn.ckpt",
        repo_id=HF_REPO_ID,
    )
    api.upload_file(
        path_or_fileobj=str(Path(__file__).resolve().parent / "MODEL_CARD.md"),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
    )

    print(f"Uploaded to https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
