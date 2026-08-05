#!/usr/bin/env python3
"""Publish the epoch-670 checkpoint + model card to Hugging Face.

Requires `pip install huggingface_hub` and either `huggingface-cli login`
or an `HF_TOKEN` environment variable with write access to the target repo.

Usage:
    python checkpoints/upload.py /work/gerstung/RFD3-adj-ckpt/epoch-0670_localattn.ckpt
    python checkpoints/upload.py --skip-checkpoint   # just resync the model card

Called automatically by .githooks/pre-push — see README "Keeping Hugging
Face in sync".
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

HF_REPO_ID = "Jetzu77/Harmonia"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        help="Path to epoch-0670_localattn.ckpt. Omit with --skip-checkpoint.",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Only resync the model card (README), skip the (large) checkpoint upload.",
    )
    parser.add_argument(
        "--private", action="store_true", default=True, help="Create repo as private (default)."
    )
    args = parser.parse_args()
    if not args.skip_checkpoint and args.checkpoint is None:
        parser.error("checkpoint path is required unless --skip-checkpoint is given")

    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, private=args.private, exist_ok=True)

    if not args.skip_checkpoint:
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

    print(f"Synced to https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
