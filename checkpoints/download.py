#!/usr/bin/env python3
"""Download the Harmonia release checkpoints from Hugging Face.

    python checkpoints/download.py --model rfd3            # -> checkpoints/Harmonia-RFD3.ckpt
    python checkpoints/download.py --model rfd4            # -> checkpoints/Harmonia-RFD4/{config.json,model.safetensors}
    python checkpoints/download.py --model all

The files live in the Hugging Face bucket ``AI-PD/Harmonia`` (https://huggingface.co/buckets/AI-PD/Harmonia).
Requires ``pip install huggingface_hub`` and, while the bucket is private, ``huggingface-cli login`` or an ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

BUCKET = "AI-PD/Harmonia"
FILES = {
    "rfd3": ["Harmonia-RFD3.ckpt"],
    "rfd4": ["Harmonia-RFD4/config.json", "Harmonia-RFD4/model.safetensors"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["rfd3", "rfd4", "all"], default="all")
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent, help="directory the files are placed in")
    args = ap.parse_args()

    from huggingface_hub import download_bucket_files

    names = FILES["rfd3"] + FILES["rfd4"] if args.model == "all" else FILES[args.model]
    files = []
    for n in names:
        out = args.dest / n
        out.parent.mkdir(parents=True, exist_ok=True)
        files.append((n, str(out)))
    download_bucket_files(BUCKET, files=files, raise_on_missing_files=True)
    for _, out in files:
        print("ready:", out)


if __name__ == "__main__":
    main()
