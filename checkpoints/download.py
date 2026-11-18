#!/usr/bin/env python3
"""Download the Harmonia release checkpoints from Hugging Face.

    python checkpoints/download.py --model rfd3            # -> checkpoints/Harmonia-RFD3.ckpt
    python checkpoints/download.py --model rfd4            # -> checkpoints/Harmonia-RFD4/{config.json,model.safetensors}
    python checkpoints/download.py --model all

The files live in the Hugging Face model repo ``AI-PD/Harmonia`` (https://huggingface.co/AI-PD/Harmonia).
Requires ``pip install huggingface_hub`` and, while the repo is private, ``huggingface-cli login`` or an ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "AI-PD/Harmonia"
FILES = {
    "rfd3": ["Harmonia-RFD3.ckpt"],
    "rfd4": ["Harmonia-RFD4/config.json", "Harmonia-RFD4/model.safetensors"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["rfd3", "rfd4", "all"], default="all")
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent, help="directory the files are placed in")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    names = FILES["rfd3"] + FILES["rfd4"] if args.model == "all" else FILES[args.model]
    for n in names:
        path = hf_hub_download(repo_id=REPO_ID, filename=n, local_dir=args.dest)
        print("ready:", path)


if __name__ == "__main__":
    main()
