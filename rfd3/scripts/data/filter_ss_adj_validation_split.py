#!/usr/bin/env python3
"""Create a validation split whose requested crop centers exist in each assembly."""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

_DATASET = None


def _check_index(index: int) -> tuple[int, bool, str]:
    row = _DATASET.data.iloc[index]
    try:
        data = _DATASET.loader(row)
        query = set(data.get("query_pn_unit_iids") or [])
        available = set(data["atom_array"].pn_unit_iid.tolist())
        missing = sorted(query - available)
        if missing:
            return index, False, f"missing pn_unit_iid(s): {missing}"
        return index, True, ""
    except Exception as exc:
        return index, False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config_dir = Path(__file__).resolve().parents[2] / "foundry/models/rfd3/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=ss_adj_phase1b",
                "datasets=ss_adj_loop_focus_design",
                "paths/data=local_ss_adj",
            ],
        )

    val_cfg = cfg.datasets.val.loop.dataset
    OmegaConf.update(val_cfg, "cif_parser_args.save_to_cache", False)

    import hydra

    global _DATASET
    _DATASET = hydra.utils.instantiate(val_cfg)

    context = mp.get_context("fork")
    with context.Pool(processes=args.workers) as pool:
        results = list(
            pool.imap_unordered(_check_index, range(len(_DATASET)), chunksize=1)
        )

    results.sort()
    valid_indices = [index for index, valid, _ in results if valid]
    invalid = [(index, reason) for index, valid, reason in results if not valid]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _DATASET.data.iloc[valid_indices].to_csv(output, index=False)

    print(f"input_rows={len(_DATASET)}")
    print(f"valid_rows={len(valid_indices)}")
    print(f"excluded_rows={len(invalid)}")
    for index, reason in invalid:
        row = _DATASET.data.iloc[index]
        print(f"EXCLUDE index={index} example_id={row['example_id']} reason={reason}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
