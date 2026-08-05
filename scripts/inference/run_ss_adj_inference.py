#!/usr/bin/env python3
"""RFdiffusion3 inference wrapper for SS/ADJ benchmark runs.

The checkpoint can contain validation transforms copied from training configs.  For
SS/ADJ inputs that already carry inference conditioning annotations (ss_ratio,
is_non_loopy, ss_file, adj_file), those transforms must be forced into inference
mode before the pipeline is instantiated.
"""

from __future__ import annotations

import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

load_dotenv(override=True)

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "foundry",
    "models",
    "rfd3",
    "configs",
)


@hydra.main(config_path=_CONFIG_PATH, config_name="inference", version_base="1.3")
def run_inference(cfg: DictConfig) -> None:
    run_params_set = {"inputs", "n_batches", "out_dir"}
    run_params = {k: v for k, v in cfg.items() if k in run_params_set}

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    init_cfg_dict = {
        k: v for k, v in cfg_dict.items() if k not in run_params_set | {"_target_"}
    }

    engine = RFD3InferenceEngine(**RFD3InferenceConfig(**init_cfg_dict))
    engine.transform_overrides.update(
        {
            "is_inference": True,
            "return_atom_array": True,
        }
    )
    engine.run(**run_params)


if __name__ == "__main__":
    run_inference()
