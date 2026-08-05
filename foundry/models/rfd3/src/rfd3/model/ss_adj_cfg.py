"""Classifier-free guidance defaults for SS/ADJ conditioning features."""

from __future__ import annotations

import torch


def apply_cfg_defaults(
    features: dict[str, torch.Tensor], cfg_features: list[str]
) -> dict[str, torch.Tensor]:
    """Apply unconditional sentinels after the generic CFG feature stripping."""
    if "ss_ratio" in cfg_features and "ss_ratio" in features:
        features["ss_ratio"] = torch.full_like(features["ss_ratio"], -1.0)
    if "ss_labels" in cfg_features and "ss_labels" in features:
        features["ss_labels"] = torch.full_like(features["ss_labels"], 3)
    if "adj_matrix" in cfg_features and "adj_matrix" in features:
        features["adj_matrix"] = torch.full_like(features["adj_matrix"], 2)
    return features
