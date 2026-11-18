"""Expanded base bundle: released RFD4-Proteina weights + the zero-init ``nn.adj_embed.weight`` (strict loading in train.py, PEFT-LoRA safe).

  python -m harmonia_ext.make_bundle --src /w/rfd4_weights --dst /w/harmonia_rfd4/base

Checks (assert): the only missing key is nn.adj_embed.weight, it is exactly zero, released tensors bit-equal after the round trip,
strict reload through Proteina.from_pretrained with the registered class. Prints ``BUNDLE {json}`` incl. the SS-embedding parameter names.
"""
from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/w/rfd4_weights")
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    from omegaconf import OmegaConf

    import rfproteina.proteina as P
    from rfproteina.utils.hub import load_pretrained_bundle

    from harmonia_ext.adj_model import register

    register()
    payload, state = load_pretrained_bundle(a.src)
    cfg = payload["config"]
    cfg.update({"run_name_": "pretrained", "autoencoder_ckpt_path": None, "precision": "32", "metrics": None,
                "datasets": {"train": {}, "val": {}}, "log": {"step_log_frequency": 100}})
    model = P.Proteina(OmegaConf.create(cfg), load_training_metrics=False)
    res = model.load_state_dict(state, strict=False, assign=True)
    assert set(res.missing_keys) == {"nn.adj_embed.weight"}, res.missing_keys
    assert not res.unexpected_keys, res.unexpected_keys
    assert float(model.state_dict()["nn.adj_embed.weight"].abs().max()) == 0.0
    ss = [n for n, _ in model.named_parameters() if "ss_embedding" in n]
    os.makedirs(a.dst, exist_ok=True)
    model.save_pretrained(a.dst)
    del model
    _, new = load_pretrained_bundle(a.dst)
    assert set(new) - set(state) == {"nn.adj_embed.weight"} and all(torch.equal(new[k], state[k]) for k in state)
    m2 = P.Proteina.from_pretrained(a.dst)
    print("BUNDLE " + json.dumps({"dst": a.dst, "nn_class": type(m2.nn).__name__, "adj_shape": list(new["nn.adj_embed.weight"].shape),
                                  "ss_embedding_params": ss, "checks": "ok"}), flush=True)


if __name__ == "__main__":
    main()
