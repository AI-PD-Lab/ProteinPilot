#!/usr/bin/env python
"""derive_ckpt.py - a checkpoint whose EMA shadow carries the RAW weights, so the unchanged inference stack samples them.

How the harness selects weights (read 2026-11-07): foundry BaseInferenceEngine.initialize() torch.loads the checkpoint and calls
trainer.load_checkpoint() on the EMA-wrapped model, which fills BOTH copies (state-dict keys `model.*` = raw, `shadow.*` = EMA);
EMA.forward() uses `shadow` whenever the module is not training (rfd3/trainer/rfd3.py:1151 comment), and nothing in
rfd3/engine.py, run_inference.py or eval_checkpoint.sbatch can switch that. So "--weights raw" = this script: copy every
`model.X` tensor over `shadow.X`, keep everything else (train_cfg incl. its omegaconf objects, optimizer state, ...) untouched,
and hand the harness the derived file. `--mode ema` only prints the key statistics (the original file is the EMA eval).

Run it with a python that can unpickle the checkpoint (the rfd3 env on the node: train_cfg holds omegaconf objects; the
esmfold2 env cannot, adapter_diag.py uses a stub pickler for its read-only diagnostics). E2 epoch-0596: 814 model.* / 814
shadow.* keys, 804 pairs differ.

usage: python -I derive_ckpt.py --in CKPT --out OUT.ckpt --mode raw|ema [--no-verify]
exit codes: 0 ok, 2 key-layout problem (no shadow.*, unpaired keys, mismatching shapes), 3 verification failed."""
import argparse
import json
import os
import sys


def key_stats(sd):
    model = [k for k in sd if k.startswith("model.")]
    shadow = [k for k in sd if k.startswith("shadow.")]
    other = [k for k in sd if not (k.startswith("model.") or k.startswith("shadow."))]
    unpaired = [k for k in model if "shadow." + k[6:] not in sd] + [k for k in shadow if "model." + k[7:] not in sd]
    return model, shadow, other, unpaired


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None, help="derived checkpoint (required for --mode raw)")
    ap.add_argument("--mode", choices=("raw", "ema"), required=True)
    ap.add_argument("--no-verify", action="store_true", help="skip re-loading the written file")
    a = ap.parse_args(argv)
    ck = torch.load(a.inp, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model, shadow, other, unpaired = key_stats(sd)
    differ = sum(1 for k in model if "shadow." + k[6:] in sd and not torch.equal(sd[k], sd["shadow." + k[6:]]))
    shape_bad = [k for k in model if "shadow." + k[6:] in sd and tuple(sd[k].shape) != tuple(sd["shadow." + k[6:]].shape)]
    stats = dict(ckpt=a.inp, mode=a.mode, n_model=len(model), n_shadow=len(shadow), n_other=len(other), n_unpaired=len(unpaired),
                 n_pairs_differ=differ, n_shape_mismatch=len(shape_bad), top_keys=[k for k in ck] if isinstance(ck, dict) else [])
    print(json.dumps(stats))
    if not shadow:
        print("FATAL: no shadow.* keys - not an EMA checkpoint, raw == what inference uses already", file=sys.stderr)
        return 2
    if unpaired or shape_bad:
        print(f"FATAL: unpaired keys {unpaired[:5]} / shape mismatches {shape_bad[:5]}", file=sys.stderr)
        return 2
    if a.mode == "ema":
        return 0
    if not a.out:
        print("FATAL: --mode raw needs --out", file=sys.stderr)
        return 2
    for k in model:
        sd["shadow." + k[6:]] = sd[k].clone()
    tmp = a.out + ".part"
    torch.save(ck, tmp)
    os.replace(tmp, a.out)
    print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes): shadow.* := model.* for {len(model)} tensors")
    if a.no_verify:
        return 0
    ck2 = torch.load(a.out, map_location="cpu", weights_only=False, mmap=True)
    sd2 = ck2["model"] if isinstance(ck2, dict) and "model" in ck2 else ck2
    bad = [k for k in model if not torch.equal(sd2[k], sd2["shadow." + k[6:]]) or not torch.equal(sd2[k], sd[k])]
    extra = set(ck.keys()) ^ set(ck2.keys()) if isinstance(ck, dict) else set()
    if bad or extra:
        print(f"FATAL: verification failed: {len(bad)} tensors differ, top-level key diff {extra}", file=sys.stderr)
        return 3
    print(f"verified: {len(model)} shadow/model pairs identical in the derived file, top-level keys preserved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
