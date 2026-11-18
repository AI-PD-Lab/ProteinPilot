#!/usr/bin/env python
"""Shard plan of one checkpoint evaluation (WP-EVAL step 2 helper; runs in any env with numpy).

Reads the spec pack of build_specs.py, extracts it, and writes one rendered input JSON per shard
(shard = one reference target x one CFG scale, holding all arms of that target; the unconditional panel is
one shard per CFG list given to --uncond-cfgs, default 1.0 only: with no ss/adj/ss_ratio input the conditional
and the CFG reference pass are identical, so CFG would only double the cost). Shards are balanced over GPUs
(longest-processing-time first) so the sbatch can run one process per GPU.

  python make_plan.py --specs-tar specs_v2.tar.gz --root /tmp/specs --out /tmp/plan --gpus 4 --n 32

Writes <out>/plan.tsv (tab separated, one shard per line; header row):
  shard  gpu  cfg  seed  n_batches  batch_size  input_json  n_keys  est_cost
and <out>/plan.json (same + the keys of every shard). Seeds are deterministic: base_seed + crc32(shard) % 10^6,
so a re-run of the same plan reproduces the same designs (bit-exactness of GPU sampling is not guaranteed).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_specs as bs  # noqa: E402
import eval_common as ec  # noqa: E402


def shard_seed(base_seed: int, shard: str) -> int:
    return int(base_seed) + zlib.crc32(shard.encode()) % 1_000_000


def make_plan(inputs: dict, manifest: dict, cfgs, uncond_cfgs, n: int, batch: int, gpus: int, base_seed: int,
              out_dir: str) -> list[dict]:
    keys_by_target: dict[str, list[str]] = {}
    length_by_target: dict[str, int] = {}
    for key, rec in {**manifest["arms"], **manifest.get("unconditional", {})}.items():
        keys_by_target.setdefault(rec["target"], []).append(key)
        length_by_target[rec["target"]] = int(rec["length"])
    shards = []
    for tname, keys in keys_by_target.items():
        is_uncond = tname.startswith("uncond_")
        for cfg in (uncond_cfgs if is_uncond else cfgs):
            name = f"cfg{float(cfg):.1f}__{tname}"
            L = length_by_target[tname]
            # cost ~ designs x (L/100)^1.6, doubled when CFG runs a second pass
            cost = len(keys) * n * (L / 100.0) ** 1.6 * (2.0 if float(cfg) > 1.0 else 1.0)
            shards.append({"shard": name, "cfg": float(cfg), "target": tname, "keys": sorted(keys),
                           "n_batches": int(math.ceil(n / batch)), "batch_size": batch,
                           "seed": shard_seed(base_seed, name), "est_cost": cost})
    shards.sort(key=lambda s: -s["est_cost"])
    load = [0.0] * gpus
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
    for s in shards:
        g = min(range(gpus), key=lambda i: load[i])
        s["gpu"] = g
        load[g] += s["est_cost"]
        path = os.path.join(out_dir, "shards", s["shard"] + ".json")
        ec.write_json(path, {k: inputs[k] for k in s["keys"]}, indent=1)
        s["input_json"] = path
    shards.sort(key=lambda s: (s["gpu"], -s["est_cost"]))
    return shards


def write_plan(shards: list[dict], out_dir: str):
    cols = ["shard", "gpu", "cfg", "seed", "n_batches", "batch_size", "input_json", "n_keys", "est_cost"]
    with open(os.path.join(out_dir, "plan.tsv"), "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for s in shards:
            row = dict(s, n_keys=len(s["keys"]), est_cost=f"{s['est_cost']:.1f}")
            fh.write("\t".join(str(row[c]) for c in cols) + "\n")
    ec.write_json(os.path.join(out_dir, "plan.json"), shards, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specs-tar", default=ec.SPECS_TAR_DEFAULT)
    ap.add_argument("--root", required=True, help="extraction directory for the spec pack (node-local /tmp in the sbatch)")
    ap.add_argument("--out", required=True, help="plan directory")
    ap.add_argument("--cfgs", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--uncond-cfgs", type=float, nargs="+", default=[1.0])
    ap.add_argument("--n", type=int, default=16, help="designs per key and CFG (n_batches x batch_size >= n); 16 = screening, 32 = finalists")
    ap.add_argument("--batch", type=int, default=4, help="diffusion_batch_size")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261014, help="base seed (recorded in meta.json)")
    a = ap.parse_args(argv)
    inputs, manifest = bs.unpack_and_render(a.specs_tar, a.root)
    shards = make_plan(inputs, manifest, a.cfgs, a.uncond_cfgs, a.n, a.batch, a.gpus, a.seed, a.out)
    write_plan(shards, a.out)
    per_gpu = {}
    for s in shards:
        per_gpu[s["gpu"]] = per_gpu.get(s["gpu"], 0) + s["est_cost"]
    print(f"{len(shards)} shards on {a.gpus} GPUs; est. cost per GPU: " + ", ".join(f"{g}:{c:.0f}" for g, c in sorted(per_gpu.items())))
    print(f"wrote {os.path.join(a.out, 'plan.tsv')}")


if __name__ == "__main__":
    main()
