"""Stand-in for rfd3's run_inference.py used by the sbatch dry-run test (CPU, no model): same hydra-style arguments
(key=value), same output naming (<key>_<batch>_model_<m>.cif.gz + .json), synthetic helical backbones of the requested
length. FAKE_FAIL=<substring of the input json name> makes the matching shard exit 1."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np  # noqa: E402

import conftest as cf  # noqa: E402


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    fail = os.environ.get("FAKE_FAIL")
    if fail and fail in os.path.basename(args["inputs"]):
        print("fake inference: forced failure", file=sys.stderr)
        sys.exit(1)
    specs = json.load(open(args["inputs"]))
    out = args["out_dir"]
    os.makedirs(out, exist_ok=True)
    nb, bs = int(args["n_batches"]), int(args["diffusion_batch_size"])
    rng = np.random.RandomState(int(args.get("seed", 0)))
    print("fake inference", args.get("ckpt_path"), "cfg", args.get("inference_sampler.use_classifier_free_guidance"), flush=True)
    for key, spec in specs.items():
        L = int(str(spec["length"]).split("-")[0])
        for b in range(nb):
            for m in range(bs):
                ca = cf.helix_bundle_ca(n_helix=1, helix_len=L) + rng.randn(L, 3) * 0.3
                base = os.path.join(out, f"{key}_{b}_model_{m}")
                open(base + ".cif.gz", "wb").write(cf.to_cif_gz_bytes(cf.atoms_from_ca(ca)))
                json.dump({"metrics": {"num_residues": L}, "specification": spec, "seed": int(args.get("seed", 0))}, open(base + ".json", "w"))


if __name__ == "__main__":
    main()
