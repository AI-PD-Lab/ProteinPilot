# Harmonia-RFD3

Secondary-structure and block-adjacency ("SS/ADJ") fold conditioning for
[RFdiffusion3](https://www.biorxiv.org/content/10.1101/2025.09.18.676967), plus the global `ss_ratio` knob. The released checkpoint,
`Harmonia-RFD3.ckpt`, is epoch 698 of the final fine-tuning stage ("phase 10", epochs 695-703, a rollback to epoch 694 with a minimal
change set; see [`docs/ss_adj_training_plan.md`](docs/ss_adj_training_plan.md)).

This directory is a trimmed fork of RFD3 / foundry scoped to this feature: the source needed to run it, the adapters and transforms that
add it, the scripts that build training data and launch inference, and docs. RFD3's other model variants (rfd3na, rf3), training data and
logs are not included.

## What it adds

Both mechanisms are zero-initialised adapters in `TokenInitializer` (`foundry/models/rfd3/src/rfd3/model/layers/`), so a checkpoint still
behaves like the base model wherever the conditioning is unset.

- **SS/ADJ (per-residue).** Condition on a secondary-structure string (helix / strand / loop) and/or a block adjacency matrix. Adjacency
  covers protein, ligand and nucleic-acid blocks, loop contacts, and distinguishes *contact* (1) from *explicit non-contact* (0) from
  *masked* (2). Exact input format: [`docs/ss_adj_inference.md`](docs/ss_adj_inference.md).
- **ss_ratio (global).** One scalar per chain, `E / (H + E)`, biasing the overall fold composition without per-residue topology
  ([`docs/ss_ratio_conditioning.md`](docs/ss_ratio_conditioning.md)).

## Layout

```
foundry/models/rfd3/   RFD3 model, configs (experiment/ss_adj_phase10_rollback.yaml = the release stage), source, SS/ADJ tests
foundry/src/           core foundry framework
patches/atomworks/     small patch on top of public atomworks[ml] (tar-backed data loading, training only)
scripts/data/          build the SS/ADJ training data (DSSP + block adjacency, splits, PDB mirror index)
scripts/inference/     example inputs and the inference launcher
scripts/training/      launcher for the release stage
scripts/eval/          evaluation harness (scorer, designability, metrics) and kit/ with the natural-panel manifest
docs/                  input format, training plan, ss_ratio
```

## Setup

Python 3.12.

```bash
cd rfd3
pip install -e "./foundry[rfd3]"
pip install "atomworks[ml]>=2.1.1"
python ../checkpoints/download.py --model rfd3       # -> ../checkpoints/Harmonia-RFD3.ckpt
```

The AtomWorks patch is only needed for tar-backed training data ([`patches/atomworks/README.md`](patches/atomworks/README.md)).

## Running inference

Create the two tensors (or let [ProteinPilot](https://ai-pd-lab.github.io/ProteinPilot/) write them): `ss` is `[L]` with
`0` helix, `1` strand, `2` loop, `3` masked; `adj` is `[L, L]` with `1` contact, `0` non-contact, `2` masked. Then name them in an RFD3
input JSON:

```json
{
  "example": {
    "length": "150",
    "ss_file": "/path/target_ss.pt",
    "adj_file": "/path/target_adj.pt"
  }
}
```

and run:

```bash
CKPT=../checkpoints/Harmonia-RFD3.ckpt INPUT_JSON=my_inputs.json OUT_ROOT=out \
    bash scripts/inference/normal_ss_adj_inference/run_normal_ss_adj_inference.sh
```

which wraps `foundry/models/rfd3/src/rfd3/run_inference.py` (`ckpt_path`, `inputs`, `out_dir`, `n_batches`, `diffusion_batch_size`,
`inference_sampler.*`). The checkpoint carries its own model config with the trained `ss_adapter` / `adj_adapter`. Classifier-free guidance
over the conditioning is `inference_sampler.use_classifier_free_guidance=True` with `cfg_scale` (2.0 in the launcher).
Without the script, the same overrides work with `rfd3 design` / `run_inference.py` directly. If you run the script with no `INPUT_JSON`, it
generates a small example input set first.

**Fill unspecified pairs with 2 (masked), not 0.** A dense-zero matrix asserts that those pairs must not touch, which tears folds apart
on this lineage. Reserve 0 for non-contacts you actually mean.

## Training

Stage 10 is reproducible with `scripts/training/train_harmonia_rfd3.sh` given the epoch-694 checkpoint and your own PDB mirror; read the
header of that script and [`docs/ss_adj_training_plan.md`](docs/ss_adj_training_plan.md). Data paths are set in
`foundry/models/rfd3/configs/paths/data/local_ss_adj.yaml` (copy and edit). The launcher is a reference, not turnkey: it carries no
scheduler directives and the training data are not part of this repository. Earlier phases (1-9) and their launchers are on the `main`, `phase9-adjacency-spots` and `phase10` branches. Path configs for training are set in
`foundry/models/rfd3/configs/paths/data/local_ss_adj.yaml`.

## License

Original HARMONIA parts: PolyForm Noncommercial 1.0.0 (root [`LICENSE`](../LICENSE)); foundry-derived code: BSD 3-Clause ([`NOTICE.md`](../NOTICE.md)).
