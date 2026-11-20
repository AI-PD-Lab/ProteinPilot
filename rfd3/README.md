# Harmonia-RFD3

Secondary-structure and block-adjacency ("SS/ADJ") fold conditioning for
[RFdiffusion3](https://www.biorxiv.org/content/10.1101/2025.09.18.676967), plus the global `ss_ratio` knob. The released checkpoint,
`Harmonia-RFD3.ckpt`, is epoch 698 of the final fine-tuning stage ("phase 10", epochs 695-703, a rollback to epoch 694 with a minimal
change set).

This directory is a trimmed fork of RFD3 / foundry that contains only what is needed to run the model: the inference code, the adapters and
transforms that add the conditioning, the configs and an inference launcher. Training code and data are not included.

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
foundry/models/rfd3/   RFD3 model, inference configs, source
foundry/src/           core foundry framework
scripts/inference/     example inputs and the inference launcher
docs/                  input format, ss_ratio
```

## Setup

Python 3.12.

```bash
cd rfd3
pip install -e "./foundry[rfd3]"
pip install "atomworks[ml]>=2.1.1"
python ../checkpoints/download.py --model rfd3
```

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

## License

Original HARMONIA parts: PolyForm Noncommercial 1.0.0 (root [`LICENSE`](../LICENSE)); foundry-derived code: BSD 3-Clause ([`NOTICE.md`](../NOTICE.md)).
