# HARMONIA

Secondary-structure and block-adjacency ("SS/ADJ") fold conditioning for
[RFdiffusion3](https://www.biorxiv.org/content/10.1101/2025.09.18.676967) —
plus the related global `ss_ratio` conditioning knob — with a trained
checkpoint at epoch 670.

This repo is a trimmed-down fork scoped to just that feature: the RFD3/
AtomWorks source needed to run it, the adapters and transforms that add it,
the scripts used to build training data and launch training/inference, and
docs explaining how it works. It does **not** include RFD3's other model
variants (rfd3na, rf3), training data, logs, or the unrelated RING project
this work was originally developed alongside.

The checkpoint (2.6 GB) is hosted on Hugging Face, not in this repo — see
[Checkpoint](#checkpoint).

## What this adds to RFD3

Two independent conditioning mechanisms, both trained as zero-init adapters
on top of a pretrained RFD3 checkpoint so they're safe to warm-start from:

- **SS/ADJ (per-residue)** — condition generation on a target secondary
  structure string (helix/strand/loop) and/or a block adjacency matrix
  (which secondary-structure elements pack against each other), the same
  tensor conventions RFdiffusion1 used. See
  [`docs/ss_adj_inference.md`](docs/ss_adj_inference.md) for the exact input
  format, and [`docs/ss_adj_training_plan.md`](docs/ss_adj_training_plan.md)
  for how the checkpoint was trained (data, phases, submission scripts).

- **ss_ratio (global)** — a single scalar per chain (`E / (H + E)`, i.e. the
  helix/strand balance) that biases generation toward an overall fold
  composition without specifying per-residue topology. See
  [`docs/ss_ratio_conditioning.md`](docs/ss_ratio_conditioning.md) for the
  architecture (Fourier-feature encoder, classifier-free-guidance dropout).

Both are implemented as additive, zero-initialized adapters injected into
`TokenInitializer` (`foundry/models/rfd3/src/rfd3/model/layers/`), so a
checkpoint trained with them enabled still behaves like the base model at
step zero and only diverges as those adapter weights are trained.

## Repo layout

```
foundry/                  trimmed RFD3 + core foundry framework (needed to run it)
  src/                     core training/inference framework
  models/rfd3/             the RFD3 model itself — config, source, tests
patches/atomworks/         small patch on top of public atomworks[ml] (tar-backed data loading)
scripts/
  data/                    build/precompute/validate SS/ADJ training data
  training/                SLURM launchers for the SS/ADJ and ss_ratio training phases
  inference/                generate conditioning inputs, run inference, analyze/plot results
docs/                      SS/ADJ + ss_ratio design docs
checkpoints/               download.py — pulls epoch-0670 from Hugging Face
```

## Setup

Requires Python 3.12.

```bash
pip install "rc-foundry[rfd3]"   # or: pip install -e ./foundry[rfd3]
pip install "atomworks[ml]>=2.1.1"
```

Apply the small AtomWorks patch (only needed if you're using tar-backed
training data mirrors — see [`patches/atomworks/README.md`](patches/atomworks/README.md)):

```bash
ATOMWORKS_SRC=$(python -c "import atomworks, pathlib; print(pathlib.Path(atomworks.__file__).parent)")
cp -r patches/atomworks/src/atomworks/* "$ATOMWORKS_SRC/"
```

## Checkpoint

The epoch-670 checkpoint is published on Hugging Face at
[`Jetzu77/Harmonia`](https://huggingface.co/Jetzu77/Harmonia). Download it with:

```bash
pip install huggingface_hub
python checkpoints/download.py
```

This fetches `epoch-0670_localattn.ckpt` into `checkpoints/`. If the HF repo
is private, run `huggingface-cli login` (or set `HF_TOKEN`) first.

### Keeping Hugging Face in sync

A `pre-push` hook mirrors the checkpoint + model card to
`Jetzu77/Harmonia` automatically whenever you `git push`. Enable it once per
clone:

```bash
git config core.hooksPath .githooks
```

It only re-uploads the (large) checkpoint when its size/mtime changed since
the last sync — otherwise it just refreshes the model card, which is cheap.
It never blocks the GitHub push: if `huggingface_hub` isn't installed, no
one's logged in, or the upload fails, it prints a warning and lets the push
through anyway. The checkpoint isn't in this repo, so point
`HARMONIA_CKPT_PATH` at it if it's not at the default
(`/work/gerstung/RFD3-adj-ckpt/epoch-0670_localattn.ckpt`).

## Running inference

Enable the adapters you need in the model config
(`foundry/models/rfd3/configs/model/components/rfd3_net.yaml`):

```yaml
token_initializer:
  ss_adapter:
    enabled: true   # per-residue SS/ADJ conditioning
  adj_adapter:
    enabled: true
```

`ss_conditioning` (the ss_ratio Fourier encoder) is always present in the
config — it's controlled at the input level instead (see below), not by an
`enabled` flag.

Then run RFD3 inference the normal way, pointing `ckpt_path` at the
downloaded checkpoint and supplying SS/ADJ tensors or an `ss_ratio` value in
your input JSON:

```bash
rfd3 design \
    ckpt_path=checkpoints/epoch-0670_localattn.ckpt \
    out_dir=logs/inference_outs/demo/0 \
    inputs=my_inputs.json
```

Minimal SS/ADJ input example (full format in
[`docs/ss_adj_inference.md`](docs/ss_adj_inference.md)):

```json
{
  "example": {
    "length": "150",
    "ss_file": "/path/target_ss.pt",
    "adj_file": "/path/target_adj.pt"
  }
}
```

`ss_file`/`adj_file` are `.pt`/`.npy` tensors: SS is `[L]` with
`0=helix, 1=strand, 2=loop`; ADJ is a symmetric `[L,L]` binary contact
matrix with zero diagonal. `select_ss` is also available for assigning
labels inline by residue selection. Classifier-free guidance can be enabled
with `inference_engine.inference_sampler.use_classifier_free_guidance=true`.

For `ss_ratio`-only conditioning (no per-residue tensors), set a target
value in `[0, 1]` per chain — see
[`docs/ss_ratio_conditioning.md`](docs/ss_ratio_conditioning.md) for the
exact feature key and CFG behavior.

`scripts/inference/` has the concrete scripts used to generate benchmark
inputs and run/analyze inference for this project (`run_ss_adj_inference.py`,
`generate_ss_adj_conditioning_inputs*.py`, `analyze_ss_adj_conditioning.py`,
`plot_ss_adj_results.py`).

## Training / fine-tuning

Checkpoint 670 was produced with the phased fine-tuning plan in
[`docs/ss_adj_training_plan.md`](docs/ss_adj_training_plan.md): adapters
first, then a standard/loop-conditioning data mix, then the full model at
low learning rate. `scripts/data/` builds the training data (DSSP + block
adjacency precomputation, loop-focused datasets); `scripts/training/` has
the SLURM launchers for each phase.

These launchers are reference scripts tied to the original cluster setup
(SLURM partition/reservation names, a specific Python environment path, and
a `Data_Mirrors/` training-data root that isn't part of this repo) — treat
them as a starting point rather than turnkey, and adjust `ROOT`, `PYTHON`,
`PARTITION`, and the data-mirror paths for your own environment.

## Citation

This builds on RFdiffusion3:

```bibtex
@article{butcher2025_rfdiffusion3,
  author = {Butcher, Jasper and Krishna, Rohith and Mitra, Raktim and Brent, Rafael Isaac and Li, Yanjing and Corley, Nathaniel and Kim, Paul T and Funk, Jonathan and Mathis, Simon Valentin and Salike, Saman and Muraishi, Aiko and Eisenach, Helen and Thompson, Tuscan Rock and Chen, Jie and Politanska, Yuliya and Sehgal, Enisha and Coventry, Brian and Zhang, Odin and Qiang, Bo and Didi, Kieran and Kazman, Maxwell and DiMaio, Frank and Baker, David},
  title = {De novo Design of All-atom Biomolecular Interactions with RFdiffusion3},
  year = {2025},
  doi = {10.1101/2025.09.18.676967},
  journal = {bioRxiv}
}
```

## License

BSD 3-Clause — see [`LICENSE`](LICENSE). RFD3/foundry source is
Copyright (c) 2025 Institute for Protein Design, University of Washington;
the SS/ADJ and ss_ratio conditioning work in this repo is an original
contribution on top of it.
