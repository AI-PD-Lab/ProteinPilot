# HARMONIA

Adjacency-conditioned protein design for RFdiffusion3 and RFdiffusion4.

HARMONIA adds two conditioning inputs to the diffusion models: a per-residue **secondary-structure** string and a **block adjacency**
map saying which secondary-structure elements touch. You describe a fold by its topology, and the model builds the structure.
Two models are released:

| Model | Base | Checkpoint | Code |
|---|---|---|---|
| **Harmonia-RFD3** | [RFdiffusion3](https://www.biorxiv.org/content/10.1101/2025.09.18.676967) | `Harmonia-RFD3.ckpt` (epoch 698, 2.7 GB) | [`rfd3/`](rfd3) |
| **Harmonia-RFD4** | [RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina) | `Harmonia-RFD4/` (R3, 8k steps, 1.7 GB) | [`rfd4/`](rfd4) |

Both are fine-tunes of the released base models. With the conditioning left unspecified, the added inputs contribute exactly zero, so
the models still run as the original base model.

**[ProteinPilot](https://ai-pd-lab.github.io/ProteinPilot/)** is the companion web app: draw a topology in the browser and download the
conditioning inputs and run script for Harmonia-RFD3, Harmonia-RFD4 or RFdiffusion
([source](https://github.com/AI-PD-Lab/ProteinPilot), [Hugging Face Space](https://huggingface.co/spaces/AI-PD/ProteinPilot)).

## Checkpoints

Hosted in the Hugging Face model repo [`AI-PD/Harmonia`](https://huggingface.co/AI-PD/Harmonia):

```bash
pip install huggingface_hub
python checkpoints/download.py --model rfd3     # checkpoints/Harmonia-RFD3.ckpt
python checkpoints/download.py --model rfd4     # checkpoints/Harmonia-RFD4/{config.json,model.safetensors}
```

Log in first (`huggingface-cli login` or `HF_TOKEN`) while the repo is private. See [`checkpoints/README.md`](checkpoints/README.md)
for what each file is.

## Inputs

Both models read the same two tensors, so one drawing can be compared across models:

- `ss` `[L]`: `0` helix, `1` strand, `2` loop, `3` masked.
- `adj` `[L, L]`: block adjacency, `1` contact (any C-beta pair within 6 A), `0` explicit non-contact, `2` masked (unspecified).
  Fill unspecified pairs with 2, not 0: a dense-zero matrix asserts that those pairs must not touch.

Harmonia-RFD4 additionally accepts the states `3` HOT (contact with a large interface) and `4` COLD (clearly apart).

## Quick start

Harmonia-RFD3: [`rfd3/README.md`](rfd3/README.md). Harmonia-RFD4: [`rfd4/README.md`](rfd4/README.md). The easiest way to produce inputs is
ProteinPilot.

## Results in brief

Natural 5-target panel (5 targets x full / sparse / dense-zero adjacency), guidance weight 1.5, 240 designs per model.

| | stock RFD4 (SS only) | Harmonia-RFD3 lineage (epoch 694) | **Harmonia-RFD4** |
|---|---|---|---|
| fold recovery (SS + adjacency) | 0.81 | 0.86 | **0.93** |
| TM >= 0.5 to the native | 0.13 | 0.25 | **0.49** |
| split rate | 10.8 % | 1.2 % | **0.4 %** |
| designability (8 x ProteinMPNN, ESMFold2) | 0.95 | 0.59-0.62 | 0.93 |

On a benchmark of 49 conditions exported from ProteinPilot (fold conditioning, motif scaffolding, binder interfaces, loop, ligand and
nucleic-acid contacts), RFD4 formed drawn loop contacts far more often than RFD3 (about 79 % vs 28 % mean recovery), and both needed less
sampling time per design than RFdiffusion. Known limitation of the released Harmonia-RFD4: unconditional generation
splits 54 % of designs (stock RFD4: 69 %); later development checkpoints reduce this (see `rfd4/README.md`).

## Repository layout

```
rfd3/            Harmonia-RFD3: trimmed RFdiffusion3 / foundry inference code, adapters, configs, launcher, docs
rfd4/            Harmonia-RFD4: harmonia_ext, the inference add-on for an unmodified RFD4-Proteina install
checkpoints/     download.py and the Hugging Face model card
```

This repository contains only what is needed to run the models. Training code, data pipelines, evaluation harnesses and the later research lines
are on the branches `phase10`, `harmonia-v2`, `rfd4-harmonia` and `rfd4-harmonia-full`.

## License

Original HARMONIA code and the two fine-tuned checkpoints: [PolyForm Noncommercial 1.0.0](LICENSE). Code derived from RFdiffusion3 / foundry
stays under BSD 3-Clause ([`LICENSE-BSD-3-Clause`](LICENSE-BSD-3-Clause)); RFD4-Proteina is installed separately (BSD 3-Clause). Which
file is under which licence: [`NOTICE.md`](NOTICE.md).

## Citation

```bibtex
@article{butcher2025_rfdiffusion3,
  author = {Butcher, Jasper and Krishna, Rohith and Mitra, Raktim and Brent, Rafael Isaac and Li, Yanjing and Corley, Nathaniel and Kim, Paul T and Funk, Jonathan and Mathis, Simon Valentin and Salike, Saman and Muraishi, Aiko and Eisenach, Helen and Thompson, Tuscan Rock and Chen, Jie and Politanska, Yuliya and Sehgal, Enisha and Coventry, Brian and Zhang, Odin and Qiang, Bo and Didi, Kieran and Kazman, Maxwell and DiMaio, Frank and Baker, David},
  title = {De novo Design of All-atom Biomolecular Interactions with RFdiffusion3},
  year = {2025},
  doi = {10.1101/2025.09.18.676967},
  journal = {bioRxiv}
}
@article{corley2026_rfd4,
  author = {Corley, Nathaniel and others},
  title = {Biomolecular design through latent reasoning with RFD4-Proteina},
  year = {2026},
  doi = {10.64898/2026.11.12.766037},
  journal = {bioRxiv}
}
```
