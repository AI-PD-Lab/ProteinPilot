---
license: bsd-3-clause
tags:
  - protein-design
  - diffusion
  - rfdiffusion
  - structural-biology
---

# Harmonia — RFD3 SS/ADJ + ss_ratio fold conditioning (epoch 670)

A [RFdiffusion3](https://www.biorxiv.org/content/10.1101/2025.09.18.676967)
checkpoint fine-tuned with two additional fold-conditioning mechanisms on top
of the pretrained base model:

- **SS/ADJ (per-residue)** — condition on a target secondary-structure string
  (helix/strand/loop) and/or a block adjacency matrix (which secondary
  structure elements pack against each other), using the same tensor
  conventions as RFdiffusion1.
- **ss_ratio (global)** — a single scalar per chain (helix/strand balance)
  that biases generation toward an overall fold composition.

Both are zero-initialized adapters injected into RFD3's `TokenInitializer`,
trained so the checkpoint still runs like the base model when the
conditioning is left unset, and only diverges when SS/ADJ tensors or an
`ss_ratio` value are supplied.

Code, config, training scripts, and usage docs:
[github.com/Jetzu/HARMONIA](https://github.com/Jetzu/HARMONIA) (private —
request access if you've been pointed here without it).

## Training

Fine-tuned from a pretrained RFD3 checkpoint through phased adapter
training — adapters only, then a standard/loop-conditioning data mix, then
the full model at low learning rate — see
[`docs/ss_adj_training_plan.md`](https://github.com/Jetzu/HARMONIA/blob/main/docs/ss_adj_training_plan.md)
and
[`docs/ss_ratio_conditioning.md`](https://github.com/Jetzu/HARMONIA/blob/main/docs/ss_ratio_conditioning.md)
in the repo for the full data/phase breakdown. This checkpoint is from
epoch 670.

## Usage

```bash
pip install huggingface_hub
python checkpoints/download.py   # from the HARMONIA repo
```

or directly:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="Jetzu77/Harmonia", filename="epoch-0670_localattn.ckpt")
```

Then follow "Running inference" in the
[HARMONIA README](https://github.com/Jetzu/HARMONIA#running-inference) to
enable the adapters and supply SS/ADJ or ss_ratio conditioning.

## License

BSD 3-Clause, inherited from RFdiffusion3 (Institute for Protein Design,
University of Washington). See the
[HARMONIA LICENSE](https://github.com/Jetzu/HARMONIA/blob/main/LICENSE).

## Citation

```bibtex
@article{butcher2025_rfdiffusion3,
  author = {Butcher, Jasper and Krishna, Rohith and Mitra, Raktim and Brent, Rafael Isaac and Li, Yanjing and Corley, Nathaniel and Kim, Paul T and Funk, Jonathan and Mathis, Simon Valentin and Salike, Saman and Muraishi, Aiko and Eisenach, Helen and Thompson, Tuscan Rock and Chen, Jie and Politanska, Yuliya and Sehgal, Enisha and Coventry, Brian and Zhang, Odin and Qiang, Bo and Didi, Kieran and Kazman, Maxwell and DiMaio, Frank and Baker, David},
  title = {De novo Design of All-atom Biomolecular Interactions with RFdiffusion3},
  year = {2025},
  doi = {10.1101/2025.09.18.676967},
  journal = {bioRxiv}
}
```
