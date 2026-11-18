# Harmonia-RFD4

SS + block-adjacency fold conditioning for [RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina), as a full fine-tune
(release = **R3**, 8,000 steps, EMA weights; Hugging Face `Harmonia-RFD4/`: `config.json` + `model.safetensors`, RFD4 bundle format).

`harmonia_ext/` is an add-on package. It does not modify RFD4-Proteina: put it on `PYTHONPATH` next to an unmodified install
(tested with upstream commit `fe95487`).

## What the model takes

| input | tensor | channel |
|---|---|---|
| length | - | as in RFD4 |
| secondary structure | `[L]` int, codes 0 helix / 1 strand / 2 loop / 3 masked | RFD4's native `C_SEC` channel (remapped to 1 / 2 / 3 / 0) |
| block adjacency | `[L, L]` int, states 0 non-contact / 1 contact / 2 masked / 3 HOT / 4 COLD / 5 pad | `adj_embed` = `nn.Embedding(6, pair_dim)`, zero-init, added to the pair input |

Masked and pad states are multiplied by 0, so an all-masked map reproduces the released RFD4 bit for bit. Contact = any C-beta pair
within 6 A. **Guidance:** `pred = pred_masked + w (pred_cond - pred_masked)` over SS and adjacency jointly (the masked pass sees neither);
release operating point **w = 1.5**.

## Use

Install RFD4-Proteina (container or `setup.sh`, see its README), download the weights
(`python ../checkpoints/download.py --model rfd4`), and put one `<name>.cif` (annotated CIF carrying `C_SEC`) plus `<name>.adj.npy`
(`[L, L]`, optionally `<name>.ss.npy`) in an input directory. [ProteinPilot](https://ai-pd-lab.github.io/ProteinPilot/) writes exactly this
folder together with a run script. Then:

```bash
PYTHONPATH=/path/to/HARMONIA/rfd4:/path/to/RFD4-Proteina/src \
python -m harmonia_ext.rfd4_generate \
    inputs=<input dir> out_dir=<out dir> hf_ckpt_path=checkpoints/Harmonia-RFD4 hf_ckpt_revision=null \
    harmonia_w=1.5 num_replicates=100 batch_size=4 recycle_strategy=thinking_medium generation.args.nsteps=60 \
    generation.model.local_latents.simulation_step_params.tsr_k=1.5 '~metrics'
```

`rfd4_generate` is stock `rfd4-generate` plus three patches: it registers the trunk class with `adj_embed` (`harmonia_ext.adj_model.register()`,
without which the checkpoint does not load), reads `<stem>.adj.npy` into the batch of the matching input (checked against the token
count), and installs the SS+ADJ guidance (`harmonia_ext.generate.install_guidance`). `harmonia_w=` is removed before Hydra parses the
remaining overrides; any other `rfd4-generate` override works. `tsr_k=1.5` for the latent track is the sampling setting the release numbers
were measured at (stock: 4). It needs a CUDA GPU.

Loading from Python:

```python
from harmonia_ext.adj_model import register
register()                               # before building the model: adds adj_embed to the trunk
from rfproteina.proteina import Proteina
model = Proteina.from_pretrained("checkpoints/Harmonia-RFD4")
```

## Release numbers (R3, w = 1.5; 5 natural targets x full / sparse8 / dense0_true, n = 240)

| | stock RFD4 (SS only) | RFD3 HARMONIA (epoch 694) | **Harmonia-RFD4** |
|---|---|---|---|
| fold recovery (SS + adjacency) | 0.81 | 0.86 | **0.93** |
| TM >= 0.5 to native | 0.13 | 0.25 | **0.49** |
| split rate (panel) | 10.8 % | 1.2 % | **0.4 %** |
| designability (8 x ProteinMPNN, ESMFold2) | 0.95 | 0.59-0.62 | 0.93 |
| requested cross-chain contacts realised (7 held-out dimers) | 0.21 | - | 0.61 |

HOT / COLD shift the contact count at the marked pairs by +0.80 / +0.61. **Limitations:** unconditional generation splits 54 % of designs
(stock RFD4 69 %, RFD3 HARMONIA 36 %), and on a dense-zero Rossmann stress spec 12.5 / 25 / 7 % (full / SS-only / adjacency-only) of designs
split. Later development checkpoints (R4: unconditional split 8 %, R5c: signed parallel / antiparallel contacts) and their code are on the
`rfd4-harmonia` and `rfd4-harmonia-full` branches of the development history.

## Files

Inference (what ProteinPilot exports run):

- `harmonia_ext/adj_model.py`: `HarmoniaLocalLatentsTransformer` and `register()`.
- `harmonia_ext/adj_labels.py`: state definitions and the label derivation (block detection, contact rule, HOT / COLD).
- `harmonia_ext/generate.py`: `install_guidance` (joint SS+ADJ guidance, optional autoguidance against a reference trunk) and the
  panel / spec generation CLI (`python -m harmonia_ext.generate --bundle <dir> --manifest <manifest.json> --cfgs 1.5 ...`).
- `harmonia_ext/rfd4_generate.py`: the `rfd4-generate` wrapper.

Training and evaluation (reproduce R1-R3):

- `harmonia_ext/adj_data.py`, `callbacks.py`, `train_entry.py`: training rows (PDB through RFD4's own pipeline plus SS and ADJ labels,
  curriculum, compactness gate, dense-zero dialect), step logging, and the RFD4 trainer entry with the new modules unfrozen.
- `harmonia_ext/make_bundle.py`: expands the released RFD4 weights with a zero `adj_embed` (the training starting point).
- `harmonia_ext/configs/experiment/harmonia_r{1,1_full,2,3}.yaml`: the recipes (each builds on the previous); **release = `harmonia_r3`**.
  `python -m harmonia_ext.make_bundle --src <RFD4 weights> --dst <base bundle>`, then
  `python -m harmonia_ext.train_entry experiment=harmonia_r3 hf_ckpt_path=<base bundle> out_dir=<dir>` (add the usual RFD4 trainer overrides
  for GPUs, batch size and workers; the runs used 2 x 64 rows per step, 8,000 steps).
- `harmonia_ext/hotcold_score.py`: HOT/COLD effect scoring. `../rfd3/scripts/eval/` (scorer, designability, metrics) with `kit/` (natural
  5-target panel manifest `kit/manifest.json`, reference tensors, fold-recovery scripts) is the evaluation harness behind the numbers
  above; `generate.py` defaults to this manifest (`HARMONIA_MANIFEST` overrides).
- `triadj_ext/`: helper modules the data and generation code import (RING-format dataset reader, optimiser glue, conditional
  generation, design-with-contacts). They come from the TRIAD-J research code, are used unmodified, and are only the subset that
  `harmonia_ext` imports.

Training needs data that is not in this repository (a PDB mirror tar with index and the RING-format metadata the dataset reader loads;
see `triadj_ext/ring_data.py`).
