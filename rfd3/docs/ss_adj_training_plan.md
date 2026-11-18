# SS/ADJ training plan (Harmonia-RFD3)

Harmonia-RFD3 (`epoch-0698`, from stage 10) is the end of a staged fine-tune of the pretrained RFD3 checkpoint. Every stage after the first
keeps the adapters zero-initialised at the start, so the model reproduces RFD3 until the conditioning is learnt.

| Stage | What was trained | Added |
|---|---|---|
| 1 | SS and ADJ adapters, frozen backbone | per-residue SS and block adjacency (protein H/E blocks) |
| 2 | full model, low learning rate | integration with the backbone |
| 3-4 | full model | local attention, motif scaffolding, tar-backed data |
| 5 | full model | geometry: contact-well adjacency loss (`use_cb`, negative weight 0.5) |
| 6 | full model (epoch 0694) | molecule-type-aware blocks (ligand and nucleic acid), typed adjacency |
| 7 | full model | loop adjacency |
| **10** | full model, rollback to epoch 0694 with a minimal change set; epochs 695-703 | **epoch 0698 = Harmonia-RFD3** |

Stages 7-9 added several changes that were never individually attributed (antibody upsampling, interface-area term, SUPER /
HOTSPOT / COLDSPOT states, binder-oriented motif masks) and produced worse folds at saturated recovery. Stage 10 therefore restarts
from epoch 0694 and carries only: loop adjacency, crop size 384, the fix that skips block pairs whose both sides are fixed motif
(so the designed-side gradient is not diluted), cross-chain interface pinning (`ss_adj_cross_chain_seed_prob 0.7`), and the
monotonic steric normalisation with a NaN guard. See the header of
`foundry/models/rfd3/configs/experiment/ss_adj_phase10_rollback.yaml`.

## Reproducing stage 10

You need epoch 0694 (`epoch-0694_phase6.ckpt`, the intermediate checkpoint it starts from; not part of the release download, available from the authors), a PDB mirror with the metadata parquets, and the AtomWorks patch.

```bash
# 1. data: DSSP + block adjacency tables (scripts/data), PDB mirror as one tar + index
python scripts/data/build_pdb_mirror_tar_index.py --tar_path /path/to/pdb_mirror.tar --out_path /path/to/pdb_mirror_tar_index.parquet
# 2. paths: copy foundry/models/rfd3/configs/paths/data/local_ss_adj.yaml and edit (or set HARMONIA_* environment variables)
# 3. train (4 GPUs, resumes from the newest checkpoint in LOG_DIR)
SEED_CKPT=epoch-0694_phase6.ckpt LOG_DIR=runs/phase10 GPUS=4 MAX_EPOCHS=704 bash scripts/training/train_harmonia_rfd3.sh
```

`max_epochs` is an absolute target epoch. The script is a reference launcher: it carries no scheduler directives, and the training
data are not part of this repository.
