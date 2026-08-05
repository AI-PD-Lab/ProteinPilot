# SS/ADJ Training Plan

## Data

- Standard training metadata: task-specific copies of the PDB pn-unit and interface parquets.
- Loop training metadata: `Data_Mirrors/ss_adj_loop/loop_train_dataset.csv` (1,790 clusters).
- Loop validation metadata: `Data_Mirrors/ss_adj_loop/loop_val_dataset.csv` (201 clusters, 191 PDBs).
- Validation PDBs are excluded from both standard training parquets and the loop training CSV.

Regenerate the data products with:

```bash
NODELIST=t38cn056 atomworks/Shell/build_ss_adj_loop_dataset.sh
```

## Phases

| Phase | Trainable parameters | Standard/loop mix | Examples | Expected loop draws |
|---|---|---:|---:|---:|
| 1A | SS and ADJ adapters | 100/0 | 19,200 | 0 |
| 1B | SS and ADJ adapters | 70/30 | 30,000 | 9,000 |
| 2 | Full model except shadow weights | 93/7 | 156,000 | 10,920 |

Across Phases 1B and 2, each loop-training cluster is sampled about 11 times on average. Phase 1A first teaches the adapters broad secondary-structure and ordinary adjacency conditioning. Phase 1B then concentrates on loop-mediated contacts. Phase 2 integrates conditioning into the full model at low learning rate.

## Submission Order

```bash
NODELIST=t38cn053 atomworks/Shell/train_ss_adj_phase1a.sh
NODELIST=t38cn053 atomworks/Shell/train_ss_adj_phase1b.sh
NODELIST=t38cn053 atomworks/Shell/train_ss_adj_phase2.sh
```

Node `t38cn056` is also accepted. Its launchers force a single GPU request so Slurm can assign one MIG partition safely. Each later phase requires the expected final checkpoint from the preceding phase unless `CKPT` is supplied explicitly.
