# HARMONIA on RFD4

SS + block-adjacency fold conditioning for RFD4-Proteina: the HARMONIA idea from the RFD3 lineage, ported as a full fine-tune of RFD4.
No RING labels, no LoRA, no CATH or other extra labels: compactness comes from the adjacency conditioning itself.

## What the model takes

| input | tensor | channel |
|---|---|---|
| length | - | as in RFD4 |
| secondary structure | `[L]` int, RFD3 codes 0 helix / 1 strand / 2 loop / 3 masked | RFD4's native `C_SEC` channel (remapped to 1 / 2 / 3 / 0) |
| block adjacency | `[L, L]` int states 0 non-contact / 1 contact / 2 masked / 3 HOT / 4 COLD / 5 pad | `adj_embed` = `nn.Embedding(6, pair_dim)`, zero-init, added to the pair input `z0` |

The RFD3 HARMONIA spec tensors (`<target>__<arm>.ss.npy` / `.adj.npy`) are used as-is. Masked / pad states are multiplied by 0, so an
all-masked map reproduces the released RFD4 bit-for-bit. Guidance: `pred = pred_masked + w (pred_cond - pred_masked)` over SS + ADJ jointly
(masked pass = no SS, no ADJ); release operating point **w = 1.5**.

## Code

- `adj_model.py` - `HarmoniaLocalLatentsTransformer` (registered over RFD4's trunk class via `register()`).
- `adj_labels.py` / `adj_data.py` - SS blocks, CB block contacts (min CB-CB < 6 A), curriculum (uncond / ss_only / adj_only / full / sparse),
  HOT/COLD, compactness gate, dense-zero dialect.
- `generate.py` - panel / spec generation with HARMONIA guidance (`--cfgs 1.5`, `--hot/--cold`, `--manifest`).
- `configs/experiment/harmonia_r{1,2,3}.yaml` - training recipes (release = `harmonia_r3`); `train_r1.sbatch` - preempt-safe training.
- `eval_r1.sbatch`, `score_eval.sh`, `hotcold_score.py` - evaluation harness (natural 5-target panel, Rossmann dense-zero stress, HOT/COLD).

## Weights

Hugging Face `AI-PD/Harmonia`, folder `Harmonia-RFD4/` (R3) (`config.json` + `model.safetensors`, RFD4 bundle format):

```python
from harmonia_ext.adj_model import register
register()                                   # before building the model: adds adj_embed to the trunk
from rfproteina.proteina import Proteina
model = Proteina.from_pretrained("path/to/Harmonia-RFD4")
```

Generation (container `rfd4-cu13.sif`, PYTHONPATH `src:.`):
`python -m harmonia_ext.generate --bundle <dir> --tag <tag> --out-tar <tar> --manifest <manifest.json> --cfgs 1.5 --n 16`

## Release numbers (R3 8k EMA, CFG 1.5; 5 natural targets x full/sparse8/dense0_true, n = 240)

| | stock RFD3 (length only) | stock RFD4 (SS only) | RFD3 HARMONIA p0694 | **HARMONIA-RFD4 R3** |
|---|---|---|---|---|
| fold recovery (SS + ADJ) | 0.48 | 0.81 | 0.86 | **0.93** |
| perfect-spec rate | 0.01 | 0.14 | 0.30 | **0.45** |
| TM >= 0.5 to native | 0.00 | 0.13 | 0.25 | **0.49** |
| split rate | 0.8 % | 10.8 % | 1.2 % | **0.4 %** |
| designability (MPNN x8 -> ESMFold2) | 0.72 | 0.95 | 0.59-0.62 | 0.93 |

HOT +0.80 / COLD +0.61 contact-count shift at the marked pairs (w = 1.5). 
