# SS-Ratio Conditioning for RFD3

Adds a continuous **global** secondary-structure conditioning token (`ss_ratio`)
to RFD3. One scalar per protein chain biases generation toward a target α/β
ratio without requiring per-residue topology knowledge at inference time.

---

## Concept

```
ss_ratio = E / (H + E)   ∈ [0, 1]
  0.0  →  pure α-helix
  1.0  →  pure β-sheet
 -1.0  →  unconditional sentinel (model uses its prior; used for CFG)

H = helix fraction  (biotite annotate_sse label 'a')
E = strand fraction (biotite annotate_sse label 'b')
```

**Global, not per-residue.** A single scalar is computed per chain and
broadcast to every token. The model learns to associate that global budget
with the overall SS composition of generated structures.

A **25 % conditioning dropout** is applied at training time: one quarter of
examples receive `ss_ratio = -1.0` (unconditional). This trains the model to
run both with and without the signal, which is required for
classifier-free guidance (CFG) at inference.

---

## Architecture

```
ss_ratio scalar  [I, 1]   (same value broadcast to all I tokens)
        │
        ▼
  SSRatioEncoder                    ← NEW module (ss_conditioning.py)
  ┌─────────────────────────────┐
  │  clamp to [0, 1]            │
  │  × fixed Fourier freqs      │   k·π for k = 1…16
  │  → [sin(·), cos(·)]  [I,32] │   fixed basis, not learned
  │  Linear(32 → 384)           │   learned
  │  SiLU                       │
  │  Linear(384 → 384)          │   learned, ZERO-INITIALISED
  │  × mask (0 if sentinel)     │   zeros out unconditional samples
  └─────────────────────────────┘
        │
        ▼  additive injection
       S_I  [I, 384]               token representation in TokenInitializer
        │
        ▼
  → rest of diffusion module
```

The final `Linear(384 → 384)` is zero-initialised so at the start of
fine-tuning the encoder contributes exactly zero to `S_I`. The pre-trained
backbone behaviour is preserved until gradients begin to update it.

**Why Fourier features and not a single linear layer?**
A `Linear(1, 384)` can only learn a monotone response to the ratio — one
global "direction" in representation space. The Fourier expansion lets the
model learn independent, frequency-specific responses (e.g. behave
differently near 0.0, 0.5, and 1.0 simultaneously).

---

## Files changed

### Data preprocessing (new)

| File | Purpose |
|------|---------|
| `atomworks/Shell/create_parquet_splits.py` | Splits metadata parquets by deposition date into train / val sets |
| `atomworks/Shell/create_parquet_splits.sh` | SLURM wrapper (~3 min, 32 GB RAM, no GPU) |

**Output:**
```
Data_Mirrors/metadata/splits/
  pn_units_df_train.parquet    (deposition < 2024-09-01)   3,713,949 rows
  pn_units_df_val.parquet      (2024-09-01 – 2024-12-16)      38,927 rows
  interfaces_df_train.parquet                             6,686,397 rows
  interfaces_df_val.parquet                                  79,882 rows
```

### Code changes

| File | Change |
|------|--------|
| `transforms/ppi_transforms.py` | Extended `AddGlobalIsNonLoopyFeature.forward()` to compute `ss_ratio` and set atom-array annotation with 25 % dropout |
| `transforms/design_transforms.py` | `FeaturizeAtoms.forward()` reads the `ss_ratio` annotation and writes `data["feats"]["ss_ratio"]` as `[I, 1]` token-level tensor — bypasses `OneDFeatureEmbedder` so there is no double-processing |
| `model/layers/ss_conditioning.py` | **New.** `SSRatioEncoder`: fixed Fourier basis → Linear → SiLU → Linear (zero-init) |
| `model/layers/encoders.py` | `TokenInitializer.__init__` optionally instantiates `SSRatioEncoder`; `init_tokens()` injects it additively into `S_I` after `process_s_init` |
| `configs/model/components/rfd3_net.yaml` | Removed `ss_ratio: 1` from `token_1d_features`; added `ss_conditioning: {num_freqs: 16}` block |
| `constants.py` | Added `"ss_ratio": -1.0` to `OPTIONAL_CONDITIONING_VALUES` (inference default = unconditional) |

### New config files

| File | Purpose |
|------|---------|
| `configs/paths/data/gerstung_local.yaml` | Local path overrides (PDB mirror + split parquets) |
| `configs/experiment/ss_ratio_finetune.yaml` | Phase 1 fine-tune: frozen backbone, 5 epochs, LR 1e-4 |

---

## Full training pipeline

```
CIF file loaded → atom_array
       │
       ▼  [99 % of batches]
AddGlobalIsNonLoopyFeature.forward()
  ├── annotate_sse()  →  H/E/C per token   (backbone geometry, ~1 ms/structure)
  ├── is_non_loopy annotation              (unchanged)
  └── ss_ratio annotation:
        75 %  →  ss_ratio = E/(H+E)  ∈ [0, 1]   (conditioned)
        25 %  →  ss_ratio = -1.0                  (unconditional dropout)
       │
       ▼
FeaturizeAtoms.forward()
  reads atom_array.ss_ratio at token_starts
  → data["feats"]["ss_ratio"]  shape [I, 1]
  (OneDFeatureEmbedder never sees ss_ratio)
       │
       ▼
TokenInitializer.forward()  →  init_tokens()
  S_I = token_1d_embedder(f, I)         # existing features unchanged
  S_I = process_s_init(S_I)
  S_I += SSRatioEncoder(f["ss_ratio"])  # NEW: Fourier-encoded additive injection
  Z_II = ...                            # pair repr derived from updated S_I
       │
       ▼
Diffusion module  →  loss
  L_total = 4.0 × L_diff  +  0.1 × L_seq
```

### Why `annotate_sse` and not full DSSP

`biotite.struc.annotate_sse` uses backbone dihedral geometry only, assigns
`a` (helix) / `b` (strand) / `c` (coil). It is already called inside
`get_secondary_structure_types()` which `AddGlobalIsNonLoopyFeature` already
invokes — the ss_ratio computation adds negligible overhead. Full DSSP
requires an external binary and is ~100× slower per structure.

### Why -1.0 as the unconditional sentinel

Valid `ss_ratio` values live in `[0, 1]`. The value `-1.0` is unambiguously
outside this range. `SSRatioEncoder.forward` clamps the input to `[0, 1]`
and zeroes the output for any token where `ss_ratio < -0.5`, so the sentinel
is clean and consistent with how `is_non_loopy` uses `-1` for its
"loopy / no conditioning" state.

---

## Running the fine-tune

### Step 1 — Create parquet splits (already done — job 393501)

```bash
bash atomworks/Shell/create_parquet_splits.sh
# Monitor: tail -f Data_Mirrors/logs/parquet_splits-<JOB>.out
```

### Step 2 — Smoke-test (job 393511 submitted)

```bash
bash atomworks/Shell/smoketest_ss_ratio.sh
# Monitor: tail -f logs/smoketest-<JOB>.out
```

What to verify in the smoke-test log:
- No import or config errors
- `data["feats"]["ss_ratio"]` shape is `[I, 1]`
- Values are in `[0, 1]` or exactly `-1.0`
- No NaN in loss or gradients
- Only `token_initializer.ss_encoder.*` parameters appear in the optimizer

### Step 3 — Submit Phase 1 training

Once the smoke test passes, write and submit a full SLURM training job:

```bash
python models/rfd3/train.py \
    experiment=ss_ratio_finetune \
    ckpt_config.path=/work/gerstung/RFD3_Training/foundry/base_models/rfd3_latest.ckpt \
    "paths/data=gerstung_local" \
    paths.log_dir=/work/gerstung/RFD3_Training/logs
```

Expected: ~3–6 hours on a single GPU for 5 epochs at 2400 examples/epoch.

---

## Verifying that conditioning works (post-training)

Generate structures at fixed `ss_ratio` targets and measure achieved SS
content with DSSP:

```python
from biotite.application.dssp import DsspApp

for target in [0.0, 0.25, 0.5, 0.75, 1.0]:
    feats["ss_ratio"] = torch.full((n_tokens, 1), target)
    # ... run model inference ...
    app = DsspApp.create_app(atom_array)
    app.start(); app.join()
    sse = app.get_sse()
    achieved = (sse == "E").sum() / max((sse != "C").sum(), 1)
    print(f"target={target:.2f}  achieved={achieved:.2f}")
```

A working model shows monotonically increasing sheet content as `ss_ratio`
goes from 0 → 1.

---

## Phase 2 — optional backbone unfreeze

If Phase 1 conditioning is weak (achieved ratio doesn't correlate with
target), unfreeze the backbone with differential LR. In
`ss_ratio_finetune.yaml`:

```yaml
parameter_freezing_config:
  freeze_by_default: false   # unfreeze all

model:
  optimizer:
    lr: 5.0e-6               # backbone LR (10× lower than encoder)

trainer:
  max_epochs: 15
```

True per-param-group LR (encoder at 5e-5, backbone at 5e-6) requires
extending `construct_optimizer` in `FabricTrainer` to support param groups —
left for the larger training project.

---

## What this project teaches

| Concern | Where you see it |
|---------|-----------------|
| Extending a data transform | `AddGlobalIsNonLoopyFeature` edit |
| Bypassing `OneDFeatureEmbedder` for custom encoding | `FeaturizeAtoms` + `rfd3_net.yaml` |
| Building a Fourier feature encoder | `SSRatioEncoder` in `ss_conditioning.py` |
| Injecting into `TokenInitializer` | `encoders.py` — `init_tokens()` |
| Conditioning dropout for CFG | 25 % dropout sentinel in `ppi_transforms.py` |
| Zero-init for safe warm start | `SSRatioEncoder.linear2` + `WeightLoadingConfig` |
| Backbone freezing | `ParameterFreezingConfig` in experiment config |
| Local path overrides | `gerstung_local.yaml` |
| Train/val splitting by date | `create_parquet_splits.py` |
