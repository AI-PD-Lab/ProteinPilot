#!/bin/bash
# patch_ss_ratio_weights.sh
#
# After compute_ss_ratio_cache.sh has finished, this script:
#   1. Verifies the enriched parquet exists
#   2. Replaces the interface training parquet pointer in the ss_ratio
#      finetune experiment config with the new ss-ratio-annotated one
#   3. Adds ss_ratio_stratum_weight as a multiplier in the sampler weight
#      formula via a new dataset config overlay
#
# Usage:
#   bash atomworks/Shell/patch_ss_ratio_weights.sh

set -euo pipefail
DATA_ROOT=${DATA_ROOT:-/path/to/work/RFD3_Training/Data_Mirrors}

PYTHON=/path/to/home/miniforge3/envs/atomworks/bin/python3
SS_PARQUET=${DATA_ROOT}/metadata/ss_ratio/interfaces_df_train_with_ss.parquet
FOUNDRY=/path/to/work/HARMONIA/foundry

if [[ ! -f "$SS_PARQUET" ]]; then
    echo "ERROR: $SS_PARQUET not found — run compute_ss_ratio_cache.sh first"
    exit 1
fi

echo "Verifying enriched parquet..."
$PYTHON - "$SS_PARQUET" << 'PY'
import pandas as pd, sys
df = pd.read_parquet(sys.argv[1])
print(f"  Rows           : {len(df):,}")
print(f"  ss_ratio cover : {df['ss_ratio'].notna().mean():.1%}")
print(f"  Stratum weight : min={df['ss_ratio_stratum_weight'].min():.3f}  max={df['ss_ratio_stratum_weight'].max():.3f}  mean={df['ss_ratio_stratum_weight'].mean():.3f}")
PY

# ── Add ss_ratio_stratum_weight support to the weight calculator ──────────────
# The existing calculate_weights_for_pdb_dataset_df function doesn't know about
# ss_ratio_stratum_weight.  We monkey-patch it by adding a wrapper that multiplies
# the existing weight by the stratum weight column when present.
SAMPLERS_PY=$FOUNDRY/src/foundry/utils/datasets.py
WEIGHT_PY=/path/to/work/HARMONIA/atomworks/src/atomworks/ml/samplers.py

echo "Patching sampler weight function..."
$PYTHON - "$WEIGHT_PY" << 'PY'
import sys, re
path = sys.argv[1]
src = open(path).read()

PATCH = '''

def calculate_weights_for_pdb_dataset_df(df, **kwargs):
    """Drop-in replacement that multiplies AF3 cluster weights by ss_ratio_stratum_weight.

    If the column is absent (old parquet without ss annotation) the function
    falls back to the original behaviour, so existing pipelines are unaffected.
    """
    base_weights = calculate_af3_example_weights(df, **kwargs)
    if "ss_ratio_stratum_weight" in df.columns:
        import pandas as _pd
        strat = df["ss_ratio_stratum_weight"].fillna(1.0).astype(float)
        base_weights = base_weights * strat
    return base_weights
'''

if 'def calculate_weights_for_pdb_dataset_df' not in src:
    src = src + PATCH
    open(path, 'w').write(src)
    print("  Patched calculate_weights_for_pdb_dataset_df into samplers.py")
else:
    print("  calculate_weights_for_pdb_dataset_df already present — skipping")
PY

# ── Write new cluster_a_local data paths pointing at enriched parquet ──────────
NEW_PATHS=$FOUNDRY/models/rfd3/configs/paths/data/cluster_a_local_ss.yaml
cat > "$NEW_PATHS" << YAML
# Local paths for Gerstung lab cluster — same as cluster_a_local but with the
# ss-ratio-enriched training parquet for Phase 3 stratified sampling.

pdb_data_dir: ${DATA_ROOT}/pdb
pdb_parquet_dir: ${DATA_ROOT}/metadata/ss_ratio_splits

monomer_distillation_data_dir: null
monomer_distillation_parquet_dir: null
design_benchmark_data_dir: null
design_model_weight_dir: null
residue_cache_dir: null
cif_cache_dir: ${DATA_ROOT}/cif_cache
failed_examples_dir: /path/to/work/HARMONIA/logs/failed_examples
YAML

# Write ss_ratio parquet into a splits directory that the training pipeline expects
SS_SPLITS=${DATA_ROOT}/metadata/ss_ratio_splits
mkdir -p "$SS_SPLITS"

echo "Creating split symlinks..."
$PYTHON - "$SS_PARQUET" "$SS_SPLITS" << PY
import pandas as pd, shutil, sys
from pathlib import Path

src = Path(sys.argv[1])
splits_dir = Path(sys.argv[2])
orig_splits = Path("${DATA_ROOT}/metadata/splits")

# Copy enriched training parquet
dst = splits_dir / "interfaces_df_train.parquet"
shutil.copy2(src, dst)
print(f"  Copied enriched train parquet → {dst}")

# Symlink the val and pn_units parquets (unchanged)
for name in ["interfaces_df_val.parquet", "pn_units_df_train.parquet", "pn_units_df_val.parquet"]:
    link = splits_dir / name
    if not link.exists():
        link.symlink_to(orig_splits / name)
        print(f"  Symlinked {name}")
    else:
        print(f"  {name} already exists")

# interfaces_df.parquet is a symlink alias for interfaces_df_train.parquet (expected by dataset loader)
alias = splits_dir / "interfaces_df.parquet"
if not alias.exists():
    alias.symlink_to(splits_dir / "interfaces_df_train.parquet")
    print(f"  Symlinked interfaces_df.parquet → interfaces_df_train.parquet")
else:
    print(f"  interfaces_df.parquet already exists")
PY

echo ""
echo "Done. To launch Phase 3 training, use:"
echo "  paths/data=cluster_a_local_ss  (instead of cluster_a_local)"
echo "  in your EXTRA_OVERRIDES, or update train_ss_ratio_phase3.sh"
