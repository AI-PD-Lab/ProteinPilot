#!/bin/bash
# compute_ss_ratio_cache.sh
#
# Computes ss_ratio = E/(H+E) for every assembly in the CIF cache and writes
# per-task parquets to metadata/ss_ratio/partial/.  A merge job is submitted
# automatically once all array tasks succeed, which joins the results back into
# the training parquet and writes interfaces_df_train_with_ss.parquet.
#
# Usage:
#   bash atomworks/Shell/compute_ss_ratio_cache.sh

set -euo pipefail
DATA_ROOT=${DATA_ROOT:-/work/gerstung/RFD3_Training/Data_Mirrors}

MANIFEST=${DATA_ROOT}/metadata/ss_ratio/manifest.txt
PARTIAL_DIR=${DATA_ROOT}/metadata/ss_ratio/partial
PYTHON=/zfs/s01/z04/home/gerstung/miniforge3/envs/atomworks/bin/python3
LOG_DIR=/work/gerstung/HARMONIA/logs/ss_ratio_cache

N_LINES=$(wc -l < "$MANIFEST")
N_TASKS=800
CHUNK=$(( (N_LINES + N_TASKS - 1) / N_TASKS ))   # ceil division

mkdir -p "$PARTIAL_DIR" "$LOG_DIR"

echo "Manifest : $MANIFEST  ($N_LINES files)"
echo "Tasks    : $N_TASKS  (~$CHUNK files each)"
echo "Partial  : $PARTIAL_DIR"

# ── Array script ──────────────────────────────────────────────────────────────
ARRAY_SCRIPT=$(mktemp /tmp/ss_ratio_array.XXXXXX.sh)
cat > "$ARRAY_SCRIPT" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_ratio_cache
#SBATCH --partition=barracuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-$((N_TASKS - 1))
#SBATCH --output=${LOG_DIR}/task-%A-%a.out
#SBATCH --error=${LOG_DIR}/task-%A-%a.err

set -euo pipefail
DATA_ROOT=${DATA_ROOT:-/work/gerstung/RFD3_Training/Data_Mirrors}

TASK_ID=\$SLURM_ARRAY_TASK_ID
START=\$(( TASK_ID * ${CHUNK} ))
END=\$(( START + ${CHUNK} - 1 ))
OUT=${PARTIAL_DIR}/task_\$(printf '%06d' \$TASK_ID).parquet

# Skip if already done
if [[ -f "\$OUT" ]]; then
    echo "[SKIP] \$OUT exists"
    exit 0
fi

echo "Task \$TASK_ID: lines \$START-\$END → \$OUT"

${PYTHON} - "${MANIFEST}" "\$START" "${CHUNK}" "\$OUT" << 'PYEOF'
import gzip, pickle, re, sys
import numpy as np
import pandas as pd
import biotite.structure as struc

manifest_path, start_s, chunk_s, out_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

with open(manifest_path) as f:
    lines = f.read().splitlines()

batch = lines[start_s : start_s + chunk_s]
rows = []

for path in batch:
    # Extract pdb_id and assembly_id from path
    # e.g. .../5qlb.cif/5qlb.cif_assembly_1.pkl.gz
    m = re.search(r'/([a-z0-9]{4})\.cif/\1\.cif_assembly_(\d+)\.pkl\.gz$', path)
    if not m:
        continue
    pdb_id, assembly_id = m.group(1), m.group(2)

    try:
        with gzip.open(path, 'rb') as f:
            obj = pickle.load(f)
        atoms = obj['asym_unit'][0]
        protein = atoms[struc.filter_amino_acids(atoms)]
        if protein.array_length() == 0:
            continue
        sse = struc.annotate_sse(protein)
        n_h = int(np.sum(sse == 'a'))
        n_b = int(np.sum(sse == 'b'))
        n_c = int(np.sum(sse == 'c'))
        structured = n_h + n_b
        ss_ratio = float(n_b / structured) if structured > 0 else float('nan')
        rows.append({
            'pdb_id':      pdb_id,
            'assembly_id': assembly_id,
            'ss_ratio':    ss_ratio,
            'n_helix':     n_h,
            'n_sheet':     n_b,
            'n_coil':      n_c,
            'n_residues':  protein.array_length(),
        })
    except Exception as e:
        # Log and skip bad structures — don't crash the whole task
        print(f"[WARN] {path}: {e}", flush=True)
        continue

if rows:
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"Wrote {len(rows)} rows → {out_path}")
else:
    print("No valid structures in this chunk")
PYEOF
EOFJOB

ARRAY_JOB=$(sbatch --parsable "$ARRAY_SCRIPT")
echo "Submitted array job ${ARRAY_JOB} (tasks 0–$((N_TASKS-1)))"

# ── Merge + join script (runs after all array tasks) ─────────────────────────
MERGE_SCRIPT=$(mktemp /tmp/ss_ratio_merge.XXXXXX.sh)
cat > "$MERGE_SCRIPT" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_ratio_merge
#SBATCH --dependency=afterok:${ARRAY_JOB}
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --partition=barracuda
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/merge-%j.out
#SBATCH --error=${LOG_DIR}/merge-%j.err

set -euo pipefail
DATA_ROOT=${DATA_ROOT:-/work/gerstung/RFD3_Training/Data_Mirrors}
echo "=== Merge + join started: \$(date) ==="

${PYTHON} - << 'PYEOF'
import pandas as pd
import numpy as np
from pathlib import Path

partial_dir = Path("${DATA_ROOT}/metadata/ss_ratio/partial")
out_dir     = Path("${DATA_ROOT}/metadata/ss_ratio")
train_path  = Path("${DATA_ROOT}/metadata/splits/interfaces_df_train.parquet")

# ── 1. Merge all partial parquets ─────────────────────────────────────────────
parts = sorted(partial_dir.glob("task_*.parquet"))
print(f"Merging {len(parts)} partial files...")
ss_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

# De-duplicate: keep one ss_ratio per (pdb_id, assembly_id)
ss_df = ss_df.drop_duplicates(subset=['pdb_id','assembly_id'])
ss_df.to_parquet(out_dir / "ss_ratio_by_assembly.parquet", index=False)
print(f"Wrote {len(ss_df)} assemblies → ss_ratio_by_assembly.parquet")

# Distribution summary
valid = ss_df['ss_ratio'].dropna()
print(f"\nss_ratio distribution ({len(valid)} valid assemblies):")
for lo in np.arange(0, 1.0, 0.1):
    hi = lo + 0.1
    n = int(((valid >= lo) & (valid < hi)).sum())
    bar = '█' * (n // 500)
    print(f"  [{lo:.1f}–{hi:.1f}): {n:6d}  {bar}")

# ── 2. Join to training parquet ───────────────────────────────────────────────
print("\nJoining to training parquet...")
train_df = pd.read_parquet(train_path)
train_df['assembly_id'] = train_df['assembly_id'].astype(str)
ss_df['assembly_id']    = ss_df['assembly_id'].astype(str)

merged = train_df.merge(
    ss_df[['pdb_id','assembly_id','ss_ratio','n_helix','n_sheet','n_coil','n_residues']],
    on=['pdb_id','assembly_id'],
    how='left',
)
coverage = merged['ss_ratio'].notna().mean()
print(f"Coverage: {coverage:.1%} of training interfaces have ss_ratio")

# ── 3. Compute stratified reweighting ─────────────────────────────────────────
# Target: uniform distribution across 10 ss_ratio decile bins.
# Multiply each row's existing sampler weight by 1/p(bin) so rare bins
# are upsampled. Uncovered rows (ss_ratio NaN) get weight=1 (neutral).
print("\nComputing stratified reweighting...")
bins = np.linspace(0, 1.0, 11)
bin_labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(bins[:-1], bins[1:])]
merged['ss_ratio_bin'] = pd.cut(
    merged['ss_ratio'], bins=bins, labels=bin_labels, include_lowest=True
)
bin_counts = merged['ss_ratio_bin'].value_counts()
bin_freq   = bin_counts / bin_counts.sum()
# Inverse-frequency weight, clipped to 10× max to avoid extreme upsampling
inv_freq   = 1.0 / bin_freq
inv_freq   = inv_freq / inv_freq.mean()          # normalise to mean=1
inv_freq   = inv_freq.clip(upper=10.0)           # cap at 10×
print("Stratum weights (clipped at 10×):")
for b, w in inv_freq.sort_index().items():
    n = bin_counts.get(b, 0)
    print(f"  {b}: n={n:7d}  weight={w:.3f}")

merged['ss_ratio_stratum_weight'] = merged['ss_ratio_bin'].map(inv_freq).fillna(1.0)

# ── 4. Write final parquet ────────────────────────────────────────────────────
out_path = out_dir / "interfaces_df_train_with_ss.parquet"
merged.to_parquet(out_path, index=False)
print(f"\nWrote {len(merged)} rows → {out_path}")
print("New columns added: ss_ratio, n_helix, n_sheet, n_coil, n_residues, ss_ratio_bin, ss_ratio_stratum_weight")
print("=== Done ===")
PYEOF

echo "=== Merge + join complete: \$(date) ==="
EOFJOB

MERGE_JOB=$(sbatch --parsable "$MERGE_SCRIPT")
echo "Submitted merge job ${MERGE_JOB} (runs after array completes)"
echo ""
echo "Monitor array : tail -f ${LOG_DIR}/task-${ARRAY_JOB}-0.out"
echo "Monitor merge : tail -f ${LOG_DIR}/merge-${MERGE_JOB}.out"
echo "Output        : ${DATA_ROOT}/metadata/ss_ratio/"
