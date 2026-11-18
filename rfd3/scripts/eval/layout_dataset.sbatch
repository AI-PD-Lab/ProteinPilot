#!/bin/bash
#SBATCH --job-name=n031layouts
#SBATCH --partition=preempt_cpu
#SBATCH --requeue
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --array=0-11
#SBATCH --output=/path/to/work_rfd3/v2/eval/logs/n031_layout_%A_%a.out
# Submit this array, then the SAME script --dependency=afterok:<array> --array=12.
# CPU: 48 independent label workers, full NUMA fraction; no GPU benefit for CIF/P-SEA.
set -euo pipefail
REPO=${REPO:-/path/to/work_rfd3/harmonia_study}
OUT=${OUT:-/path/to/work_rfd3/v2/eval/layout_planner}
mkdir -p "$OUT"
RESTORE=$(mktemp -d "/tmp/n031_env_${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf "$RESTORE"' EXIT
if [[ -n ${RFD3_PY:-} && -x ${RFD3_PY} ]]; then
    PY=$RFD3_PY
else
    tar xf /path/to/work/rfd3/rfd3_env.tar -C "$RESTORE"
    PY=$RESTORE/rfd3_ssadj_test/bin/python
fi
export PYTHONPATH="$REPO/scripts/eval:$REPO/foundry/models/rfd3/src:$REPO/foundry/src:$REPO/atomworks/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$RESTORE/mpl" XDG_CACHE_HOME="$RESTORE/cache"
if [[ ${SLURM_ARRAY_TASK_ID} -eq 12 ]]; then
    # Global homology scans use all 48 workers. Compression/split are serial
    # reductions after that scan. Keep pieces for reproducible merge reruns.
    "$PY" "$REPO/scripts/eval/layout_dataset.py" merge --pieces "$OUT"/piece_??.npz --out "$OUT/layouts.npz" --workers "$SLURM_CPUS_PER_TASK" ${MERGE_EXTRA:-}
    # MERGE_EXTRA, e.g. --allow-shortfall (red-team 5: the beta quota margin is only ~16 layouts)
else
    "$PY" "$REPO/scripts/eval/layout_dataset.py" extract --piece "$SLURM_ARRAY_TASK_ID" \
        --pieces 12 --workers "$SLURM_CPUS_PER_TASK" --budget 420 --out-dir "$OUT"
fi
