#!/bin/bash
# Comprehensive SS/ADJ inference benchmark.
set -euo pipefail

ROOT=${ROOT:-/work/gerstung/HARMONIA}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}
FOUNDRY=${FOUNDRY:-${ROOT}/foundry}
PYTHON=${PYTHON:-/zfs/s01/z04/home/gerstung/miniforge3/envs/atomworks/bin/python3}

NODELIST=${NODELIST:-t38cn056}
PARTITION=${PARTITION:-barracuda}
RESERVATION=${RESERVATION:-AIPD}
DEVICES_PER_NODE=${DEVICES_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-96G}
TIME=${TIME:-48:00:00}

# 25 * 4 = 100 designs per case by default.
N_BATCHES=${N_BATCHES:-25}
DIFFUSION_BATCH_SIZE=${DIFFUSION_BATCH_SIZE:-4}
NUM_TIMESTEPS=${NUM_TIMESTEPS:-120}
CFG_SCALE=${CFG_SCALE:-2.0}
USE_CFG=${USE_CFG:-True}
SUBMIT=${SUBMIT:-1}

latest_checkpoint() {
    find "${LOG_DIR}/train/ss-adj-phase2" -name 'epoch-*.ckpt' -printf '%f\t%p\n' 2>/dev/null \
        | sort -V | tail -1 | cut -f2-
}

CKPT=${CKPT:-$(latest_checkpoint)}
[ -n "${CKPT}" ] && [ -s "${CKPT}" ] || { echo "ERROR: no Phase 2 checkpoint found"; exit 1; }

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-${LOG_DIR}/ss_adj_conditioning/${RUN_ID}}
INPUT_JSON=${OUT_ROOT}/ss_adj_conditioning_inputs.json
MANIFEST_JSON=${OUT_ROOT}/ss_adj_conditioning_manifest.json
mkdir -p "${OUT_ROOT}"

"${PYTHON}" "${ROOT}/atomworks/Shell/generate_ss_adj_conditioning_inputs.py" --out-root "${OUT_ROOT}"
N_CASES=$("${PYTHON}" - "${INPUT_JSON}" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))))
PY
)
N_PER_CASE=$(( N_BATCHES * DIFFUSION_BATCH_SIZE ))
N_TOTAL=$(( N_CASES * N_PER_CASE ))

echo "SS/ADJ conditioning benchmark"
echo "  Checkpoint    : ${CKPT}"
echo "  Output root   : ${OUT_ROOT}"
echo "  Input JSON    : ${INPUT_JSON}"
echo "  Manifest      : ${MANIFEST_JSON}"
echo "  Cases         : ${N_CASES}"
echo "  Designs/case  : ${N_PER_CASE} (${N_BATCHES} x ${DIFFUSION_BATCH_SIZE})"
echo "  Total designs : ${N_TOTAL}"
echo "  Node          : ${NODELIST}"
echo "  CFG           : ${USE_CFG} scale=${CFG_SCALE}"

if [ "${SUBMIT}" = "0" ]; then
    echo "SUBMIT=0: generated inputs only."
    exit 0
fi

INF_SCRIPT=$(mktemp /tmp/ss_adj_conditioning_inf.XXXXXX.sh)
cat > "${INF_SCRIPT}" << EOFINF
#!/bin/bash
#SBATCH --job-name=ss_adj_infer
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=${RESERVATION}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${DEVICES_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${OUT_ROOT}/inference-%j.out
#SBATCH --error=${OUT_ROOT}/inference-%j.err

set -euo pipefail
export HYDRA_FULL_ERROR=1
export DEBUG=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== SS/ADJ conditioning inference started: \$(date) ==="
echo "Checkpoint: ${CKPT}"
echo "Cases: ${N_CASES}; designs/case: ${N_PER_CASE}; total: ${N_TOTAL}"
echo "GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

cd "${FOUNDRY}"
srun --cpu-bind=none "${PYTHON}" models/rfd3/src/rfd3/run_inference.py \
    ckpt_path="${CKPT}" \
    inputs="${INPUT_JSON}" \
    out_dir="${OUT_ROOT}/designs" \
    global_prefix="" \
    skip_existing=False \
    n_batches=${N_BATCHES} \
    diffusion_batch_size=${DIFFUSION_BATCH_SIZE} \
    devices_per_node=${DEVICES_PER_NODE} \
    inference_sampler.num_timesteps=${NUM_TIMESTEPS} \
    inference_sampler.use_classifier_free_guidance=${USE_CFG} \
    inference_sampler.cfg_scale=${CFG_SCALE} \
    dump_prediction_metadata_json=True \
    dump_trajectories=False \
    prevalidate_inputs=True

echo "=== SS/ADJ conditioning inference complete: \$(date) ==="
EOFINF

INF_JOB=$(sbatch --parsable "${INF_SCRIPT}")
echo "Submitted inference job ${INF_JOB}"

AN_SCRIPT=$(mktemp /tmp/ss_adj_conditioning_analysis.XXXXXX.sh)
cat > "${AN_SCRIPT}" << EOFAN
#!/bin/bash
#SBATCH --job-name=ss_adj_analyze
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --dependency=afterany:${INF_JOB}
#SBATCH --output=${OUT_ROOT}/analysis-%j.out
#SBATCH --error=${OUT_ROOT}/analysis-%j.err

set -euo pipefail
export DEBUG=false
"${PYTHON}" "${ROOT}/atomworks/Shell/analyze_ss_adj_conditioning.py" --out-root "${OUT_ROOT}"
EOFAN

AN_JOB=$(sbatch --parsable "${AN_SCRIPT}")
echo "Submitted analysis job ${AN_JOB} (afterany:${INF_JOB})"
echo "Monitor inference: tail -f ${OUT_ROOT}/inference-${INF_JOB}.out"
echo "Monitor analysis : tail -f ${OUT_ROOT}/analysis-${AN_JOB}.out"
