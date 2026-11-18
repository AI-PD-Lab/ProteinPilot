#!/bin/bash
# SS/ADJ benchmark launcher — 6 parallel GPU jobs on t38cn056 (AIPD reservation).
#
# Strategy:
#   1. Generate 52-case input JSON via generate_ss_adj_conditioning_inputs_v2.py
#   2. Shard into N_SHARDS=6 sub-JSONs; each shard runs on one GPU
#   3. All shards write to the same designs/ directory (unique filenames per case)
#   4. Dependent analysis job: analyze_ss_adj_conditioning.py over merged designs
#   5. Dependent plot job:     plot_ss_adj_results.py (benchmark + training + binder refs)
#
# Usage:
#   bash run_ss_adj_benchmark_6gpu.sh
# Override defaults with env vars:
#   N_BATCHES=2  N_SHARDS=4  bash run_ss_adj_benchmark_6gpu.sh
set -euo pipefail

ROOT=${ROOT:-/path/to/work/HARMONIA}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}
FOUNDRY=${FOUNDRY:-${ROOT}/foundry}
SHELL_DIR=${SHELL_DIR:-${ROOT}/atomworks/Shell}
PYTHON=${PYTHON:-/path/to/home/miniforge3/envs/atomworks/bin/python3}

PARTITION=${PARTITION:-barracuda}
NODELIST=${NODELIST:-t38cn056}
RESERVATION=${RESERVATION:-AIPD}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM_PER_JOB=${MEM_PER_JOB:-96G}
INFER_TIME=${INFER_TIME:-24:00:00}
ANALYSIS_TIME=${ANALYSIS_TIME:-2:00:00}
PLOT_TIME=${PLOT_TIME:-0:30:00}

N_SHARDS=${N_SHARDS:-6}
N_BATCHES=${N_BATCHES:-1}
DIFFUSION_BATCH_SIZE=${DIFFUSION_BATCH_SIZE:-4}
NUM_TIMESTEPS=${NUM_TIMESTEPS:-120}
USE_CFG=${USE_CFG:-True}
CFG_SCALE=${CFG_SCALE:-2.0}

RUN_ID=${RUN_ID:-ss_adj_benchmark_6gpu_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-${LOG_DIR}/ss_adj_conditioning/${RUN_ID}}
DESIGNS_DIR=${DESIGNS_DIR:-${RUN_ROOT}/designs}
SHARD_DIR=${SHARD_DIR:-${RUN_ROOT}/shards}
PLOT_DIR=${PLOT_DIR:-${LOG_DIR}/ss_adj_plots/${RUN_ID}}

# External 20-design reference runs for binder-specific plots.
BETA_BINDER_REF=${BETA_BINDER_REF:-${LOG_DIR}/ss_adj_binding/beta_binder_adj_20260617_065857}
LOOP_BINDER_REF=${LOOP_BINDER_REF:-${LOG_DIR}/ss_adj_binding/loop_binder_adj_20260617_073232}

latest_checkpoint() {
    find "${LOG_DIR}/train/ss-adj-phase2" -name 'epoch-*.ckpt' -printf '%f\t%p\n' 2>/dev/null \
        | sort -V | tail -1 | cut -f2-
}

CKPT=${CKPT:-$(latest_checkpoint)}
[ -n "${CKPT}" ] && [ -s "${CKPT}" ] || {
    echo "ERROR: no checkpoint found; set CKPT=/path/to/model.ckpt"
    exit 1
}

TRAIN_ROOT=$(dirname "$(dirname "${CKPT}")")

mkdir -p "${RUN_ROOT}" "${DESIGNS_DIR}" "${SHARD_DIR}" "${PLOT_DIR}"

echo "=== SS/ADJ 6-GPU Benchmark: ${RUN_ID} ==="
echo "Checkpoint : ${CKPT}"
echo "Run root   : ${RUN_ROOT}"
echo "Shards     : ${N_SHARDS}"
echo "Designs    : $((N_BATCHES * DIFFUSION_BATCH_SIZE)) per case (${N_BATCHES}x${DIFFUSION_BATCH_SIZE})"
echo ""

# ── Step 1: generate inputs ──────────────────────────────────────────────────
echo "[1/4] Generating benchmark inputs..."
"${PYTHON}" "${SHELL_DIR}/generate_ss_adj_conditioning_inputs_v2.py" \
    --out-root "${RUN_ROOT}"

INPUT_JSON="${RUN_ROOT}/ss_adj_conditioning_inputs.json"
MANIFEST_JSON="${RUN_ROOT}/ss_adj_conditioning_manifest.json"
N_CASES=$("${PYTHON}" -c "import json; d=json.load(open('${INPUT_JSON}')); print(len(d))")
echo "  Cases generated: ${N_CASES}"
echo ""

# ── Step 2: shard the input JSON ─────────────────────────────────────────────
echo "[2/4] Sharding ${N_CASES} cases into ${N_SHARDS} shards..."
"${PYTHON}" - "${INPUT_JSON}" "${N_SHARDS}" "${SHARD_DIR}" <<'PYEOF'
import json, math, sys
from pathlib import Path

input_json = Path(sys.argv[1])
n_shards = int(sys.argv[2])
out_dir = Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(input_json.read_text())
keys = list(data.keys())
shard_size = math.ceil(len(keys) / n_shards)
for i in range(n_shards):
    shard_keys = keys[i * shard_size : (i + 1) * shard_size]
    if not shard_keys:
        continue
    shard = {k: data[k] for k in shard_keys}
    shard_path = out_dir / f"shard_{i:02d}.json"
    shard_path.write_text(json.dumps(shard, indent=2) + "\n")
    print(f"  shard_{i:02d}: {len(shard_keys)} cases → {shard_path}")
PYEOF
echo ""

# ── Step 3: submit inference jobs ────────────────────────────────────────────
echo "[3/4] Submitting ${N_SHARDS} inference jobs (1 GPU each, node ${NODELIST})..."
INFER_JOB_IDS=()

for SHARD_JSON in "${SHARD_DIR}"/shard_*.json; do
    SHARD_NAME=$(basename "${SHARD_JSON}" .json)
    SHARD_OUT="${RUN_ROOT}/${SHARD_NAME}"
    mkdir -p "${SHARD_OUT}"

    JOB_SCRIPT=$(mktemp /tmp/ss_adj_infer_shard.XXXXXX.sh)
    cat > "${JOB_SCRIPT}" <<EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_adj_6gpu
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=${RESERVATION}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_JOB}
#SBATCH --time=${INFER_TIME}
#SBATCH --output=${SHARD_OUT}/inference-%j.out
#SBATCH --error=${SHARD_OUT}/inference-%j.err

set -euo pipefail
export HYDRA_FULL_ERROR=1
export DEBUG=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Shard ${SHARD_NAME} inference started: \$(date) ==="
echo "GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

cd "${FOUNDRY}"
srun --cpu-bind=none "${PYTHON}" models/rfd3/src/rfd3/run_inference.py \
    ckpt_path="${CKPT}" \
    inputs="${SHARD_JSON}" \
    out_dir="${DESIGNS_DIR}" \
    global_prefix="" \
    skip_existing=False \
    n_batches=${N_BATCHES} \
    diffusion_batch_size=${DIFFUSION_BATCH_SIZE} \
    devices_per_node=1 \
    inference_sampler.num_timesteps=${NUM_TIMESTEPS} \
    inference_sampler.use_classifier_free_guidance=${USE_CFG} \
    inference_sampler.cfg_scale=${CFG_SCALE} \
    dump_prediction_metadata_json=True \
    dump_trajectories=False \
    prevalidate_inputs=True

echo "=== Shard ${SHARD_NAME} inference finished: \$(date) ==="
EOFJOB

    JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
    INFER_JOB_IDS+=("${JOB_ID}")
    N_CASES_SHARD=$("${PYTHON}" -c "import json; d=json.load(open('${SHARD_JSON}')); print(len(d))")
    echo "  ${SHARD_NAME}: ${N_CASES_SHARD} cases → job ${JOB_ID}"
done
echo ""

# Build SLURM dependency string: afterok:J1:J2:...
INFER_DEPS=$(IFS=:; echo "afterok:${INFER_JOB_IDS[*]}")

# ── Step 4: dependent analysis job ───────────────────────────────────────────
echo "[4/4] Submitting analysis + plot jobs (deps: ${INFER_DEPS})..."

ANALYSIS_SCRIPT=$(mktemp /tmp/ss_adj_analysis.XXXXXX.sh)
cat > "${ANALYSIS_SCRIPT}" <<EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_adj_analyze
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=${ANALYSIS_TIME}
#SBATCH --output=${RUN_ROOT}/analysis-%j.out
#SBATCH --error=${RUN_ROOT}/analysis-%j.err

set -euo pipefail
echo "=== SS/ADJ analysis started: \$(date) ==="
N_DESIGNS=\$(ls "${DESIGNS_DIR}"/*.cif.gz 2>/dev/null | wc -l)
echo "Designs found: \${N_DESIGNS}"

"${PYTHON}" "${SHELL_DIR}/analyze_ss_adj_conditioning.py" \
    --out-root "${RUN_ROOT}" \
    --adj-threshold 6.0

echo "=== SS/ADJ analysis finished: \$(date) ==="
EOFJOB

ANALYSIS_JOB_ID=$(sbatch --parsable --dependency="${INFER_DEPS}" "${ANALYSIS_SCRIPT}")
echo "  Analysis job: ${ANALYSIS_JOB_ID} (after all inference shards)"

PLOT_SCRIPT=$(mktemp /tmp/ss_adj_plot.XXXXXX.sh)
cat > "${PLOT_SCRIPT}" <<EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_adj_plot
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=${PLOT_TIME}
#SBATCH --output=${RUN_ROOT}/plot-%j.out
#SBATCH --error=${RUN_ROOT}/plot-%j.err

set -euo pipefail
echo "=== SS/ADJ plotting started: \$(date) ==="

EXTRA_ARGS=""
[ -d "${BETA_BINDER_REF}/analysis" ] && EXTRA_ARGS="\${EXTRA_ARGS} --beta-binder-root ${BETA_BINDER_REF}"
[ -d "${LOOP_BINDER_REF}/analysis" ] && EXTRA_ARGS="\${EXTRA_ARGS} --loop-binder-root ${LOOP_BINDER_REF}"

"${PYTHON}" "${SHELL_DIR}/plot_ss_adj_results.py" \
    --benchmark-root "${RUN_ROOT}" \
    --train-root "${TRAIN_ROOT}" \
    --out-dir "${PLOT_DIR}" \
    \${EXTRA_ARGS}

echo "Plots written to: ${PLOT_DIR}"
echo "=== SS/ADJ plotting finished: \$(date) ==="
EOFJOB

PLOT_JOB_ID=$(sbatch --parsable --dependency="afterok:${ANALYSIS_JOB_ID}" "${PLOT_SCRIPT}")
echo "  Plot job:     ${PLOT_JOB_ID} (after analysis)"
echo ""

echo "=== Benchmark pipeline submitted ==="
echo "  Run root     : ${RUN_ROOT}"
echo "  Designs dir  : ${DESIGNS_DIR}"
echo "  Plots dir    : ${PLOT_DIR}"
echo "  Inference jobs: ${INFER_JOB_IDS[*]}"
echo "  Analysis job  : ${ANALYSIS_JOB_ID}"
echo "  Plot job      : ${PLOT_JOB_ID}"
echo ""
echo "Monitor:"
echo "  squeue -u \$(whoami)"
echo "  tail -f ${RUN_ROOT}/shard_00/inference-*.out"
