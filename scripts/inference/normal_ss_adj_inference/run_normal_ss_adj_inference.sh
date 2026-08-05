#!/bin/bash
# Normal RFdiffusion3 SS/ADJ inference launcher.
# Usage:
#   INPUT_JSON=/path/to/inputs.json OUT_ROOT=/path/to/output bash run_normal_ss_adj_inference.sh
# If INPUT_JSON is unset, a small example JSON plus tensors is generated in OUT_ROOT/input.
set -euo pipefail

ROOT=${ROOT:-/work/gerstung/HARMONIA}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}
FOUNDRY=${FOUNDRY:-${ROOT}/foundry}
PYTHON=${PYTHON:-/zfs/s01/z04/home/gerstung/miniforge3/envs/atomworks/bin/python3}

NODELIST=${NODELIST:-t38cn056}
PARTITION=${PARTITION:-barracuda}
RESERVATION=${RESERVATION:-AIPD}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-96G}
TIME=${TIME:-12:00:00}
DEVICES_PER_NODE=${DEVICES_PER_NODE:-1}

RUN_ID=${RUN_ID:-normal_ss_adj_$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-${LOG_DIR}/ss_adj_inference/${RUN_ID}}
OUT_DESIGNS=${OUT_DESIGNS:-${OUT_ROOT}/designs}
N_BATCHES=${N_BATCHES:-1}
DIFFUSION_BATCH_SIZE=${DIFFUSION_BATCH_SIZE:-4}
NUM_TIMESTEPS=${NUM_TIMESTEPS:-120}
USE_CFG=${USE_CFG:-True}
CFG_SCALE=${CFG_SCALE:-2.0}
SUBMIT=${SUBMIT:-1}

latest_checkpoint() {
    find "${LOG_DIR}/train/ss-adj-phase2" -name 'epoch-*.ckpt' -printf '%f\t%p\n' 2>/dev/null \
        | sort -V | tail -1 | cut -f2-
}

CKPT=${CKPT:-$(latest_checkpoint)}
[ -n "${CKPT}" ] && [ -s "${CKPT}" ] || { echo "ERROR: no checkpoint found; set CKPT=/path/to/model.ckpt"; exit 1; }

mkdir -p "${OUT_ROOT}"
if [ -z "${INPUT_JSON:-}" ]; then
    INPUT_DIR=${OUT_ROOT}/input
    INPUT_JSON=$("${PYTHON}" "${ROOT}/atomworks/Shell/normal_ss_adj_inference/make_example_inputs.py" --out-dir "${INPUT_DIR}")
fi

cat <<EOF
Normal SS/ADJ inference
  Checkpoint : ${CKPT}
  Input JSON : ${INPUT_JSON}
  Output dir : ${OUT_DESIGNS}
  Designs    : $((N_BATCHES * DIFFUSION_BATCH_SIZE)) (${N_BATCHES} x ${DIFFUSION_BATCH_SIZE}) per input key
  Node       : ${NODELIST}
  CFG        : ${USE_CFG} scale=${CFG_SCALE}
EOF

if [ "${SUBMIT}" = "0" ]; then
    echo "SUBMIT=0: prepared inputs only."
    exit 0
fi

JOB_SCRIPT=$(mktemp /tmp/normal_ss_adj_infer.XXXXXX.sh)
cat > "${JOB_SCRIPT}" <<EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_adj_normal
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

cd "${FOUNDRY}"
srun --cpu-bind=none "${PYTHON}" models/rfd3/src/rfd3/run_inference.py \
    ckpt_path="${CKPT}" \
    inputs="${INPUT_JSON}" \
    out_dir="${OUT_DESIGNS}" \
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
EOFJOB

JOB_ID=$(sbatch --parsable "${JOB_SCRIPT}")
echo "Submitted normal SS/ADJ inference job ${JOB_ID}"
echo "stdout: ${OUT_ROOT}/inference-${JOB_ID}.out"
echo "stderr: ${OUT_ROOT}/inference-${JOB_ID}.err"
