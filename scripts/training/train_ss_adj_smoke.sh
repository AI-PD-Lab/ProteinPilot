#!/bin/bash
# Phase 0 SS/ADJ smoke run. Submits only to node t38cn053.
set -euo pipefail

ROOT=${ROOT:-/work/gerstung/HARMONIA}
DATA_ROOT=${DATA_ROOT:-/work/gerstung/RFD3_Training/Data_Mirrors}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}
FOUNDRY=${FOUNDRY:-${ROOT}/foundry}
PYTHON=${PYTHON:-/zfs/s01/z04/home/gerstung/miniforge3/envs/atomworks/bin/python3}
CKPT=${CKPT:-${ROOT}/logs/train/ss-ratio-finetune-phase2/2026-06-02_08-45_JOB_399909/ckpt/epoch-0605.ckpt}
PARTITION=${PARTITION:-barracuda}
NODELIST=${NODELIST:-t38cn056}
RESERVATION=${RESERVATION:-AIPD}
TIME=${TIME:-02:00:00}
MEM=${MEM:-12G}

LOOP_METADATA=${DATA_ROOT}/ss_adj_loop/loop_smoke_dataset.csv
if [ ! -s "${LOOP_METADATA}" ]; then
    echo "ERROR: missing loop metadata: ${LOOP_METADATA}"
    echo "Build it with scripts/build_ss_adj_smoke_metadata.py before submitting the smoke run."
    exit 1
fi

mkdir -p "${LOG_DIR}"
JOB_SCRIPT=$(mktemp /tmp/train_ss_adj_smoke.XXXXXX.sh)
cat > "${JOB_SCRIPT}" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_adj_smoke
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=${RESERVATION}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/train_ss_adj_smoke-%j.out
#SBATCH --error=${LOG_DIR}/train_ss_adj_smoke-%j.err

set -euo pipefail
export DEBUG=false
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${FOUNDRY}
srun --cpu-bind=none ${PYTHON} models/rfd3/src/rfd3/train.py \
    experiment=ss_adj_smoke \
    datasets=ss_adj_smoke_design \
    "paths/data=gerstung_local_ss_adj" \
    paths.data.ss_adj_loop_metadata=${LOOP_METADATA} \
    ckpt_config.path=${CKPT} \
    paths.log_dir=${LOG_DIR} \
    trainer.devices_per_node=1 \
    trainer.num_nodes=1
EOFJOB

JOB=$(sbatch --parsable "${JOB_SCRIPT}")
echo "Submitted SS/ADJ smoke job ${JOB} to ${NODELIST}"
echo "Output: ${LOG_DIR}/train_ss_adj_smoke-${JOB}.out"
