#!/bin/bash
# Adapter-only SS/ADJ training. Submits only to node t38cn053.
set -euo pipefail

ROOT=${ROOT:-/work/gerstung/HARMONIA}
DATA_ROOT=${DATA_ROOT:-/work/gerstung/RFD3_Training/Data_Mirrors}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}
FOUNDRY=${FOUNDRY:-${ROOT}/foundry}
PYTHON=${PYTHON:-/zfs/s01/z04/home/gerstung/miniforge3/envs/atomworks/bin/python3}
CKPT=${CKPT:-${ROOT}/logs/train/ss-ratio-finetune-phase2/2026-06-02_08-45_JOB_399909/ckpt/epoch-0605.ckpt}
DEVICES_PER_NODE=${DEVICES_PER_NODE:-4}
PARTITION=${PARTITION:-barracuda}
NODELIST=t38cn053
RESERVATION=${RESERVATION:-AIPD}
TIME=${TIME:-24:00:00}
MEM=${MEM:-300G}

LOOP_METADATA=${DATA_ROOT}/ss_adj_loop/loop_dataset.csv
if [ ! -s "${LOOP_METADATA}" ]; then
    echo "ERROR: missing loop metadata: ${LOOP_METADATA}"
    exit 1
fi

mkdir -p "${LOG_DIR}"
JOB_SCRIPT=$(mktemp /tmp/train_ss_adj_phase1.XXXXXX.sh)
cat > "${JOB_SCRIPT}" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_adj_phase1
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=${RESERVATION}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${DEVICES_PER_NODE}
#SBATCH --cpus-per-task=$(( DEVICES_PER_NODE * 8 ))
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/train_ss_adj_phase1-%j.out
#SBATCH --error=${LOG_DIR}/train_ss_adj_phase1-%j.err

set -euo pipefail
export DEBUG=false
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_ALGO=Ring
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -1)
MASTER_PORT=29503
cd ${FOUNDRY}
srun --cpu-bind=none ${PYTHON} -m torch.distributed.run \
    --nproc_per_node=${DEVICES_PER_NODE} \
    --master_addr=\${MASTER_ADDR} \
    --master_port=\${MASTER_PORT} \
    models/rfd3/src/rfd3/train.py \
    experiment=ss_adj_phase1 \
    datasets=ss_adj_design \
    "paths/data=gerstung_local_ss_adj" \
    ckpt_config.path=${CKPT} \
    paths.log_dir=${LOG_DIR} \
    trainer.devices_per_node=${DEVICES_PER_NODE} \
    trainer.num_nodes=1
EOFJOB

JOB=$(sbatch --parsable "${JOB_SCRIPT}")
echo "Submitted SS/ADJ Phase 1 job ${JOB} to ${NODELIST}"
echo "Output: ${LOG_DIR}/train_ss_adj_phase1-${JOB}.out"
