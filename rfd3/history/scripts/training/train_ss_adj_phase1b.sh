#!/bin/bash
# Loop-focused adapter training for SS/ADJ conditioning (Phase 1B).
#
# Continues from Phase 1A (adapters only) and introduces loop-contact data
# at a 70/30 standard/loop mixture. Adapters learn loop-mediated adjacency
# patterns while backbone stays frozen.
#
# Usage:
#   NODELIST=t38cn053 bash atomworks/Shell/train_ss_adj_phase1b.sh
#
# Useful overrides:
#   CKPT=/path/to/epoch-0613.ckpt NODELIST=t38cn053 bash train_ss_adj_phase1b.sh

set -euo pipefail

ROOT=${ROOT:-/path/to/work/HARMONIA}
DATA_ROOT=${DATA_ROOT:-/path/to/work/RFD3_Training/Data_Mirrors}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}
FOUNDRY=${FOUNDRY:-${ROOT}/foundry}
PYTHON=${PYTHON:-/path/to/home/miniforge3/envs/atomworks/bin/python3}
CKPT=${CKPT:-$(find "${LOG_DIR}/train/ss-adj-phase1a" -name 'epoch-0613.ckpt' 2>/dev/null | sort -V | tail -1)}
NODELIST=${NODELIST:-t38cn053}
PARTITION=${PARTITION:-barracuda}
RESERVATION=${RESERVATION:-AIPD}
TIME=${TIME:-36:00:00}
AUTO_RESTART_COUNT=${AUTO_RESTART_COUNT:-0}
MAX_AUTO_RESTARTS=${MAX_AUTO_RESTARTS:-3}

case "${NODELIST}" in
    t38cn053) DEVICES_PER_NODE=${DEVICES_PER_NODE:-4}; MEM=${MEM:-100G} ;;
    t38cn056) DEVICES_PER_NODE=1; MEM=${MEM:-64G} ;;
    *) echo "ERROR: NODELIST must be t38cn053 or t38cn056"; exit 1 ;;
esac

if [ -z "${CKPT}" ] || [ ! -s "${CKPT}" ]; then
    if [ -z "${SBATCH_DEPENDENCY:-}" ]; then
        echo "ERROR: no Phase 1A epoch-0613 checkpoint found under ${LOG_DIR}/train/ss-adj-phase1a"; exit 1
    fi
    echo "INFO: Phase 1A checkpoint not yet available; will resolve at job start (dependency: ${SBATCH_DEPENDENCY})"
fi
[ -s "${DATA_ROOT}/ss_adj_loop/loop_train_dataset.csv" ] || { echo "ERROR: missing loop training split"; exit 1; }
mkdir -p "${LOG_DIR}"

JOB_SCRIPT=$(mktemp /tmp/train_ss_adj_phase1b.XXXXXX.sh)
cat > "${JOB_SCRIPT}" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_adj_phase1b
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=${RESERVATION}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${DEVICES_PER_NODE}
#SBATCH --cpus-per-task=$(( DEVICES_PER_NODE * 8 ))
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/train_ss_adj_phase1b-%j.out
#SBATCH --error=${LOG_DIR}/train_ss_adj_phase1b-%j.err

set -euo pipefail

export HYDRA_FULL_ERROR=1
export DEBUG=false
# NVLink is broken on this node; disable all P2P / shared-memory / IB transports
# so NCCL falls back to Ring-allreduce over PCIe, which is stable.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
if [ -n "\${NCCL_PROTO:-}" ]; then export NCCL_PROTO; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -1)
MASTER_PORT=29504

# Resolve checkpoint at job start (handles deferred dependency submissions)
CKPT=\${CKPT:-\$(find "${LOG_DIR}/train/ss-adj-phase1a" -name 'epoch-0613.ckpt' 2>/dev/null | sort -V | tail -1)}
[ -n "\${CKPT}" ] && [ -s "\${CKPT}" ] || { echo "ERROR: Phase 1A epoch-0613 checkpoint not found at job start"; exit 1; }

echo "=== SS/ADJ Phase 1B training started: \$(date) ==="
echo "    Checkpoint: \${CKPT}"
echo "    Nodes     : \$(scontrol show hostnames \$SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "    GPUs      : \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "    Mode      : torchrun ${DEVICES_PER_NODE}-GPU DDP (NVLink disabled, Ring allreduce)"
echo ""

cd ${FOUNDRY}

if ! srun --cpu-bind=none ${PYTHON} -m torch.distributed.run \
    --nproc_per_node=${DEVICES_PER_NODE} \
    --master_addr=\${MASTER_ADDR} \
    --master_port=\${MASTER_PORT} \
    models/rfd3/src/rfd3/train_ss_adj.py \
    experiment=ss_adj_phase1b \
    datasets=ss_adj_loop_focus_design \
    "paths/data=cluster_a_local_ss_adj" \
    ckpt_config.path=\${CKPT} \
    paths.log_dir=${LOG_DIR} \
    trainer.devices_per_node=${DEVICES_PER_NODE} \
    trainer.num_nodes=1; then
    if [ ${AUTO_RESTART_COUNT} -ge ${MAX_AUTO_RESTARTS} ]; then
        echo "    Auto-restart limit reached (${MAX_AUTO_RESTARTS}); stopping retry loop."
        exit 1
    fi
    echo ""
    echo "=== Phase 1B crashed — checking for auto-restart checkpoint: \$(date) ==="
    RESTART_CKPT=\$(find "${LOG_DIR}/train/ss-adj-phase1b" -name '*.ckpt' 2>/dev/null | sort -V | tail -1)
    if [ -n "\${RESTART_CKPT}" ] && [ -s "\${RESTART_CKPT}" ]; then
        echo "    Restarting from: \${RESTART_CKPT}"
        cd ${ROOT}
        AUTO_RESTART_COUNT=$(( AUTO_RESTART_COUNT + 1 )) MAX_AUTO_RESTARTS=${MAX_AUTO_RESTARTS} DEVICES_PER_NODE=${DEVICES_PER_NODE} CKPT="\${RESTART_CKPT}" NODELIST=${NODELIST} bash atomworks/Shell/train_ss_adj_phase1b.sh
    else
        echo "    No checkpoint found — cannot auto-restart."
        exit 1
    fi
else
    echo ""
    echo "=== SS/ADJ Phase 1B training COMPLETE: \$(date) ==="
fi
EOFJOB

DEPENDENCY=${SBATCH_DEPENDENCY:-}
JOB=$(sbatch --parsable ${DEPENDENCY:+--dependency=${DEPENDENCY}} "${JOB_SCRIPT}")
echo "Submitted SS/ADJ Phase 1B job ${JOB} to ${NODELIST}"
echo "  Checkpoint : ${CKPT:-<deferred — resolves at job start>}"
echo "  GPUs       : ${DEVICES_PER_NODE} via torchrun (srun --cpu-bind=none)"
echo "  Monitor    : tail -f ${LOG_DIR}/train_ss_adj_phase1b-${JOB}.out"
