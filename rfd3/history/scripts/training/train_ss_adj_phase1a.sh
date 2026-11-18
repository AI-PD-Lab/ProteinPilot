#!/bin/bash
# Standard-corpus adapter warm-up for SS/ADJ conditioning (Phase 1A).
#
# Trains ONLY the ss_adapter and adj_adapter weights (all backbone frozen).
# Uses the standard PDB corpus (no loop data) to teach the model what
# secondary-structure and adjacency tensors look like before loop contacts
# are introduced.
#
# Usage:
#   NODELIST=t38cn053 bash atomworks/Shell/train_ss_adj_phase1a.sh
#
# Useful overrides:
#   CKPT=/path/to/custom.ckpt NODELIST=t38cn053 bash train_ss_adj_phase1a.sh

set -euo pipefail

ROOT=${ROOT:-/path/to/work/HARMONIA}
DATA_ROOT=${DATA_ROOT:-/path/to/work/RFD3_Training/Data_Mirrors}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}
FOUNDRY=${FOUNDRY:-${ROOT}/foundry}
PYTHON=${PYTHON:-/path/to/home/miniforge3/envs/atomworks/bin/python3}
CKPT=${CKPT:-${ROOT}/logs/train/ss-ratio-finetune-phase2/2026-06-02_08-45_JOB_399909/ckpt/epoch-0605.ckpt}
NODELIST=${NODELIST:-t38cn053}
PARTITION=${PARTITION:-barracuda}
RESERVATION=${RESERVATION:-AIPD}
TIME=${TIME:-36:00:00}

case "${NODELIST}" in
    t38cn053) DEVICES_PER_NODE=${DEVICES_PER_NODE:-4}; MEM=${MEM:-300G} ;;
    t38cn056) DEVICES_PER_NODE=1; MEM=${MEM:-64G} ;;
    *) echo "ERROR: NODELIST must be t38cn053 or t38cn056"; exit 1 ;;
esac

[ -s "${CKPT}" ] || { echo "ERROR: missing checkpoint: ${CKPT}"; exit 1; }
[ -s "${DATA_ROOT}/ss_adj_loop/loop_val_dataset.csv" ] || { echo "ERROR: missing loop validation split"; exit 1; }
mkdir -p "${LOG_DIR}"

JOB_SCRIPT=$(mktemp /tmp/train_ss_adj_phase1a.XXXXXX.sh)
cat > "${JOB_SCRIPT}" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_adj_phase1a
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=${RESERVATION}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${DEVICES_PER_NODE}
#SBATCH --cpus-per-task=$(( DEVICES_PER_NODE * 8 ))
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/train_ss_adj_phase1a-%j.out
#SBATCH --error=${LOG_DIR}/train_ss_adj_phase1a-%j.err

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
# Prevent OOM crashes from memory fragmentation during backward pass.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -1)
MASTER_PORT=29503

echo "=== SS/ADJ Phase 1A training started: \$(date) ==="
echo "    Checkpoint: ${CKPT}"
echo "    Nodes     : \$(scontrol show hostnames \$SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "    GPUs      : \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "    Mode      : torchrun ${DEVICES_PER_NODE}-GPU DDP (NVLink disabled, Ring allreduce)"
echo ""

cd ${FOUNDRY}

srun --cpu-bind=none ${PYTHON} -m torch.distributed.run \
    --nproc_per_node=${DEVICES_PER_NODE} \
    --master_addr=\${MASTER_ADDR} \
    --master_port=\${MASTER_PORT} \
    models/rfd3/src/rfd3/train_ss_adj.py \
    experiment=ss_adj_phase1a \
    datasets=ss_adj_standard_design \
    "paths/data=cluster_a_local_ss_adj" \
    ckpt_config.path=${CKPT} \
    paths.log_dir=${LOG_DIR} \
    trainer.devices_per_node=${DEVICES_PER_NODE} \
    trainer.num_nodes=1

echo ""
echo "=== SS/ADJ Phase 1A training COMPLETE: \$(date) ==="
EOFJOB

JOB=$(sbatch --parsable "${JOB_SCRIPT}")
echo "Submitted SS/ADJ Phase 1A job ${JOB} to ${NODELIST}"
echo "  Checkpoint : ${CKPT}"
echo "  GPUs       : ${DEVICES_PER_NODE} via torchrun (srun --cpu-bind=none)"
echo "  Monitor    : tail -f ${LOG_DIR}/train_ss_adj_phase1a-${JOB}.out"
