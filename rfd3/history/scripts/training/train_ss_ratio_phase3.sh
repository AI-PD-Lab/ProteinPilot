#!/bin/bash
# train_ss_ratio_phase3.sh
#
# Phase 3 fine-tuning for ss_ratio conditioning.
#
# Continues from the Phase 2 epoch-585 checkpoint with:
#   - Stratified data reweighting (cluster_a_local_ss parquet)
#   - Optimizer state preserved (momentum buffers intact, LR ~3.9e-6)
#   - No warmup (already converged)
#
# Usage:
#   bash atomworks/Shell/train_ss_ratio_phase3.sh
#
# Useful overrides:
#   DEVICES_PER_NODE=4 MAX_EPOCHS=606 bash atomworks/Shell/train_ss_ratio_phase3.sh
#   CKPT=/path/to/epoch-0585.ckpt bash atomworks/Shell/train_ss_ratio_phase3.sh

set -euo pipefail

LOG_DIR=${LOG_DIR:-/path/to/work/HARMONIA/logs}
FOUNDRY=${FOUNDRY:-/path/to/work/HARMONIA/foundry}
PYTHON=${PYTHON:-/path/to/home/miniforge3/envs/atomworks/bin/python3}

DEVICES_PER_NODE=${DEVICES_PER_NODE:-4}
ALLOW_SINGLE_GPU=${ALLOW_SINGLE_GPU:-0}
NNODES=${NNODES:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
N_EXAMPLES_PER_EPOCH=${N_EXAMPLES_PER_EPOCH:-12000}
DIFFUSION_BATCH_SIZE_TRAIN=${DIFFUSION_BATCH_SIZE_TRAIN:-16}

# Phase 2 ended at epoch 585; 596 trains 10 more epochs (586–595).
MAX_EPOCHS=${MAX_EPOCHS:-596}
# Base LR kept for scheduler config; actual LR is loaded from checkpoint optimizer state.
LR=${LR:-5.0e-6}
WARMUP_STEPS=${WARMUP_STEPS:-0}      # no warmup — already converged
DECAY_STEPS=${DECAY_STEPS:-6000}
DECAY_FACTOR=${DECAY_FACTOR:-0.95}

PARTITION=${PARTITION:-barracuda}
NODELIST=${NODELIST:-t38cn053}
RESERVATION=${RESERVATION:-AIPD}
CPUS_PER_GPU=${CPUS_PER_GPU:-8}
MEM=${MEM:-300G}
TIME=${TIME:-48:00:00}

CKPT=${CKPT:-$(find ${LOG_DIR}/train/ss-ratio-finetune-phase2 -name "*.ckpt" 2>/dev/null | sort -V | tail -1)}
if [ -z "$CKPT" ]; then
    echo "ERROR: no Phase 2 checkpoint found under ${LOG_DIR}/train/ss-ratio-finetune-phase2"
    exit 1
fi

if [ "$DEVICES_PER_NODE" -lt 2 ] && [ "$ALLOW_SINGLE_GPU" != "1" ]; then
    echo "ERROR: DEVICES_PER_NODE=${DEVICES_PER_NODE}; set ALLOW_SINGLE_GPU=1 to deliberately run single-GPU."
    exit 1
fi

mkdir -p "$LOG_DIR"

JOB_SCRIPT=$(mktemp /tmp/train_phase3.XXXXXX.sh)
cat > "$JOB_SCRIPT" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_ratio_phase3
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NNODES}
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=${RESERVATION}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${DEVICES_PER_NODE}
#SBATCH --cpus-per-task=$(( DEVICES_PER_NODE * CPUS_PER_GPU ))
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/train_phase3-%j.out
#SBATCH --error=${LOG_DIR}/train_phase3-%j.err

set -euo pipefail

export HYDRA_FULL_ERROR=1
export DEBUG=false
# Conservative NCCL settings for RTX 5090 / Blackwell DDP stability.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ALGO=${NCCL_ALGO:-Ring}
if [ -n "${NCCL_PROTO:-}" ]; then export NCCL_PROTO; fi
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
# Prevent OOM crashes from memory fragmentation during backward pass.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -1)
MASTER_PORT=29501

echo "=== Phase 3 training started: \$(date) ==="
echo "    Checkpoint: ${CKPT}"
echo "    Data      : cluster_a_local_ss (stratified ss_ratio reweighting)"
echo "    Nodes     : \$(scontrol show hostnames \$SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "    GPUs      : \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "    Mode      : torchrun ${DEVICES_PER_NODE}-GPU"
echo "    LR (base) : ${LR}  (actual LR loaded from checkpoint optimizer state)"
echo "    Max epoch : ${MAX_EPOCHS}"
echo "    NCCL      : ALGO=${NCCL_ALGO:-Ring}"
echo "    Extra     : ${EXTRA_OVERRIDES:-<none>}"
echo ""

cd ${FOUNDRY}

srun --cpu-bind=none ${PYTHON} -m torch.distributed.run \
    --nproc_per_node=${DEVICES_PER_NODE} \
    --master_addr=\${MASTER_ADDR} \
    --master_port=\${MASTER_PORT} \
    models/rfd3/src/rfd3/train.py \
    experiment=ss_ratio_finetune_phase2 \
    ckpt_config.path=${CKPT} \
    ckpt_config.reset_optimizer=false \
    trainer.skip_optimizer_loading=false \
    "paths/data=cluster_a_local_ss" \
    paths.log_dir=${LOG_DIR} \
    trainer.devices_per_node=${DEVICES_PER_NODE} \
    trainer.num_nodes=${NNODES} \
    trainer.grad_accum_steps=${GRAD_ACCUM} \
    trainer.n_examples_per_epoch=${N_EXAMPLES_PER_EPOCH} \
    trainer.max_epochs=${MAX_EPOCHS} \
    trainer.checkpoint_every_n_epochs=1 \
    model.optimizer.lr=${LR} \
    model.lr_scheduler.base_lr=${LR} \
    model.lr_scheduler.warmup_steps=${WARMUP_STEPS} \
    model.lr_scheduler.decay_steps=${DECAY_STEPS} \
    model.lr_scheduler.decay_factor=${DECAY_FACTOR} \
    datasets.diffusion_batch_size_train=${DIFFUSION_BATCH_SIZE_TRAIN} \
    ${EXTRA_OVERRIDES:-}

echo ""
echo "=== Phase 3 training COMPLETE: \$(date) ==="
EOFJOB

JOB=$(sbatch --parsable "$JOB_SCRIPT")
STEPS_PER_EPOCH=$(( N_EXAMPLES_PER_EPOCH / DEVICES_PER_NODE / GRAD_ACCUM ))
EFFECTIVE_BATCH=$(( DEVICES_PER_NODE * GRAD_ACCUM ))

echo "Submitted Phase 3 training -> job ${JOB}"
echo "  Checkpoint      : ${CKPT}"
echo "  GPUs            : ${DEVICES_PER_NODE} via torchrun"
echo "  Examples/update : ${DEVICES_PER_NODE}x${GRAD_ACCUM}=${EFFECTIVE_BATCH}"
echo "  Steps/epoch     : ~${STEPS_PER_EPOCH}"
echo "  LR (base)       : ${LR}  (resumed from checkpoint)"
echo "  Max epochs      : ${MAX_EPOCHS}  (epochs 586–595)"
echo "  Data            : cluster_a_local_ss (stratified ss_ratio)"
echo ""
echo "Monitor : tail -f ${LOG_DIR}/train_phase3-${JOB}.out"
echo "Errors  : tail -f ${LOG_DIR}/train_phase3-${JOB}.err"
echo "Output  : ${LOG_DIR}/train/ss-ratio-finetune-phase2/"
