#!/bin/bash
# train_ss_ratio_phase2.sh
#
# Phase 2 fine-tuning for ss_ratio conditioning.
#
# Starts from the latest Phase 1 adapter checkpoint, unfreezes the backbone, and
# trains the full model at a conservative low LR. Uses torchrun by default.
# Single-GPU mode is allowed only when ALLOW_SINGLE_GPU=1 is set explicitly.
#
# Usage:
#   bash atomworks/Shell/train_ss_ratio_phase2.sh
#
# Useful overrides:
#   DEVICES_PER_NODE=4 MAX_EPOCHS=596 bash atomworks/Shell/train_ss_ratio_phase2.sh
#   CKPT=/path/to/epoch-0580.ckpt bash atomworks/Shell/train_ss_ratio_phase2.sh

set -euo pipefail

LOG_DIR=${LOG_DIR:-/work/gerstung/HARMONIA/logs}
FOUNDRY=${FOUNDRY:-/work/gerstung/HARMONIA/foundry}
PYTHON=${PYTHON:-/zfs/s01/z04/home/gerstung/miniforge3/envs/atomworks/bin/python3}

# Multi-GPU by default. Keep this >=2 unless deliberately debugging.
DEVICES_PER_NODE=${DEVICES_PER_NODE:-3}
ALLOW_SINGLE_GPU=${ALLOW_SINGLE_GPU:-0}
NNODES=${NNODES:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
N_EXAMPLES_PER_EPOCH=${N_EXAMPLES_PER_EPOCH:-12000}
DIFFUSION_BATCH_SIZE_TRAIN=${DIFFUSION_BATCH_SIZE_TRAIN:-16}

# Phase 1 ended at epoch 580; checkpoint load increments to 581.
# 586 trains epochs 581-585 = 5 additional epochs.
MAX_EPOCHS=${MAX_EPOCHS:-586}
LR=${LR:-5.0e-6}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
DECAY_STEPS=${DECAY_STEPS:-6000}
DECAY_FACTOR=${DECAY_FACTOR:-0.95}

PARTITION=${PARTITION:-barracuda}
NODELIST=${NODELIST:-t38cn053}
RESERVATION=${RESERVATION:-AIPD}
CPUS_PER_GPU=${CPUS_PER_GPU:-8}
MEM=${MEM:-300G}
TIME=${TIME:-48:00:00}

CKPT=${CKPT:-$(find ${LOG_DIR}/train/ss-ratio-finetune-phase1 -name "*.ckpt" 2>/dev/null | sort -V | tail -1)}
if [ -z "$CKPT" ]; then
    echo "ERROR: no Phase 1 checkpoint found under ${LOG_DIR}/train/ss-ratio-finetune-phase1"
    exit 1
fi

if [ "$DEVICES_PER_NODE" -lt 2 ] && [ "$ALLOW_SINGLE_GPU" != "1" ]; then
    echo "ERROR: DEVICES_PER_NODE=${DEVICES_PER_NODE}; set ALLOW_SINGLE_GPU=1 to deliberately run single-GPU."
    exit 1
fi

mkdir -p "$LOG_DIR"

JOB_SCRIPT=$(mktemp /tmp/train_phase2.XXXXXX.sh)
cat > "$JOB_SCRIPT" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_ratio_phase2
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NNODES}
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=${RESERVATION}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${DEVICES_PER_NODE}
#SBATCH --cpus-per-task=$(( DEVICES_PER_NODE * CPUS_PER_GPU ))
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/train_phase2-%j.out
#SBATCH --error=${LOG_DIR}/train_phase2-%j.err

set -euo pipefail

export HYDRA_FULL_ERROR=1
export DEBUG=false
# Conservative NCCL settings for RTX 5090 / Blackwell DDP stability on this cluster.
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

MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -1)
MASTER_PORT=29501

echo "=== Phase 2 training started: \$(date) ==="
echo "    Checkpoint: ${CKPT}"
echo "    Nodes     : \$(scontrol show hostnames \$SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "    GPUs      : \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "    Mode      : torchrun ${DEVICES_PER_NODE}-GPU"
echo "    LR        : ${LR}"
echo "    Max epoch : ${MAX_EPOCHS}"
echo "    NCCL     : P2P=${NCCL_P2P_DISABLE:-1} IB=${NCCL_IB_DISABLE:-1} SHM=${NCCL_SHM_DISABLE:-1} NVLS=${NCCL_NVLS_ENABLE:-0} CUMEM=${NCCL_CUMEM_ENABLE:-0} ALGO=${NCCL_ALGO:-auto} PROTO=${NCCL_PROTO:-auto}"
echo "    Extra    : ${EXTRA_OVERRIDES:-<none>}"
echo ""

cd ${FOUNDRY}

srun --cpu-bind=none ${PYTHON} -m torch.distributed.run \
    --nproc_per_node=${DEVICES_PER_NODE} \
    --master_addr=\${MASTER_ADDR} \
    --master_port=\${MASTER_PORT} \
    models/rfd3/src/rfd3/train.py \
    experiment=ss_ratio_finetune_phase2 \
    ckpt_config.path=${CKPT} \
    ckpt_config.reset_optimizer=true \
    trainer.skip_optimizer_loading=true \
    "paths/data=gerstung_local" \
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
echo "=== Phase 2 training COMPLETE: \$(date) ==="
EOFJOB

JOB=$(sbatch --parsable "$JOB_SCRIPT")
STEPS_PER_EPOCH=$(( N_EXAMPLES_PER_EPOCH / DEVICES_PER_NODE / GRAD_ACCUM ))
EFFECTIVE_BATCH=$(( DEVICES_PER_NODE * GRAD_ACCUM ))

echo "Submitted Phase 2 training -> job ${JOB}"
echo "  Checkpoint      : ${CKPT}"
echo "  GPUs            : ${DEVICES_PER_NODE} via torchrun"
echo "  Examples/update : ${DEVICES_PER_NODE}x${GRAD_ACCUM}=${EFFECTIVE_BATCH}"
echo "  Steps/epoch     : ~${STEPS_PER_EPOCH}"
echo "  LR              : ${LR}"
echo "  Max epochs      : ${MAX_EPOCHS}"
echo ""
echo "Monitor : tail -f ${LOG_DIR}/train_phase2-${JOB}.out"
echo "Errors  : tail -f ${LOG_DIR}/train_phase2-${JOB}.err"
echo "Output  : ${LOG_DIR}/train/ss-ratio-finetune-phase2/"
