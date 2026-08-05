#!/bin/bash
# train_ss_ratio_phase1.sh
#
# Phase 1 fine-tuning: ss_ratio Fourier conditioning adapter.
#
# GPU modes (set DEVICES_PER_NODE):
#   1 GPU  → single-process, no DDP
#   2 GPU  → torchrun DDP (2 fresh subprocesses; avoids SLURM per-task GPU binding issues)
#
# torchrun is used instead of srun multi-task because:
#   - srun with ntasks-per-node=2 gives CUDA_VISIBLE_DEVICES=0,1 to BOTH tasks,
#     causing CUDA context initialisation on device 0 for all ranks before
#     Lightning calls set_device(local_rank). On Blackwell this triggers
#     destroyEvent crashes and cross-GPU NCCL memory leaks.
#   - torchrun spawns fresh subprocesses: set_device(LOCAL_RANK) is the very
#     first CUDA call, so the context is initialised on the correct device.
#
# Effective batch: batch_size(16) × DEVICES_PER_NODE × GRAD_ACCUM
# Optimizer steps: n_examples(2400) / batch_size(16) / DEVICES_PER_NODE / GRAD_ACCUM × max_epochs
#
# Run: bash train_ss_ratio_phase1.sh

LOG_DIR=/work/gerstung/HARMONIA/logs
CKPT=/work/gerstung/HARMONIA/foundry/base_models/rfd3_latest.ckpt
FOUNDRY=/work/gerstung/HARMONIA/foundry
PYTHON=/zfs/s01/z04/home/gerstung/miniforge3/envs/atomworks/bin/python3

DEVICES_PER_NODE=2   # set to 1 for single-GPU fallback
NNODES=1
# grad_accum must be set so that optimizer steps = n_examples/batch_size/DEVICES_PER_NODE/GRAD_ACCUM × epochs
# Target: ~75 steps/epoch → 825 total → 2.4% warmup at warmup_steps=20
#   1 GPU:  accum=2  → 2400/16/1/2 = 75 steps/epoch ✓
#   2 GPUs: accum=1  → 2400/16/2/1 = 75 steps/epoch ✓  (DDP splits dataset across GPUs)
GRAD_ACCUM=1

mkdir -p "$LOG_DIR"

JOB_SCRIPT=$(mktemp /tmp/train_phase1.XXXXXX.sh)
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=ss_ratio_phase1
#SBATCH --partition=barracuda
#SBATCH --nodes=${NNODES}
#SBATCH --nodelist=t38cn053
#SBATCH --reservation=AIPD
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${DEVICES_PER_NODE}
#SBATCH --cpus-per-task=$(( DEVICES_PER_NODE * 8 ))
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=${LOG_DIR}/train_phase1-%j.out
#SBATCH --error=${LOG_DIR}/train_phase1-%j.err

set -euo pipefail

export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1   # RTX 5090 Blackwell: P2P over PCIe causes illegal memory access

MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -1)
MASTER_PORT=29500

echo "=== Phase 1 training started: \$(date) ==="
echo "    Nodes  : \$(scontrol show hostnames \$SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "    GPUs   : \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort | uniq -c)"
echo "    Mode   : torchrun ${DEVICES_PER_NODE}-GPU DDP"
echo ""

cd ${FOUNDRY}

srun --cpu-bind=none ${PYTHON} -m torch.distributed.run \\
    --nproc_per_node=${DEVICES_PER_NODE} \\
    --master_addr=\${MASTER_ADDR} \\
    --master_port=\${MASTER_PORT} \\
    models/rfd3/src/rfd3/train.py \\
    experiment=ss_ratio_finetune \\
    ckpt_config.path=${CKPT} \\
    "paths/data=gerstung_local" \\
    paths.log_dir=${LOG_DIR} \\
    trainer.devices_per_node=${DEVICES_PER_NODE} \\
    trainer.num_nodes=${NNODES} \\
    trainer.grad_accum_steps=${GRAD_ACCUM} \\
    trainer.max_epochs=581 \\
    model.lr_scheduler.warmup_steps=20

echo ""
echo "=== Phase 1 training COMPLETE: \$(date) ==="
EOF

JOB=$(sbatch --parsable "$JOB_SCRIPT")
echo "Submitted Phase 1 training → job ${JOB}"
echo "  ${DEVICES_PER_NODE} GPU(s) via torchrun, effective batch = 16×${DEVICES_PER_NODE}×${GRAD_ACCUM}=$(( 16 * DEVICES_PER_NODE * GRAD_ACCUM ))"
echo "  ~$(( 2400 / 16 / DEVICES_PER_NODE / GRAD_ACCUM )) optimizer steps/epoch × 11 epochs ≈ $(( 2400 / 16 / DEVICES_PER_NODE / GRAD_ACCUM * 11 )) total"
echo ""
echo "Monitor : tail -f ${LOG_DIR}/train_phase1-${JOB}.out"
echo "Errors  : tail -f ${LOG_DIR}/train_phase1-${JOB}.err"
echo "Output  : ${LOG_DIR}/train/ss-ratio-finetune-phase1/"
