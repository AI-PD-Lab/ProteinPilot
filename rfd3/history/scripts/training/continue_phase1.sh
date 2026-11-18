#!/bin/bash
# continue_phase1.sh
#
# Continuation job: picks up the latest checkpoint from a previous phase 1
# run and resumes training to max_epochs=581.
#
# Schedule to run automatically after a job finishes:
#   sbatch --dependency=afterany:<JOBID> continue_phase1.sh
#
# Or submit standalone to resume manually.

LOG_DIR=/path/to/work/HARMONIA/logs
FOUNDRY=/path/to/work/HARMONIA/foundry
PYTHON=/path/to/home/miniforge3/envs/atomworks/bin/python3

# Find the latest checkpoint across all phase 1 runs
LATEST_CKPT=$(find ${LOG_DIR}/train/ss-ratio-finetune-phase1 -name "*.ckpt" 2>/dev/null \
              | sort | tail -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found under ${LOG_DIR}/train/ss-ratio-finetune-phase1"
    exit 1
fi

EPOCH=$(basename "$LATEST_CKPT" .ckpt | sed 's/epoch-//')
echo "Latest checkpoint: $LATEST_CKPT  (epoch $EPOCH)"

# How many epochs remain?
MAX_EPOCHS=581
REMAINING=$(( MAX_EPOCHS - 10#$EPOCH ))
if [ "$REMAINING" -le 0 ]; then
    echo "Training already complete (epoch $EPOCH >= $MAX_EPOCHS). Nothing to do."
    exit 0
fi
echo "Epochs remaining: $REMAINING"

GRAD_ACCUM=2
DEVICES_PER_NODE=1

mkdir -p "$LOG_DIR"

JOB_SCRIPT=$(mktemp /tmp/continue_phase1.XXXXXX.sh)
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=ss_ratio_cont
#SBATCH --partition=barracuda
#SBATCH --nodes=1
#SBATCH --nodelist=t38cn053
#SBATCH --reservation=AIPD
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${DEVICES_PER_NODE}
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=${LOG_DIR}/continue_phase1-%j.out
#SBATCH --error=${LOG_DIR}/continue_phase1-%j.err

set -euo pipefail
export HYDRA_FULL_ERROR=1

echo "=== Continuation training started: \$(date) ==="
echo "    Resuming from: ${LATEST_CKPT}"
echo "    Target epoch:  ${MAX_EPOCHS}"
echo ""

cd ${FOUNDRY}

srun --cpu-bind=none ${PYTHON} models/rfd3/src/rfd3/train.py \\
    experiment=ss_ratio_finetune \\
    ckpt_config.path=${LATEST_CKPT} \\
    ckpt_config.reset_optimizer=false \\
    trainer.skip_optimizer_loading=false \\
    "paths/data=cluster_a_local" \\
    paths.log_dir=${LOG_DIR} \\
    trainer.devices_per_node=${DEVICES_PER_NODE} \\
    trainer.num_nodes=1 \\
    trainer.grad_accum_steps=${GRAD_ACCUM} \\
    trainer.max_epochs=${MAX_EPOCHS} \\
    model.lr_scheduler.warmup_steps=0

echo ""
echo "=== Continuation training COMPLETE: \$(date) ==="
EOF

JOB=$(sbatch --parsable "$JOB_SCRIPT")
echo "Submitted continuation → job ${JOB}"
echo "Monitor: tail -f ${LOG_DIR}/continue_phase1-${JOB}.out"
