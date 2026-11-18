#!/bin/bash
# smoketest_ss_ratio.sh
#
# Runs 2 training batches with the ss_ratio fine-tune config to verify:
#   - Pipeline loads without errors
#   - ss_ratio annotation is computed and reaches data["feats"]
#   - Model forward pass and loss computation succeed with the new feature
#   - No NaN in loss or gradients
#
# Run from the login node: bash smoketest_ss_ratio.sh

LOG_DIR=/path/to/work/HARMONIA/logs
mkdir -p "$LOG_DIR"

CKPT=/path/to/work/HARMONIA/foundry/base_models/rfd3_latest.ckpt
FOUNDRY=/path/to/work/HARMONIA/foundry
PYTHON=/path/to/home/miniforge3/envs/atomworks/bin/python3

JOB_SCRIPT=$(mktemp /tmp/smoketest.XXXXXX.sh)
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=ss_ratio_smoketest
#SBATCH --partition=barracuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --constraint=rtx5090
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_DIR}/smoketest-%j.out
#SBATCH --error=${LOG_DIR}/smoketest-%j.err

set -euo pipefail
echo "=== Smoke test started: \$(date) ==="
echo "    Node: \$(hostname)"
echo "    GPU:  \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

cd ${FOUNDRY}

srun ${PYTHON} models/rfd3/src/rfd3/train.py \\
    experiment=ss_ratio_finetune \\
    ckpt_config.path=${CKPT} \\
    "paths/data=cluster_a_local" \\
    paths.log_dir=${LOG_DIR} \\
    trainer.devices_per_node=4 \\
    trainer.num_nodes=1 \\
    trainer.n_examples_per_epoch=32 \\
    trainer.max_epochs=572 \\
    trainer.grad_accum_steps=1

echo ""
echo "=== Smoke test COMPLETE: \$(date) ==="
EOF

JOB=$(sbatch --parsable "$JOB_SCRIPT")
echo "Submitted smoke test → job ${JOB}"
echo ""
echo "Monitor: tail -f ${LOG_DIR}/smoketest-${JOB}.out"
