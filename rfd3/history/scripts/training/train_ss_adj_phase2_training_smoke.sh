#!/bin/bash
# Short Phase 2 gate: full-model DDP forward/backward on real training data.
set -euo pipefail
ROOT=${ROOT:-/path/to/work/HARMONIA}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}
FOUNDRY=${FOUNDRY:-${ROOT}/foundry}
PYTHON=${PYTHON:-/path/to/home/miniforge3/envs/atomworks/bin/python3}
CKPT=${CKPT:-$(find "${LOG_DIR}/train/ss-adj-phase1b" -name 'epoch-0621.ckpt' | sort -V | tail -1)}
NODELIST=${NODELIST:-t38cn053}
DEVICES_PER_NODE=${DEVICES_PER_NODE:-2}
LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES:-8}
[ -s "${CKPT}" ] || { echo "ERROR: missing checkpoint: ${CKPT}"; exit 1; }
JOB_SCRIPT=$(mktemp /tmp/train_ss_adj_phase2_smoke.XXXXXX.sh)
cat > "${JOB_SCRIPT}" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_adj_p2_smoke
#SBATCH --partition=barracuda
#SBATCH --nodes=1
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=AIPD
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${DEVICES_PER_NODE}
#SBATCH --cpus-per-task=$(( DEVICES_PER_NODE * 8 ))
#SBATCH --mem=180G
#SBATCH --time=00:45:00
#SBATCH --output=${LOG_DIR}/train_ss_adj_phase2_smoke-%j.out
#SBATCH --error=${LOG_DIR}/train_ss_adj_phase2_smoke-%j.err
set -euo pipefail
export DEBUG=false HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_ALGO=Ring
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${FOUNDRY}"
MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -1)
srun --cpu-bind=none ${PYTHON} -m torch.distributed.run \
    --nproc_per_node=${DEVICES_PER_NODE} --master_addr=\${MASTER_ADDR} --master_port=29517 \
    models/rfd3/src/rfd3/train_ss_adj.py experiment=ss_adj_phase2_revised \
    datasets=ss_adj_design paths/data=cluster_a_local_ss_adj ckpt_config.path=${CKPT} \
    ckpt_config.reset_optimizer=true trainer.skip_optimizer_loading=true \
    paths.log_dir=${LOG_DIR} trainer.devices_per_node=${DEVICES_PER_NODE} trainer.num_nodes=1 \
    trainer.max_epochs=623 trainer.n_examples_per_epoch=16 \
    +trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES} trainer.validate_every_n_epochs=99999 \
    trainer.checkpoint_every_n_epochs=99999
EOFJOB
JOB=$(sbatch --parsable "${JOB_SCRIPT}")
echo "Submitted Phase 2 training smoke ${JOB} (${LIMIT_TRAIN_BATCHES} batches)"
