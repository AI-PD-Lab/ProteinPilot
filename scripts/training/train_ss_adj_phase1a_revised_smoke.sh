#!/bin/bash
# One-epoch smoke test for the revised leakage-free Phase 1A configuration.
set -euo pipefail
ROOT=${ROOT:-/work/gerstung/HARMONIA}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}
FOUNDRY=${FOUNDRY:-${ROOT}/foundry}
PYTHON=${PYTHON:-/zfs/s01/z04/home/gerstung/miniforge3/envs/atomworks/bin/python3}
CKPT=${CKPT:-${ROOT}/logs/train/ss-ratio-finetune-phase2/2026-06-02_08-45_JOB_399909/ckpt/epoch-0605.ckpt}
NODELIST=t38cn056
mkdir -p "${LOG_DIR}"
JOB_SCRIPT=$(mktemp /tmp/train_ss_adj_phase1a_revised_smoke.XXXXXX.sh)
cat > "${JOB_SCRIPT}" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_adj_p1a_smoke
#SBATCH --partition=barracuda
#SBATCH --nodes=1
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=AIPD
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/train_ss_adj_phase1a_revised_smoke-%j.out
#SBATCH --error=${LOG_DIR}/train_ss_adj_phase1a_revised_smoke-%j.err
set -euo pipefail
export DEBUG=false HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_ALGO=Ring
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${FOUNDRY}
${PYTHON} models/rfd3/src/rfd3/train.py experiment=ss_adj_phase1a datasets=ss_adj_standard_design \
  "paths/data=gerstung_local_ss_adj" ckpt_config.path=${CKPT} paths.log_dir=${LOG_DIR} \
  trainer.devices_per_node=1 trainer.num_nodes=1 trainer.max_epochs=607 \
  trainer.n_examples_per_epoch=64 trainer.validate_every_n_epochs=1 trainer.limit_val_batches=2
EOFJOB
JOB=$(sbatch --parsable "${JOB_SCRIPT}")
echo "${JOB}"
