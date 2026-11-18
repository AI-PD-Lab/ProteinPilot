#!/bin/bash
# Reference launcher for the final Harmonia-RFD3 stage (phase 10 = rollback to the epoch-0694 checkpoint with a minimal
# change set; produced epoch-0698 = Harmonia-RFD3). Single node, N GPUs, no scheduler directives: put your own #SBATCH
# header on top. It needs training data (see foundry/models/rfd3/configs/paths/data/local_ss_adj.yaml) that is not part of
# this repo.
#
#   SEED_CKPT=epoch-0694_phase6.ckpt LOG_DIR=runs/phase10 GPUS=4 MAX_EPOCHS=704 bash train_harmonia_rfd3.sh
#
# max_epochs is an ABSOLUTE target epoch. Re-running the script resumes from the newest epoch-*.ckpt under LOG_DIR
# (ranked by epoch number, not mtime) and only falls back to SEED_CKPT if none exists.
set -euo pipefail

RFD3_DIR=${RFD3_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
FOUNDRY=${RFD3_DIR}/foundry
PYTHON=${PYTHON:-python3}
GPUS=${GPUS:-4}
LOG_DIR=${LOG_DIR:?set LOG_DIR (run output directory)}
SEED_CKPT=${SEED_CKPT:?set SEED_CKPT (epoch-0694_phase6.ckpt)}
MAX_EPOCHS=${MAX_EPOCHS:-704}
PATHS_CONFIG=${PATHS_CONFIG:-local_ss_adj}
BATCH=${DIFFUSION_BATCH_SIZE_TRAIN:-32}
WORKERS=${NUM_WORKERS:-4}
EXPERIMENT=ss_adj_phase10_rollback

export PYTHONPATH="${FOUNDRY}/models/rfd3/src:${FOUNDRY}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1 DEBUG=false TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJECT_ROOT="${FOUNDRY}"

LATEST_CKPT="$(ls ${LOG_DIR}/train/*/*/ckpt/epoch-*.ckpt 2>/dev/null \
    | sed 's#.*/epoch-\([0-9]*\)\.ckpt#\1 &#' | sort -n -k1,1 | tail -1 | cut -d' ' -f2- || true)"
CKPT="${LATEST_CKPT:-${SEED_CKPT}}"
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint ${CKPT} does not exist"; exit 1; }
echo "Starting from ${CKPT}"

cd "${FOUNDRY}"
"${PYTHON}" -m torch.distributed.run --nproc_per_node="${GPUS}" \
    models/rfd3/src/rfd3/train_ss_adj.py \
    experiment=${EXPERIMENT} \
    datasets=ss_adj_design \
    datasets.train.loop=null \
    "paths/data=${PATHS_CONFIG}" \
    ckpt_config.path="${CKPT}" \
    paths.log_dir="${LOG_DIR}" \
    trainer.devices_per_node="${GPUS}" \
    trainer.num_nodes=1 \
    datasets.diffusion_batch_size_train="${BATCH}" \
    dataloader.train.dataloader_params.num_workers="${WORKERS}" \
    trainer.max_epochs="${MAX_EPOCHS}"
