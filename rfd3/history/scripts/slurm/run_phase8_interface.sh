#!/bin/bash
#SBATCH --job-name=ss_adj_p8
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
# NOTE: --gres=gpu:4 forces a full-node 128-CPU allocation. SLURM sizes the step
# cgroup as (step_cpus / alloc_cpus) x --mem, so --cpus-per-task=32 gave the 4 ranks
# only 32/128 x 700G x 0.95 = 166 GiB, NOT 700G -- the cause of the repeated
# startup OOM-kills (job footers reporting "166.3 GiB of assigned 700 GiB").
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
# --- autorestart, same machinery as TRIAD's train_production_v2.sh --------------
# Job 743581 hit the 24h wall mid-epoch-701 and simply DIED, leaving training idle
# for 4+ hours until someone noticed. SLURM raises USR1 120 s before the wall, the
# trap requeues this same job id, and the relaunch picks up the newest checkpoint.
# max_epochs is an absolute target, overridden to 725 below (+20 epochs from 705),
# so a requeue that reaches it exits normally instead of looping forever. At ~3.8 h
# per epoch this run is ~76 h, i.e. it WILL cross the 24 h wall 3-4 times -- here the
# requeue is load-bearing, not a safety net.
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append
#SBATCH --output=/path/to/checkpoints/ss_adj_phase8_interface/logs/train-%j.out
#SBATCH --error=/path/to/checkpoints/ss_adj_phase8_interface/logs/train-%j.err

set -euo pipefail

handle_requeue_signal() {
    echo "=== Received $1 at $(date); requeueing job ${SLURM_JOB_ID:-?} from latest checkpoint ==="
    [ -n "${TORCH_PID:-}" ] && kill -TERM "${TORCH_PID}" 2>/dev/null || true
    [ -n "${SLURM_JOB_ID:-}" ] && scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
handle_error() { local rc=$?; echo "=== Launcher failed (status ${rc}); leaving job failed for inspection ==="; exit ${rc}; }
trap 'handle_requeue_signal USR1' USR1
trap 'handle_requeue_signal TERM' TERM
trap 'handle_error' ERR

export HYDRA_FULL_ERROR=1
export DEBUG=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/path/to/HARMONIA/rfd3
FOUNDRY=${ROOT}/foundry
PYTHON=/path/to/env/bin/python3
LOG_DIR=/path/to/work_rfd3/ss_adj_phase8_interface/logs
SEED_CKPT=/path/to/work_rfd3/localattn_ckpts/epoch-0709_phase7.ckpt
EXPERIMENT=ss_adj_phase8_interface

# Newest checkpoint across ALL job directories: each launch writes into its own
# <date>_JOB_<id>/ckpt/, so a requeue must look wider than its own run dir. Ranked by
# EPOCH NUMBER from the filename rather than mtime -- a file copied or touched out of
# order would otherwise be mistaken for the newest, silently rewinding training.
# Glob ANY run-name directory, not a hardcoded one. This script was derived from the
# phase-7 script by search/replace, but hydra's run NAME comes from the experiment
# config and is still "ss-adj-phase7-loop-ab", so checkpoints land under
# train/ss-adj-phase7-loop-ab/ while the glob looked for train/ss-adj-phase8-interface/.
# It matched nothing, silently fell through to the seed, and job 762074 spent 5h20m
# redoing epochs 710-711 that were already on disk.
LATEST_CKPT="$(ls ${LOG_DIR}/train/*/*/ckpt/epoch-*.ckpt 2>/dev/null \
    | sed 's#.*/epoch-\([0-9]*\)\.ckpt#\1 &#' | sort -n -k1,1 | tail -1 | cut -d' ' -f2- || true)"
if [ -n "${LATEST_CKPT}" ]; then
    CKPT="${LATEST_CKPT}"
    echo "    Resuming from newest run checkpoint: ${CKPT}"
else
    CKPT="${SEED_CKPT}"
    echo "    ############################################################"
    echo "    ## WARNING: no run checkpoint found under ${LOG_DIR}/train/"
    echo "    ## Falling back to the SEED: ${CKPT}"
    echo "    ## On a continuation this is WRONG and silently redoes epochs."
    echo "    ############################################################"
fi
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint ${CKPT} does not exist"; exit 1; }

source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate /path/to/env

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
MASTER_PORT=29531

echo "=== SS/ADJ (${EXPERIMENT}) started: $(date) | job ${SLURM_JOB_ID} | restart #${SLURM_RESTART_COUNT:-0} ==="

# Echo the RESOLVED values of the knobs this phase turns. Hydra resolves at launch, so
# a config edited after sbatch silently does nothing -- which is exactly what happened
# to crop_size 384 and the motif SS mask in job 756694: they were edited at 13:47, the
# job started at 12:35, and the discrepancy only surfaced two hours later. Printing
# them here puts a stale config in the first lines of the log instead.
export FOUNDRY_DIR="${FOUNDRY}"
"${PYTHON}" - <<'PYCFG' || echo "    (config echo failed; non-fatal)"
import os
from hydra import compose, initialize_config_dir
os.environ.setdefault("PROJECT_ROOT", os.environ["FOUNDRY_DIR"])
with initialize_config_dir(config_dir=os.path.join(os.environ["FOUNDRY_DIR"], "models/rfd3/configs"), version_base=None):
    c = compose(config_name="train", overrides=[
        "experiment=ss_adj_phase8_interface", "datasets=ss_adj_design",
        "datasets.train.loop=null", "paths/data=cluster_b_local_ss_adj"])
    d, l = c.datasets, c.trainer.loss.adjacency_contact_loss
    tc = d.global_transform_args.train_conditions
    tot = sum(v.frequency for v in tc.values())
    t0 = next(iter(d.train.standard.sub_datasets.values())).dataset.transform
    print("=== RESOLVED PHASE-8 KNOBS ===")
    print(f"    crop_size={d.crop_size}  max_atoms_in_crop={d.max_atoms_in_crop}")
    print(f"    steric_weight={l.get('steric_weight')}  floor={l.get('steric_floor')}")
    print(f"    negative_weight={l.get('negative_weight')}  detach_motif={l.get('detach_motif')}")
    print(f"    loop_adj={d.ss_adj_loop_adj}  typed_blocks={d.ss_adj_typed_blocks}")
    print(f"    motif_region_mask_prob={t0.get('ss_adj_motif_region_mask_prob')}"
          f"  motif_ss_only_mask_prob={t0.get('ss_adj_motif_ss_only_mask_prob')}"
          f"  cross_chain_seed_prob={t0.get('ss_adj_cross_chain_seed_prob')}")
    print(f"    ppi={100*tc.ppi.frequency/tot:.1f}%  allow_na_target={tc.ppi.get('allow_na_target')}"
          f"  uncond={100*tc.unconditional.frequency/tot:.1f}%  island={100*tc.island.frequency/tot:.1f}%")
    print("==============================")
PYCFG
echo "    Node      : ${MASTER_ADDR}"

cd "${FOUNDRY}"
export PROJECT_ROOT=${FOUNDRY}

# Backgrounded + wait, NOT foreground: a foreground srun blocks the shell from
# running the trap, so the USR1 would be swallowed and the job would die at the wall
# exactly as 743581 did.
srun --cpu-bind=none "${PYTHON}" -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    models/rfd3/src/rfd3/train_ss_adj.py \
    experiment=${EXPERIMENT} \
    datasets=ss_adj_design \
    datasets.train.loop=null \
    "paths/data=cluster_b_local_ss_adj" \
    ckpt_config.path=${CKPT} \
    paths.log_dir=${LOG_DIR} \
    trainer.devices_per_node=4 \
    trainer.num_nodes=1 \
    datasets.diffusion_batch_size_train=32 \
    dataloader.train.dataloader_params.num_workers=4 \
    trainer.max_epochs=725 &
TORCH_PID=$!
wait "${TORCH_PID}"

echo "=== SS/ADJ (${EXPERIMENT}) COMPLETE: $(date) ==="
