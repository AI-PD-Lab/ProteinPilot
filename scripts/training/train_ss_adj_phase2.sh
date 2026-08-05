#!/bin/bash
# Full-model SS/ADJ fine-tuning using the standard/loop training mixture.
set -euo pipefail

ROOT=${ROOT:-/work/gerstung/HARMONIA}
DATA_ROOT=${DATA_ROOT:-/work/gerstung/RFD3_Training/Data_Mirrors}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}
FOUNDRY=${FOUNDRY:-${ROOT}/foundry}
PYTHON=${PYTHON:-/zfs/s01/z04/home/gerstung/miniforge3/envs/atomworks/bin/python3}
NODELIST=${NODELIST:-t38cn053}
PARTITION=${PARTITION:-barracuda}
RESERVATION=${RESERVATION:-AIPD}
TIME=${TIME:-48:00:00}
AUTO_RESTART_COUNT=${AUTO_RESTART_COUNT:-0}
MAX_AUTO_RESTARTS=${MAX_AUTO_RESTARTS:-12}

latest_checkpoint() {
    local directory=$1
    find "${directory}" -name 'epoch-*.ckpt' -printf '%f\t%p\n' 2>/dev/null \
        | sort -V | tail -1 | cut -f2-
}

if [ -z "${CKPT:-}" ]; then
    CKPT=$(latest_checkpoint "${LOG_DIR}/train/ss-adj-phase2")
    if [ -z "${CKPT}" ]; then
        CKPT=$(latest_checkpoint "${LOG_DIR}/train/ss-adj-phase1b")
    fi
fi

case "${NODELIST}" in
    t38cn053) DEVICES_PER_NODE=${DEVICES_PER_NODE:-2}; MEM=${MEM:-180G} ;;
    t38cn056) DEVICES_PER_NODE=1; MEM=${MEM:-64G} ;;
    *) echo "ERROR: NODELIST must be t38cn053 or t38cn056"; exit 1 ;;
esac

[ -n "${CKPT}" ] && [ -s "${CKPT}" ] || {
    echo "ERROR: no Phase 1B/Phase 2 checkpoint found under ${LOG_DIR}/train"
    exit 1
}
[ -s "${DATA_ROOT}/ss_adj_loop/loop_train_dataset.csv" ] || {
    echo "ERROR: missing loop training split"
    exit 1
}

if [[ "${CKPT}" == *"/ss-adj-phase2/"* ]]; then
    RESET_OPTIMIZER=false
    SKIP_OPTIMIZER_LOADING=false
else
    RESET_OPTIMIZER=true
    SKIP_OPTIMIZER_LOADING=true
fi

mkdir -p "${LOG_DIR}"
JOB_SCRIPT=$(mktemp /tmp/train_ss_adj_phase2.XXXXXX.sh)
cat > "${JOB_SCRIPT}" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_adj_phase2
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=${RESERVATION}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${DEVICES_PER_NODE}
#SBATCH --cpus-per-task=$(( DEVICES_PER_NODE * 8 ))
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --signal=B:USR1@300
#SBATCH --output=${LOG_DIR}/train_ss_adj_phase2-%j.out
#SBATCH --error=${LOG_DIR}/train_ss_adj_phase2-%j.err

set -euo pipefail
export HYDRA_FULL_ERROR=1 DEBUG=false
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_ALGO=Ring
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -1)
MASTER_PORT=29505
RESTART_SUBMITTED=0
TRAIN_PID=

submit_restart() {
    local reason=\$1
    [ "\${RESTART_SUBMITTED}" -eq 0 ] || return
    RESTART_SUBMITTED=1
    if [ ${AUTO_RESTART_COUNT} -ge ${MAX_AUTO_RESTARTS} ]; then
        echo "Auto-restart limit reached (${MAX_AUTO_RESTARTS}); not resubmitting."
        return
    fi
    local restart_ckpt
    restart_ckpt=\$(find "${LOG_DIR}/train/ss-adj-phase2" -name 'epoch-*.ckpt' \
        -printf '%f\t%p\n' 2>/dev/null | sort -V | tail -1 | cut -f2-)
    if [ -z "\${restart_ckpt}" ] || [ ! -s "\${restart_ckpt}" ]; then
        restart_ckpt="${CKPT}"
    fi
    echo "Submitting Phase 2 continuation after \${reason} from \${restart_ckpt}"
    cd "${ROOT}"
    AUTO_RESTART_COUNT=$(( AUTO_RESTART_COUNT + 1 )) \
    MAX_AUTO_RESTARTS=${MAX_AUTO_RESTARTS} DEVICES_PER_NODE=${DEVICES_PER_NODE} \
    CKPT="\${restart_ckpt}" NODELIST=${NODELIST} TIME=${TIME} \
    bash atomworks/Shell/train_ss_adj_phase2.sh
}

handle_walltime() {
    echo "Received USR1 wall-time warning at \$(date)"
    submit_restart "wall-time warning"
    if [ -n "\${TRAIN_PID}" ]; then
        kill -TERM "\${TRAIN_PID}" 2>/dev/null || true
    fi
}
trap handle_walltime USR1

echo "=== SS/ADJ Phase 2 training started: \$(date) ==="
echo "    Checkpoint       : ${CKPT}"
echo "    Reset optimizer  : ${RESET_OPTIMIZER}"
echo "    Resume count     : ${AUTO_RESTART_COUNT}/${MAX_AUTO_RESTARTS}"
echo "    Node / GPUs      : ${NODELIST} / ${DEVICES_PER_NODE}"
echo "    Training volume  : 288,000 examples (epochs 622-645)"

cd "${FOUNDRY}"
set +e
srun --cpu-bind=none ${PYTHON} -m torch.distributed.run \
    --nproc_per_node=${DEVICES_PER_NODE} --master_addr=\${MASTER_ADDR} \
    --master_port=\${MASTER_PORT} models/rfd3/src/rfd3/train_ss_adj.py \
    experiment=ss_adj_phase2_revised datasets=ss_adj_design \
    paths/data=gerstung_local_ss_adj ckpt_config.path=${CKPT} \
    ckpt_config.reset_optimizer=${RESET_OPTIMIZER} \
    trainer.skip_optimizer_loading=${SKIP_OPTIMIZER_LOADING} \
    paths.log_dir=${LOG_DIR} trainer.devices_per_node=${DEVICES_PER_NODE} \
    trainer.num_nodes=1 &
TRAIN_PID=\$!
wait "\${TRAIN_PID}"
STATUS=\$?
set -e

if [ "\${STATUS}" -ne 0 ]; then
    submit_restart "exit status \${STATUS}"
    exit "\${STATUS}"
fi
echo "=== SS/ADJ Phase 2 training COMPLETE: \$(date) ==="
EOFJOB

JOB=$(sbatch --parsable "${JOB_SCRIPT}")
echo "Submitted SS/ADJ Phase 2 job ${JOB} to ${NODELIST}"
echo "  Checkpoint : ${CKPT}"
echo "  GPUs       : ${DEVICES_PER_NODE}"
echo "  Optimizer  : reset=${RESET_OPTIMIZER}"
echo "  Monitor    : tail -f ${LOG_DIR}/train_ss_adj_phase2-${JOB}.out"
