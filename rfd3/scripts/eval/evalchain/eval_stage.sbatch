#!/bin/bash -l
#SBATCH --job-name=ec_eval
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/work_rfd3/v2/eval/logs/%x-%j.out
# evalchain GPU stage = scripts/slurm/eval_checkpoint.sbatch plus the EMA/RAW weight choice (the harness itself always samples the
# EMA shadow; see derive_ckpt.py). Every harness variable passes through unchanged (CKPT, TAG, SPECS, N, CFGS, SEED, EXTRA, NGPU,
# REPO, OUT_DIR, STOCK, DRY, DRY_PY, DRY_INFER, TMPBASE, ...). Added:
#   WEIGHTS   ema (default: the checkpoint as is) | raw (derive shadow := model into node-local $TMPBASE, run the harness on that)
# The derived file never touches /hnvme; the harness meta.json then records the node-local path *_raw.ckpt (ORIG_CKPT is echoed here
# and recorded in the chain's chain.json). Walltime 2 h = under Helma's PreemptExemptTime (a job is not preempted in its first 2 h),
# measured 66-68 min for CFGS="1.5 2.0" N=32 on 2 H100 (N019 jobs 1209800/1214842/1214843/1216805).
set -uo pipefail
: "${CKPT:?set CKPT}" "${TAG:?set TAG}"
W=/path/to/work_rfd3
REPO=${REPO:-$W/harmonia_study}
EC=${EVALCHAIN_DIR:-$REPO/scripts/eval/evalchain}
WEIGHTS=${WEIGHTS:-ema}
DRY=${DRY:-0}; TMPBASE=${TMPBASE:-/tmp}
export ORIG_CKPT=$CKPT
echo "### evalchain stage $(date +%T) host $(hostname) job ${SLURM_JOB_ID:-none}: TAG=$TAG WEIGHTS=$WEIGHTS ORIG_CKPT=$ORIG_CKPT"
[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 1; }
case "$WEIGHTS" in
  ema) ;;
  raw)
    T=$TMPBASE/evalchain_${SLURM_JOB_ID:-local$$}; mkdir -p "$T/env"
    trap 'rm -rf "$T"' EXIT
    if [ "$DRY" = 1 ]; then
      PY=${DRY_PY:?DRY=1 needs DRY_PY}
    else
      tar xf /path/to/work/rfd3/rfd3_env.tar -C "$T/env" || { echo "FATAL: env extract failed"; exit 1; }
      PY=$T/env/rfd3_ssadj_test/bin/python
    fi
    DER="$T/$(basename "${CKPT%.ckpt}")_raw.ckpt"
    "$PY" -I "$EC/derive_ckpt.py" --in "$CKPT" --out "$DER" --mode raw || { echo "FATAL: raw-weight derivation failed"; exit 2; }
    export CKPT=$DER
    echo "### $(date +%T) harness runs on the derived raw-weight checkpoint $CKPT";;
  *) echo "FATAL: WEIGHTS must be ema or raw (got $WEIGHTS)"; exit 64;;
esac
bash -l "$REPO/scripts/slurm/eval_checkpoint.sbatch"; rc=$?
echo "### evalchain stage done rc=$rc $(date +%T)"
exit $rc
