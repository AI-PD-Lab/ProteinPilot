#!/bin/bash -l
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=harm4_eval
#SBATCH --exclude=h14-21,h12-12,h13-21,h13-10
#SBATCH --output=/path/to/home/logs/harm4_eval_%j.out
# HARMONIA-RFD4 eval: generation on the RFD3 natural 5-target manifest (1 H100: sampling is GPU-bound, 180 trunk passes per design) -> tar
# in $WD/harmonia_rfd4/eval/<TAG>.tar.gz. Preempt-safe by being short and idempotent (an existing tar is not regenerated).
# Usage: TAG=<tag> (CKPT=<ema ckpt> | BUNDLE=/w/harmonia_rfd4/base) [CFGS=1.0,2.0] [N=16] [HOT=0] [UNCOND=1] [MANIFEST=/w/...json] [CATH={"target":"code"}] sbatch harmonia_ext/eval_r1.sbatch
set -uo pipefail
WD=/path/to/work; FORK=/path/to/HARMONIA/rfd4
TAG=${TAG:?}; CFGS=${CFGS:-1.0,2.0}; N=${N:-16}; HOT=${HOT:-0}; COLD=${COLD:-0}
OUT=$WD/harmonia_rfd4/eval; mkdir -p $OUT
[ -s $OUT/$TAG.tar.gz ] && { echo "exists: $OUT/$TAG.tar.gz"; exit 0; }
SRC=$([ -n "${CKPT:-}" ] && echo "--ckpt ${CKPT/#$WD//w}" || echo "--bundle ${BUNDLE:?set CKPT or BUNDLE}")
echo "H4EVAL tag=$TAG src=$SRC cfgs=$CFGS n=$N hot=$HOT cold=$COLD node=$(hostname) placement=1 GPU (H100/H200; sampling GPU-bound)"
apptainer exec --nv --bind $WD:/w --bind /path/to/work_rfd3:/mir:ro --bind $FORK:/fork --pwd /tmp $WD/rfd4-cu13.sif \
  env CLUSTER=local DATA_PATH=/w HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/fork/src:/fork:/w/pylibs_lmdb \
  python -m harmonia_ext.generate $SRC --tag $TAG --out-tar /w/harmonia_rfd4/eval/$TAG.tar.gz --cfgs $CFGS --n $N --keys ${KEYS:-all} --hot $HOT --cold $COLD ${UNCOND:+--uncond} ${MANIFEST:+--manifest $MANIFEST} ${CATH:+--cath "$CATH"} ${REF_BUNDLE:+--ref-bundle $REF_BUNDLE} ${REF_CKPT:+--ref-ckpt ${REF_CKPT/#$WD//w}} \
  2>&1 | grep --line-buffered -v -E "Warning|warn|VAR_NAME|MIRROR|^ +\([123]\)"
RC=${PIPESTATUS[0]}
echo "H4EVAL exit rc=$RC"
exit $RC                                                 # red-team N001-2 #5: a failed generation must fail the job
