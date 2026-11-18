#!/bin/bash
# Submit the panel eval (eval_r1.sbatch, w 1.0 / 2.0, n=16/arm + unconditional) for every EMA checkpoint of a run as it appears.
#   harmonia_ext/chain_evals.sh <run name> [steps...]      e.g. chain_evals.sh R1full_s10k 2000 4000 6000 8000 10000
set -uo pipefail
RUN=$1; shift; STEPS=${*:-"2000 4000 6000 8000 10000"}
D=/path/to/work/harmonia_rfd4/runs/$RUN/checkpoints
cd /path/to/HARMONIA/rfd4
for S in $STEPS; do
  until C=$(ls $D/chk_*_step_$(printf %012d $S)-ema.ckpt 2>/dev/null | head -1) && [ -n "$C" ]; do sleep 600; done
  sleep 120
  J=$(TAG=${RUN}_s$S CKPT=$C UNCOND=1 sbatch --parsable harmonia_ext/eval_r1.sbatch)
  echo "$(date +%F_%T) step=$S ckpt=$C eval_job=$J"
done
echo CHAIN_DONE
