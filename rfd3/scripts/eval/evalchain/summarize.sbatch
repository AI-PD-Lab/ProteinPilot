#!/bin/bash -l
#SBATCH --job-name=ec_sum
#SBATCH --partition=preempt_cpu
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/path/to/work_rfd3/v2/eval/logs/%x-%j.out
# evalchain final CPU step: scores every tar of the chain (N007 kit score_all.sh -> designs/tm csv in the shared cache), runs the
# N029 score_anatomy.py (--compare against the N019 baselines), the guardrails, designability pooling, the Rossmann panel scorer and
# sacct, and writes <results item>/eval_<TAG>/summary.{md,json}. CPU: scoring is biotite/numpy, 16 procs (~5-10 min); no GPU.
# Submitted with afterany so a failed GPU stage still yields a report (UNVERIFIED / INCOMPLETE rows).
set -uo pipefail
: "${CHAIN_JSON:?set CHAIN_JSON=<results>/eval_<TAG>/chain.json}"
ES=/path/to/env/bin/python
EC=${EVALCHAIN_DIR:-/path/to/work_rfd3/harmonia_study/scripts/eval/evalchain}
echo "### $(date +%T) host $(hostname) job ${SLURM_JOB_ID:-none}: summarize $CHAIN_JSON"
exec "$ES" "$EC/summarize.py" --chain "$CHAIN_JSON" --procs "${PROCS:-16}"
