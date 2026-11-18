#!/bin/bash
# Cron wrapper: push new RFD3 SS/ADJ training metrics to wandb from the LOGIN NODE.
# Compute nodes have no internet; the training job only writes to shared storage,
# so this replays what it wrote. Incremental + idempotent -- safe to run often.
#
# Install (login node with internet, e.g. helma3):
#   crontab -l 2>/dev/null | grep -v wandb_sync_harmonia > /tmp/ct
#   echo "*/15 * * * * /path/to/HARMONIA/rfd3/scripts/wandb_sync_harmonia.sh" >> /tmp/ct
#   crontab /tmp/ct
set -uo pipefail

ROOT=/path/to/HARMONIA/rfd3
LOG=/path/to/checkpoints/wandb_sync_harmonia.log

# cron gets a bare environment -- source conda explicitly.
source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate /path/to/env

export WANDB_SILENT=true
export WANDB_DIR=/path/to/checkpoints/.wandb_tmp
mkdir -p "$WANDB_DIR"

# Serialize: a slow wandb call can push one tick past the next cron firing, and two
# concurrent syncs racing on the same run id fail with "run ID is in use" (and can
# clobber each other's log). -n = skip this tick rather than queue up behind it.
exec 9>"$LOG.lock"
if ! flock -n 9; then
  echo "=== $(date '+%F %T') skipped: previous sync still running ===" >> "$LOG"
  exit 0
fi

{
  echo "=== $(date '+%F %T') on $(hostname) ==="
  timeout 1500 python "$ROOT/scripts/wandb_sync_harmonia.py" 2>&1 | grep -v "^wandb:"
} >> "$LOG" 2>&1

# keep the log from growing without bound
tail -2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
