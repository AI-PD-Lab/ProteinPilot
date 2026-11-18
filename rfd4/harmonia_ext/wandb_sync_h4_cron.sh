#!/bin/bash
# cron (login node helma2): */15 * * * * /path/to/HARMONIA/rfd4/harmonia_ext/wandb_sync_h4_cron.sh
# Needs /path/to/home/logs/harm4_runs.json = {"<jobid>": "<run name>"} (every job of a self-chain under one run name).
export WANDB_DIR=/path/to/work/wandb
exec 9>/tmp/.wandb_h4.lock; flock -n 9 || exit 0
/path/to/miniforge3/envs/triad/bin/python /path/to/HARMONIA/rfd4/harmonia_ext/wandb_sync_h4.py >> /path/to/home/logs/harm4_wandb_cron.log 2>&1
