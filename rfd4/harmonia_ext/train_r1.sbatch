#!/bin/bash -l
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --job-name=harm4_r1
#SBATCH --exclude=h14-21,h12-12,h13-21,h13-10
#SBATCH --output=/path/to/home/logs/harm4_%j.out
# HARMONIA stage R1 fine-tune of the released RFD4-Proteina (SS + block-adjacency conditioning). Derived from TRIAD-R train_s2.sbatch (self-chain, resume, watchdog).
# Placement: GPU (the 178 M trunk on [b,256,256,384] pair tensors is >= 2x any CPU run), 1-4 H100 = ranks of one DDP job (srun, one task per GPU),
# 32 CPU cores per GPU for the RING/RFD4 DataLoader workers (12 workers per rank; 1.6 samples/s/core measured, the GPU needs <= ~7 rows/s).
# Usage (env vars, all optional except ARM):
#   ARM=A|B|C|Z  STEPS=2000  GPUS=4  BATCH=8 (rows per GPU per micro-batch)  ACCUM=<64/(GPUS*BATCH)>  NW=12  TAG=  EXTRA="<hydra overrides>"
#   4-GPU arm run   : ARM=B STEPS=4000 sbatch triadj_ext/s2/train_s2.sbatch        (4 x 6 h = 24 GPU-h per job; the job self-chains until DONE)
#   1-GPU smoke     : ARM=B GPUS=1 STEPS=100 ACCUM=2 EXTRA="opt.val_every_n_epochs=1" sbatch --gres=gpu:h100:1 --ntasks-per-node=1 --time=01:00:00 triadj_ext/s2/train_s2.sbatch
#   2-GPU smoke     : same with GPUS=2 and --gres=gpu:h100:2 --ntasks-per-node=2   (DDP, NCCL in the container, unused-parameter check)
#   RESUME smoke   : the same command with a larger STEPS (same ARM/TAG/GPUS) restores the newest last*.ckpt of $DEST and trains up to the new STEPS.
#   STEPS <= 300 selects the tag automatically: `smoke` for 1 GPU, `smoke<N>g` for N > 1 GPUs (own $DEST per ARM x GPU count: a DDP smoke is a fresh run, never
#   a resume of the 1-GPU one, and never touches a real arm run). TAG=<x> overrides; RERUN=<suffix> appends to the name (fresh $DEST for a repeat);
#   FORCE=1 deletes DONE/ABORT of $DEST first (not inherited by the chained successor).
# DONE is written only AFTER the final checkpoints (newest last*.ckpt, newest chk_*-ema.ckpt) are verified in $DEST with their node-local size; otherwise ABORT + rc 6.
# DONE marker = "steps=<n> gpus=<g> job=<id>": a run asking for STEPS <= n exits at once, a LARGER STEPS continues (legacy empty marker = steps 0).
# Emits to the SLURM log: "S2 job ..." header, "S2 optim ...", "H4 step=.." every 10 steps (columns: harmonia_ext/callbacks.py), Lightning validation
# logs every opt.val_every_n_epochs epochs, "S2 DONE" / "S2 exit rc=". Checkpoints: node-local $OUT/checkpoints, mirrored every 5 min to
# $DEST/checkpoints (last*.ckpt and the live chk_*.ckpt: newest only; every chk_*-ema.ckpt, for offline scoring). Resume: the newest $DEST/checkpoints/last*.ckpt is restored first.
# SELF-CHAIN: the successor job (--dependency=afterany) is submitted at the START (a walltime kill never runs the end of the script); it exits at once
# when $DEST/DONE exists and refuses when $DEST/ABORT exists (written by the watchdog / a fast crash). A normally finishing job cancels its successor.
# THROUGHPUT WATCHDOG (scoped to THIS job id): fewer than 2 "H4 step" lines 15 min after the start, or (after the smoke fixed S2_SMOKE_SEC_STEP) a
# median of the last five windows with sec_step > 2.2 x S2_SMOKE_SEC_STEP -> `scancel $SLURM_JOB_ID` + $DEST/ABORT (the chain stops). Metrics: wandb_sync_s2*.sh on a login node.
set -uo pipefail
JOB=${SLURM_JOB_ID:-x}
ARM=${ARM:-R1}   # run family: R1full (full fine-tune, harmonia_r1_full) | R1lora (LoRA control, harmonia_r1); EXP selects the experiment yaml
STEPS=${STEPS:-2000}
GPUS=${GPUS:-4}
BATCH=${BATCH:-8}
ACCUM=${ACCUM:-$(( 64 / (GPUS * BATCH) > 0 ? 64 / (GPUS * BATCH) : 1 ))}   # 64 rows per optimizer step (4 GPUs x 8 x 2)
NW=${NW:-12}
EXTRA=${EXTRA:-}
TAG=${TAG:-$([ "$STEPS" -le 300 ] && { [ "$GPUS" = 1 ] && echo smoke || echo "smoke${GPUS}g"; })}
RERUN=${RERUN:-}
FORCE=${FORCE:-0}
CHAIN_N=${CHAIN_N:-0}
case "${TAG:-}" in smoke*) MAX_CHAIN=${MAX_CHAIN:-0};; *) MAX_CHAIN=${MAX_CHAIN:-8};; esac   # smokes never chain
WD=${S2_WD:-/path/to/work}                      # S2_* overrides exist for the dry-run test (tests/test_s2_sbatch.py)
SIF=$WD/rfd4-cu13.sif
FORK=${S2_FORK:-/path/to/HARMONIA/rfd4}
SELF=$FORK/harmonia_ext/train_r1.sbatch
NAME=${ARM}${TAG:+_$TAG}${RERUN:+_$RERUN}
OUT=${S2_TMP:-/tmp/${USER}}/harm4_${NAME}_${JOB}
DEST=$WD/harmonia_rfd4/runs/$NAME
BUNDLE=$WD/harmonia_rfd4/base
LOG=${S2_LOG:-/path/to/home/logs/harm4_${JOB}.out}
EPOCHS=$(( (STEPS + 99) / 100 ))
mkdir -p "$OUT/checkpoints" "$OUT/hydra" "$DEST/checkpoints"
T_START=$(date +%s)

echo "S2 job id=$JOB arm=$ARM tag=${TAG:-none}${RERUN:+ rerun=$RERUN} dest=$NAME gpus=$GPUS batch=$BATCH accum=$ACCUM rows_per_step=$((GPUS*BATCH*ACCUM)) steps=$STEPS epochs=$EPOCHS nw=$NW node=$(hostname) chain=$CHAIN_N"
TL=$(squeue -h -j "$JOB" -o %l 2>/dev/null); TL=${TL:-06:00:00}
case "$TL" in *-*) TL_H=$(( 24 * 10#${TL%%-*} + 10#$(echo "${TL#*-}" | cut -d: -f1) ));; *) TL_H=$(( 10#$(echo "$TL" | awk -F: '{ if (NF==3) print $1; else print 0 }') ));; esac
echo "S2 placement: GPU ($GPUS x H100, DDP via srun), 32 cores per GPU for $NW loader workers per rank; time limit $TL => <= $(( GPUS * TL_H )) GPU-h for this job (>25 GPU-h per arm needs approval before submit)"
[ "${SLURM_NTASKS:-$GPUS}" = "$GPUS" ] || { echo "S2 error: $SLURM_NTASKS tasks but GPUS=$GPUS: submit with --ntasks-per-node=$GPUS --gres=gpu:h100:$GPUS"; exit 2; }
if [ "$FORCE" = 1 ]; then echo "S2 FORCE=1: removing DONE / ABORT markers of $DEST"; rm -f "$DEST/DONE" "$DEST/ABORT"; fi
if [ -f "$DEST/DONE" ]; then
  REC=$(sed -n 's/.*steps=\([0-9][0-9]*\).*/\1/p' "$DEST/DONE" | head -1); REC=${REC:-0}
  if [ "$STEPS" -le "$REC" ]; then echo "S2 DONE marker present in $DEST ($(cat "$DEST/DONE")): requested steps=$STEPS <= recorded $REC, nothing to do (FORCE=1 / RERUN=<suffix> to repeat)"; exit 0; fi
  echo "S2 DONE marker of $DEST records steps=$REC < requested $STEPS: continuing from its newest checkpoint"
fi
[ -f "$DEST/ABORT" ] && { echo "S2 ABORT marker present in $DEST ($(cat "$DEST/ABORT")): not (re)starting; FORCE=1 (or rm it) after the cause is fixed"; exit 3; }
# home inode pre-flight: every job and every sync state file needs home inodes; abort (and stop the chain) near the hard limit
HOME_FILES=$(quota 2>/dev/null | awk '/hpchome/ {getline; gsub(/\*/, ""); n = 0; for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) v[n++] = $i; if (n >= 4) print v[3]; exit}')
HOME_MAX=${S2_HOME_FILES_MAX:-538000}
if [ -z "$HOME_FILES" ]; then echo "S2 note: home file count unavailable (quota output not parsed)"
else
  echo "S2 home files: $HOME_FILES (abort above $HOME_MAX)"
  if [ "$HOME_FILES" -gt "$HOME_MAX" ]; then echo "home files $HOME_FILES > $HOME_MAX (job $JOB)" > "$DEST/ABORT"; echo "S2 ABORT: home inode quota nearly exhausted ($HOME_FILES files): free files first, then FORCE=1"; exit 5; fi
fi
[ -f "$BUNDLE/model.safetensors" ] || { echo "S2 error: no expanded bundle $BUNDLE (python -m harmonia_ext.make_bundle --dst /w/harmonia_rfd4/base)"; exit 2; }

# ---- resume: newest last*.ckpt of the previous job(s) (train.py:fetch_last_ckpt picks the highest version in $OUT/checkpoints) -----------------------
# A checkpoint in $DEST that cannot be restored is FATAL: falling through would start from the base bundle and later overwrite the saved lineage.
NEWEST=$(ls -t "$DEST"/checkpoints/last*.ckpt 2>/dev/null | head -1)
if [ -n "$NEWEST" ]; then
  RESTORED="$OUT/checkpoints/$(basename "$NEWEST")"
  if cp "$NEWEST" "$RESTORED" && [ -s "$RESTORED" ] && [ "$(stat -c %s "$NEWEST")" = "$(stat -c %s "$RESTORED")" ]; then
    echo "S2 resume from $NEWEST ($(stat -c %s "$RESTORED") bytes)"
  else
    echo "S2 ABORT: restoring $NEWEST failed; NOT starting from the base bundle (the saved lineage in $DEST stays untouched)"
    echo "restore of $NEWEST failed (job $JOB)" > "$DEST/ABORT"; exit 4
  fi
fi

# ---- checkpoint mirror: only files that are not being written (mtime > 90 s), copy + rename, then prune older last*.ckpt in $DEST ----------------------
sync_out() {
  local f base newest newest_chk
  newest=$(ls -t "$OUT"/checkpoints/last*.ckpt 2>/dev/null | head -1)
  newest_chk=$(ls -t "$OUT"/checkpoints/chk_*.ckpt 2>/dev/null | grep -v -e '-ema.ckpt$' | head -1)     # live (non-EMA) weights: newest only
  for f in "$newest" "$newest_chk" "$OUT"/checkpoints/chk_*-ema.ckpt "$OUT"/checkpoints/exp_config_*.json; do
    [ -n "$f" ] && [ -f "$f" ] || continue
    base=$(basename "$f")
    [ "$(( $(date +%s) - $(stat -c %Y "$f") ))" -ge "${1:-90}" ] || continue
    if [ ! -f "$DEST/checkpoints/$base" ] || [ "$(stat -c %s "$f")" != "$(stat -c %s "$DEST/checkpoints/$base")" ] || [ "$f" -nt "$DEST/checkpoints/$base" ]; then
      cp "$f" "$DEST/checkpoints/.tmp_$base" && mv -f "$DEST/checkpoints/.tmp_$base" "$DEST/checkpoints/$base" || { echo "S2 sync: copy of $base to $DEST failed"; rm -f "$DEST/checkpoints/.tmp_$base"; continue; }
      case "$base" in
        last*.ckpt) for o in "$DEST"/checkpoints/last*.ckpt; do [ "$(basename "$o")" != "$base" ] && rm -f "$o"; done;;
        chk_*-ema.ckpt) ;;
        chk_*.ckpt) for o in "$DEST"/checkpoints/chk_*.ckpt; do case "$o" in *-ema.ckpt) ;; *) [ "$(basename "$o")" != "$base" ] && rm -f "$o";; esac; done;;
      esac
    fi
  done
}
( while sleep 300; do sync_out 90; done ) &
SYNC=$!
FINAL_WAIT=0
FINAL_AGE=${S2_FINAL_AGE:-20}
final_sync() {   # age >= 20 s: a file killed mid-write (walltime) is never mirrored over the previous good copy; after a clean exit wait for the last writes
  kill $SYNC 2>/dev/null
  sleep "$FINAL_WAIT"; sync_out "$FINAL_AGE"
  echo "S2 saved: $(ls "$DEST"/checkpoints | tr '\n' ' ')"
}
trap final_sync EXIT

# ---- final persistence: DONE / "the successor continues" are only claimed after the newest checkpoints are verified in $DEST ------------------------
verify_persisted() {   # newest last*.ckpt (and the newest chk_*-ema.ckpt) of the node-local OUT must exist in $DEST with the node-local size
  local f b
  for f in "$(ls -t "$OUT"/checkpoints/last*.ckpt 2>/dev/null | head -1)" "$(ls -t "$OUT"/checkpoints/chk_*-ema.ckpt 2>/dev/null | head -1)"; do
    [ -n "$f" ] || continue
    b=$(basename "$f")
    if [ ! -f "$DEST/checkpoints/$b" ] || [ "$(stat -c %s "$f")" != "$(stat -c %s "$DEST/checkpoints/$b")" ]; then
      echo "S2 persist check FAILED: $b is missing in $DEST/checkpoints or differs in size from $f"; return 1
    fi
  done
  return 0
}
persist_final() {      # stop the periodic mirror, let the last writes age, copy, verify; 0 = everything node-local is persisted
  kill $SYNC 2>/dev/null
  sleep "${S2_FINAL_WAIT:-25}"
  sync_out "$FINAL_AGE"
  verify_persisted
}

# ---- successor job (self-chain) submitted NOW so that a walltime kill still continues the run ---------------------------------------------------------
NEXT=""
if [ "$CHAIN_N" -lt "$MAX_CHAIN" ]; then
  NEXT=$(sbatch --parsable --dependency=afterany:"$JOB" --export=ALL,CHAIN_N=$((CHAIN_N + 1)),FORCE=0 --gres=gpu:$([ "${GTYPE:-h100}" = any ] && echo "" || echo "${GTYPE:-h100}:")$GPUS --ntasks-per-node=$GPUS --time="$TL" "$SELF")
  echo "S2 successor job $NEXT submitted (afterany:$JOB, chain $((CHAIN_N + 1))/$MAX_CHAIN); add every job id to the N001 notebook item"
else
  echo "S2 MAX_CHAIN=$MAX_CHAIN reached: no successor"
fi
cancel_next() { [ -n "$NEXT" ] && scancel "$NEXT" 2>/dev/null; NEXT=""; }

# ---- throughput watchdog, scoped to this job id ----------------------------------------------------------------------------------------------------
(
  sleep "${WATCH_FIRST:-900}"
  while true; do
    n=$(grep -c '^H4 step=' "$LOG" 2>/dev/null); n=${n:-0}
    if [ "$n" -lt 2 ]; then echo "S2 WATCHDOG: $n step lines $(( ($(date +%s) - T_START) / 60 )) min after the start, aborting job $JOB"; echo "watchdog: $n step lines at 15 min (job $JOB)" > "$DEST/ABORT"; scancel ${NEXT:+$NEXT} "$JOB"; exit; fi
    if [ -n "${S2_SMOKE_SEC_STEP:-}" ] && [ "$n" -ge 5 ]; then      # needs 5 windows: a single startup outlier must not decide (Codex round 6 #7)
      # median (upper) of the last FIVE windows (50 steps): one or two slow windows (checkpoint / validation neighbour, a hiccup) never aborts a healthy chain
      last=$(grep '^H4 step=' "$LOG" | tail -5 | sed -n 's/.* sec_step=\([0-9.e+-]*\) .*/\1/p' | sort -g | awk '{v[NR]=$1} END{if (NR) print v[int(NR/2)+1]}')
      if [ -n "$last" ] && awk -v a="$last" -v b="$S2_SMOKE_SEC_STEP" 'BEGIN{exit !(a > 2.2*b)}'; then
        echo "S2 WATCHDOG: median sec_step=$last > 2.2 x $S2_SMOKE_SEC_STEP (smoke), aborting job $JOB"; echo "watchdog: median sec_step=$last (job $JOB)" > "$DEST/ABORT"; scancel ${NEXT:+$NEXT} "$JOB"; exit; fi
    fi
    sleep 300
  done ) &
WATCH=$!

# ---- binds (GPU nodes do not mount /home/atuin, measured 2026-10-01): /mir = PDB mirror tar + index of the RingDataset CifStore; /w = data + bundles; lmdb via pylibs_lmdb
BINDS="--bind $WD:/w --bind $FORK:/fork --bind $OUT:/out --bind /path/to/work_rfd3:/mir:ro"
MIRROR_ENV=""
if [ -d /path/to/data/data ]; then
  BINDS="$BINDS --bind /path/to/data/data:/pdbdata:ro"; MIRROR_ENV="CCD_MIRROR_PATH=/pdbdata/ccd_mirror"
else
  echo "S2 note: /home/atuin not visible on $(hostname); /pdbdata NOT bound (RING lmdb, manifests and the PDB tar come from /w and /mir)"
fi
[ -d "$WD/pylibs_lmdb/lmdb" ] || { echo "S2 error: no lmdb under $WD/pylibs_lmdb (needed by the RING reader in the dataloader workers)"; exit 2; }

# cwd = node-local OUT: Proteina.store_dir defaults to ./runs/<run> (would land in the repo on /home); all code and configs are found through PYTHONPATH / PROJECT_ROOT
srun --kill-on-bad-exit=1 apptainer exec --nv $BINDS --pwd /out "$SIF" \
  env CLUSTER=local DATA_PATH=/w HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH=/fork/src:/fork:/w/pylibs_lmdb OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=disabled WANDB_DIR=/out $MIRROR_ENV \
      TORCH_COMPILE_CACHE_BLOB=/w/harmonia_rfd4/runs/$NAME/compile.blob RING_ROWS_CACHE_DIR=/tmp \
  python -m harmonia_ext.train_entry experiment=${EXP:-harmonia_r1} hf_ckpt_path=/w/harmonia_rfd4/base out_dir=/out \
    hydra.run.dir=/out/hydra hydra.output_subdir=null log.log_wandb=false hardware.ngpus_per_node_=$GPUS hardware.nnodes_=1 \
    dataloaders.batch_size=$BATCH dataloaders.num_workers=$NW opt.accumulate_grad_batches=$ACCUM opt.steps_per_epoch=100 opt.max_epochs=$EPOCHS \
    $EXTRA 2>&1 | grep --line-buffered -v -E "UserWarning|Warning:|MIRROR_PATH|VAR_NAME|^ +\([123]\)"
RC=${PIPESTATUS[0]}
kill $WATCH 2>/dev/null
RUN_S=$(( $(date +%s) - T_START ))
echo "S2 exit rc=$RC after ${RUN_S}s"

# ---- outcome ---------------------------------------------------------------------------------------------------------------------------------------
# The final checkpoints are persisted and VERIFIED first; only then is DONE written and the successor cancelled. A persistence failure writes ABORT, exits 6,
# keeps the node-local checkpoints and does NOT cancel the successor (it refuses to start while ABORT exists, so it can never train from the base bundle).
persist_failed() {
  echo "S2 ABORT: final checkpoint persistence FAILED; node-local checkpoints are kept in $OUT/checkpoints on $(hostname) (copy them to $DEST/checkpoints by hand, then FORCE=1)"
  echo "persistence failed, node-local copy in $(hostname):$OUT/checkpoints (job $JOB)" > "$DEST/ABORT"
  FINAL_WAIT=0; exit 6
}
if [ "$RC" = 0 ] && ls "$OUT"/checkpoints/last*.ckpt >/dev/null 2>&1; then          # trainer.fit returned: max_epochs reached
  persist_final || persist_failed
  echo "S2 DONE"; echo "steps=$STEPS gpus=$GPUS job=$JOB" > "$DEST/DONE"; cancel_next; FINAL_WAIT=0; exit 0
fi
if [ -f "$DEST/ABORT" ]; then echo "S2 aborted by the watchdog: no successor"; cancel_next; exit "${RC:-1}"; fi
if [ "$RUN_S" -lt "${S2_MIN_RUN:-600}" ]; then echo "S2 job died after ${RUN_S}s (rc=$RC): successor cancelled, fix the cause, then rm $DEST/ABORT"; echo "crash rc=$RC after ${RUN_S}s (job $JOB)" > "$DEST/ABORT"; cancel_next; exit "${RC:-1}"; fi
persist_final || persist_failed                                                       # the successor resumes from $DEST: its newest last*.ckpt must really be there
echo "S2 stopped with rc=$RC after ${RUN_S}s: the successor ${NEXT:-none} continues from the newest last*.ckpt"
exit "$RC"
