#!/bin/bash -l
#SBATCH --job-name=ec_ross
#SBATCH --partition=preempt
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=/path/to/work_rfd3/v2/eval/logs/%x-%j.out
# evalchain Rossmann stress panel = the N017-N019 ross_*.sbatch generalised to any checkpoint (N019: results/N019-*/ross_h02r1cen2m0.sbatch).
# Old-Rossmann inputs (3 cases: full / ssonly / adjonly rossmann_like, L 120), one cell per seed, CFG 2.0, NB x BS designs per case.
#   CKPT TAG     required                      EXTRA    sampler overrides (the N019 recipe + knobs; word-split, no commas)
#   SEEDS        default "7101 7102" (N017-N019)  CFG 2.0   NB 16   BS 4   STEPS 120   NGPU 2 (one cell per GPU, round robin)
#   WEIGHTS      ema | raw (see derive_ckpt.py)  OUT_DIR  default v2/eval -> $OUT_DIR/$TAG.ross.tar.gz (cells <TAG>_s<seed>_cfg<CFG>/ + meta.json)
#   DRY=1        CPU dry run with DRY_PY + DRY_INFER (fake inference), as in the harness tests
# Measured: N019 job 1214844, 2 cells on 2 GPUs, 20.6 min wall -> 0.7 GPU-h; walltime 45 min (< the 2 h preempt exemption).
set -uo pipefail
export LC_ALL=C
: "${CKPT:?set CKPT}" "${TAG:?set TAG}"
W=/path/to/work_rfd3
REPO=${REPO:-$W/harmonia_study}
EC=${EVALCHAIN_DIR:-$REPO/scripts/eval/evalchain}
OUT_DIR=${OUT_DIR:-$W/v2/eval}
IN=${IN:-$(ls "$REPO"/.agent-research/results/N006-*/scripts/old_rossmann_inputs.json | head -1)}
SEEDS=${SEEDS:-"7101 7102"}; CFG=${CFG:-2.0}; NB=${NB:-16}; BS=${BS:-4}; STEPS=${STEPS:-120}; NGPU=${NGPU:-2}
EXTRA=${EXTRA:-}; WEIGHTS=${WEIGHTS:-ema}
DRY=${DRY:-0}; TMPBASE=${TMPBASE:-/tmp}
ES=/path/to/env/bin/python; [ -x "$ES" ] || ES=python3    # meta/json helper, as in the harness
ORIG_CKPT=$CKPT
[ -f "$CKPT" ] || { echo "FATAL: checkpoint not found: $CKPT"; exit 1; }
[ -f "$IN" ] || { echo "FATAL: inputs not found: $IN"; exit 1; }
mkdir -p "$OUT_DIR/logs"
T=$TMPBASE/evalchain_ross_${SLURM_JOB_ID:-local$$}; mkdir -p "$T/env" "$T/out"
trap 'rc=$?; rm -rf "$T"; [ $rc -eq 0 ] || echo "ABORTED rc=$rc after ${SECONDS}s"; exit $rc' EXIT
echo "### $(date +%T) host $(hostname) job ${SLURM_JOB_ID:-none}: TAG=$TAG WEIGHTS=$WEIGHTS CKPT=$CKPT seeds $SEEDS cfg $CFG"
if [ "$DRY" = 1 ]; then
  PY=${DRY_PY:?DRY=1 needs DRY_PY}; INFER=${DRY_INFER:?DRY=1 needs DRY_INFER}; echo "DRY RUN: fake inference $INFER"
else
  tar xf /path/to/work/rfd3/rfd3_env.tar -C "$T/env" || { echo "FATAL: env extract failed"; exit 1; }
  PY=$T/env/rfd3_ssadj_test/bin/python
  INFER=$REPO/foundry/models/rfd3/src/rfd3/run_inference.py
  export PATH="$T/env/rfd3_ssadj_test/bin:$PATH"
  export PYTHONPATH="$REPO/foundry/src:$REPO/foundry/models/rfd3/src:$REPO/atomworks/src"
  export HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
if [ "$WEIGHTS" = raw ]; then
  DER="$T/$(basename "${CKPT%.ckpt}")_raw.ckpt"
  "$PY" -I "$EC/derive_ckpt.py" --in "$CKPT" --out "$DER" --mode raw || { echo "FATAL: raw-weight derivation failed"; exit 2; }
  CKPT=$DER
elif [ "$WEIGHTS" != ema ]; then echo "FATAL: WEIGHTS must be ema or raw"; exit 64; fi
T0=$(date +%s); gpu=0; CELLS=""
for SEED in $SEEDS; do
  cell=${TAG}_s${SEED}_cfg${CFG}; CELLS="$CELLS $cell"; O=$T/out/$cell; mkdir -p "$O"
  ( cd "$REPO/foundry/models/rfd3" && CUDA_VISIBLE_DEVICES=$gpu "$PY" "$INFER" ckpt_path="$CKPT" inputs="$IN" out_dir="$O" \
      global_prefix="" skip_existing=False n_batches="$NB" diffusion_batch_size="$BS" seed="$SEED" \
      inference_sampler.num_timesteps="$STEPS" inference_sampler.use_classifier_free_guidance=True inference_sampler.cfg_scale="$CFG" \
      $EXTRA dump_prediction_metadata_json=False dump_trajectories=False prevalidate_inputs=True > "$T/$cell.log" 2>&1
    echo "  $cell rc=$? $(( $(date +%s) - T0 ))s" ) &
  gpu=$(( (gpu + 1) % NGPU ))
done
wait
NKEY=$("$ES" -I -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$IN")
NEXP=$((NB * BS * NKEY)); VALID=true; NOTE=""
for cell in $CELLS; do
  n=$(find "$T/out/$cell" -name '*.cif.gz' | wc -l)
  [ "$n" -eq "$NEXP" ] || { VALID=false; NOTE="$NOTE; $cell has $n/$NEXP designs"; }
  if grep -qE "(adj_adapter|ss_adapter).*(reinit|Failed to apply policy)|Failed to apply policy.*(adj_adapter|ss_adapter)" "$T/$cell.log"; then
    VALID=false; NOTE="$NOTE; ADAPTER REINIT in $cell.log"; echo "!!! ADAPTER REINIT in $cell"; fi
  tail -n 2 "$T/$cell.log"
done
"$ES" -I - "$T/out/meta.json" "$TAG" "$ORIG_CKPT" "$CKPT" "$WEIGHTS" "$SEEDS" "$CFG" "$NB" "$BS" "$STEPS" "$EXTRA" "$NEXP" "$VALID" "$NOTE" \
    "${SLURM_JOB_ID:-none}" "$(hostname)" "$(( $(date +%s) - T0 ))" "$IN" <<'PYEOF'
import json, sys
p, tag, orig, ck, w, seeds, cfg, nb, bs, steps, extra, nexp, valid, note, job, host, secs, inp = sys.argv[1:19]
json.dump(dict(tag=tag, ckpt_path=orig, run_ckpt=ck, weights=w, seeds=[int(s) for s in seeds.split()], cfg=float(cfg), n_batches=int(nb),
               batch_size=int(bs), num_timesteps=int(steps), extra_overrides=extra, n_designs_expected_per_cell=int(nexp),
               valid=(valid == "true"), note=note.strip("; "), slurm_job_id=job, host=host, wall_seconds=int(secs), inputs=inp),
          open(p, "w"), indent=1)
PYEOF
_ok=0
for _i in 1 2 3 4; do
  if tar czf "$OUT_DIR/.$TAG.ross.tar.gz.part" -C "$T/out" $(ls "$T/out") && mv "$OUT_DIR/.$TAG.ross.tar.gz.part" "$OUT_DIR/$TAG.ross.tar.gz"; then _ok=1; break; fi
  echo "tar attempt $_i failed (quota?); retrying in 60 s"; rm -f "$OUT_DIR/.$TAG.ross.tar.gz.part"; sleep 60
done
[ "$_ok" = 1 ] || { echo "FATAL: tar failed"; exit 2; }
echo "wrote $OUT_DIR/$TAG.ross.tar.gz valid=$VALID $NOTE $(( $(date +%s) - T0 ))s"
[ "$VALID" = true ] || exit 3
echo "DONE $(date +%T)"
