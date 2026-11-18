#!/bin/bash
# evalchain.sh - ONE command that evaluates a HARMONIA-v2 checkpoint end to end the way N028 / N030 / N033 pre-register it, so
# M1, S1, K1, S2 (and G1) checkpoints are judged identically by either live session instance. Submits a SLURM chain (afterok):
#   eval      specs_v2, CFGS "1.5 2.0", N=32, seed 20261014 (= N019 seed 1), N019 recipe        scripts/eval/evalchain/eval_stage.sbatch
#   [eval_sX  --full: seeds 20261015 + 20261016 in parallel, afterok eval]                        (same)
#   heldout   held-out beta pack (N029 probe), CFG 1.5, N=16                                      (same, SPECS = the pack)
#   ross      old-Rossmann stress panel, seeds 7101/7102, CFG 2.0                                 ross_panel.sbatch
#   design    designability of the eval tar, CFG 1.5 + 2.0, goal arms, 8 per cell, K=8           scripts/eval/designability.sbatch
#   summarize CPU (preempt_cpu, afterany of everything): score_all.sh + score_anatomy.py --compare N019 baselines, guardrails,
#             designability pooling, Rossmann scorer, sacct GPU-h -> <results item>/eval_<TAG>/summary.{md,json}    summarize.sbatch
# All GPU jobs: preempt, 2 x H100, 16 CPUs, walltime <= 2 h (= Helma's PreemptExemptTime: never preempted; the harness is not resumable).
#
# usage: evalchain.sh CKPT TAG --item N0xx [--weights ema|raw] [--seed S] [--full] [--extra "hydra overrides"] [--recipe n019|none]
#                     [--cfgs "1.5 2.0"] [--n 32] [--skip heldout,ross,design] [--after JOBID[,JOBID]] [--max-gpuh 6] [--yes]
#                     [--dry-run] [--overwrite]
#   CKPT        checkpoint (EMA + raw weights inside; --weights picks which one the sampler uses, see derive_ckpt.py)
#   TAG         eval tag = tar name v2/eval/<TAG>.tar.gz; derived names <TAG>_s<seed>, <TAG>_heldout, <TAG>.ross.tar.gz,
#               <TAG>_cfg1.5_2.0.designability.json. Convention: <stage>_e<epoch>[_raw], e.g. v2_M1_e0597 / v2_M1_e0597_raw
#   --item      notebook item that owns the eval (results/<N0xx>-*/eval_<TAG>/ receives chain.json, jobs.json, summary.*)
#   --extra     appended to the recipe for every sampling job (K1 arms: e.g. "inference_sampler.gamma_0=0.3"); no commas
#   --after     extra job ids the summarize step waits for (afterany), e.g. the baseline held-out job 1257548
#   --max-gpuh  refuse if the EXPECTED GPU-h (measured N019 stage times) exceed this (default 6) unless --yes; the walltime cap
#               (gpus x walltime, the approval formula of CLAUDE.md) is printed too and must stay < 25
#   --dry-run   print every sbatch line (job ids as $J_* placeholders) and the estimate; submit nothing, write nothing
# Records: <item>/eval_<TAG>/{chain.json,jobs.json,submit.log}; logs v2/eval/logs/ec_<TAG>_<stage>-<job>.out
set -euo pipefail
W=/path/to/work_rfd3
REPO=${REPO:-$W/harmonia_study}
EC=$REPO/scripts/eval/evalchain
OUT_DIR=${OUT_DIR:-$W/v2/eval}
CACHE=${CACHE:-$OUT_DIR/cache}
LOGS=$OUT_DIR/logs
EXCL=${EXCLUDE:-h14-21,h13-10,h13-21}
PROBE=$(ls -d "$REPO"/.agent-research/results/N029-*/probe | head -1)
HELD_SPECS=$PROBE/specs_beta_heldout.tar.gz
GPUS=2
# measured stage wall hours at 2 x H100 (sacct of the N019 confirmation: eval 1209800/1214842/1214843/1216805 = 3977-4086 s for
# CFGS "1.5 2.0" N=32; designability 1214841 = 464 s for 120 designs at one CFG; Rossmann 1214844 = 1236 s for 2 cells;
# held-out: 256 designs scaled from the 1376-design eval + ~3 min env restore)
EVAL_H=1.12; HELD_H=0.26; ROSS_H=0.34; DES_H=0.30
EVAL_WT=02:00:00; HELD_WT=01:00:00; ROSS_WT=00:45:00; DES_WT=00:45:00; SUM_WT=01:30:00

usage() { sed -n '2,24p' "$0"; exit "${1:-64}"; }
[ $# -ge 2 ] || usage
CKPT=$1; TAG=$2; shift 2
ITEM=""; WEIGHTS=ema; SEED=20261014; FULL=0; EXTRA=""; RECIPE=n019; CFGS="1.5 2.0"; N=32; SKIP=""; AFTER=""; MAXGPUH=6; YES=0; DRY=0; OVER=0
while [ $# -gt 0 ]; do
  case "$1" in
    --item) ITEM=$2; shift 2;;        --weights) WEIGHTS=$2; shift 2;;   --seed) SEED=$2; shift 2;;
    --full) FULL=1; shift;;           --extra) EXTRA=$2; shift 2;;       --recipe) RECIPE=$2; shift 2;;
    --cfgs) CFGS=$2; shift 2;;        --n) N=$2; shift 2;;               --skip) SKIP=$2; shift 2;;
    --after) AFTER=$2; shift 2;;      --max-gpuh) MAXGPUH=$2; shift 2;;  --yes) YES=1; shift;;
    --dry-run) DRY=1; shift;;         --overwrite) OVER=1; shift;;       -h|--help) usage 0;;
    *) echo "unknown option $1"; usage;;
  esac
done
die() { echo "FATAL: $*" >&2; exit 1; }
[ -f "$CKPT" ] || die "checkpoint not found: $CKPT"
CKPT=$(readlink -f "$CKPT")
[[ "$TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || die "TAG must match [A-Za-z0-9_.-]+"
[ -n "$ITEM" ] || die "--item N0xx is required (the notebook item that owns this eval)"
ITEMDIR=$(ls -d "$REPO/.agent-research/results/$ITEM"-* 2>/dev/null | head -1)
[ -n "$ITEMDIR" ] && [ "$(ls -d "$REPO/.agent-research/results/$ITEM"-* | wc -l)" -eq 1 ] || die "results/$ITEM-* must match exactly one dir"
[[ "$WEIGHTS" =~ ^(ema|raw)$ ]] || die "--weights ema|raw"
[[ "$RECIPE" =~ ^(n019|none)$ ]] || die "--recipe n019|none"
[[ "$EXTRA" != *,* ]] || die "--extra must not contain commas"
[ -f "$HELD_SPECS" ] || die "held-out pack missing: $HELD_SPECS"
for s in ${SKIP//,/ }; do [[ "$s" =~ ^(heldout|ross|design)$ ]] || die "--skip takes heldout,ross,design"; done
skip() { [[ ",$SKIP," == *,$1,* ]]; }
RES=$ITEMDIR/eval_$TAG
if [ "$DRY" = 0 ]; then
  [ -e "$RES" ] && [ "$OVER" = 0 ] && die "$RES exists (--overwrite to reuse; the tars would be overwritten by the harness too)"
  [ -e "$OUT_DIR/$TAG.tar.gz" ] && [ "$OVER" = 0 ] && die "$OUT_DIR/$TAG.tar.gz exists (pick a new TAG or --overwrite)"
fi
[ "$WEIGHTS" = raw ] && [[ "$TAG" != *raw* ]] && echo "WARNING: --weights raw but TAG has no 'raw' in it ($TAG)"
[ -f "$OUT_DIR/v2_E2_e0596_heldout_s1.tar.gz" ] || echo "WARNING: held-out baseline tar v2_E2_e0596_heldout_s1 not in $OUT_DIR yet (job 1257548): the held-out criterion stays UNVERIFIED until it lands (use --after 1257548, or re-run summarize later)"
RECIPE_STR=$(python3 -c "import json;print(json.load(open('$EC/criteria.json'))['recipe_n019'])")
[ "$RECIPE" = none ] && RECIPE_STR=""
ALL_EXTRA="$RECIPE_STR${EXTRA:+ $EXTRA}"; ALL_EXTRA=${ALL_EXTRA# }
NCFG=$(echo $CFGS | wc -w)
SEEDS_FULL="20261015 20261016"
# ---- estimate (expected = measured N019 times; cap = gpus x walltime, the approval formula)
read -r EXP CAP EVAL_EXP HELD_EXP ROSS_EXP DES_EXP EVAL_WT_EFF < <(python3 - "$N" "$NCFG" "$FULL" "$GPUS" "$EVAL_H" "$HELD_H" "$ROSS_H" "$DES_H" "$EVAL_WT" "$HELD_WT" "$ROSS_WT" "$DES_WT" "$(skip heldout && echo 1 || echo 0)" "$(skip ross && echo 1 || echo 0)" "$(skip design && echo 1 || echo 0)" <<'PYEOF'
import sys, math
n, ncfg, full, g, eh, hh, rh, dh, ewt, hwt, rwt, dwt, sh, sr, sd = sys.argv[1:16]
n, ncfg, full, g, sh, sr, sd = int(n), int(ncfg), int(full), int(g), int(sh), int(sr), int(sd)
def h(w): a = [int(x) for x in w.split(":")]; return a[0] + a[1] / 60 + a[2] / 3600
scale = (n / 32) * (ncfg / 2)
eval_exp = g * float(eh) * scale
nseed = 1 + (2 if full else 0)
held = 0 if sh else g * float(hh); ross = 0 if sr else g * float(rh); des = 0 if sd else g * float(dh)
exp = eval_exp * nseed + held + ross + des
ewt_h = max(h(ewt), math.ceil(float(eh) * scale * 1.8 * 4) / 4) if scale > 1 else h(ewt)
cap = g * ewt_h * nseed + (0 if sh else g * h(hwt)) + (0 if sr else g * h(rwt)) + (0 if sd else g * h(dwt))
hh_, mm = int(ewt_h), int(round((ewt_h - int(ewt_h)) * 60))
print(f"{exp:.2f} {cap:.2f} {eval_exp:.2f} {held:.2f} {ross:.2f} {des:.2f} {hh_:02d}:{mm:02d}:00")
PYEOF
)
echo "=== evalchain $TAG  ckpt $CKPT  weights $WEIGHTS  seed $SEED  cfgs \"$CFGS\"  N $N  recipe $RECIPE${EXTRA:+ + $EXTRA}  full $FULL  skip [${SKIP}]"
echo "=== results $RES"
echo "=== estimated GPU-h: EXPECTED $EXP (eval $EVAL_EXP x $((1 + 2 * FULL)) seed(s), held-out $HELD_EXP, Rossmann $ROSS_EXP, designability $DES_EXP; measured N019 stage times)"
echo "===                  WALLTIME CAP $CAP (gpus x walltime, the >25 GPU-h approval formula); eval walltime $EVAL_WT_EFF"
if python3 -c "import sys; sys.exit(0 if float('$EXP') > float('$MAXGPUH') else 1)"; then
  if [ "$YES" = 1 ]; then echo "=== expected $EXP GPU-h > --max-gpuh $MAXGPUH: proceeding on --yes"
  else echo "REFUSED: expected $EXP GPU-h > --max-gpuh $MAXGPUH (add --yes, or raise --max-gpuh; > 25 GPU-h by the cap needs the user's approval first)"; exit 3; fi
fi
if python3 -c "import sys; sys.exit(0 if float('$CAP') > 25 else 1)"; then
  echo "WARNING: walltime cap $CAP GPU-h > 25: user approval required before submitting"; [ "$YES" = 1 ] || exit 3
fi
[ "$EVAL_WT_EFF" != "$EVAL_WT" ] && echo "WARNING: eval walltime $EVAL_WT_EFF > 2 h: the job becomes preemptable (not resumable; resubmit if preempted)"

# ---- submission
mkdir -p "$LOGS"
[ "$DRY" = 1 ] || mkdir -p "$RES" "$CACHE"
SUBLOG=$RES/submit.log
declare -A JOB SB
log() { echo "$*"; [ "$DRY" = 1 ] || echo "$*" >> "$SUBLOG"; }
submit() {   # submit <stage> <dependency or ''> <sbatch options...> -- <script> <ENV=VAL ...>
  local stage=$1 dep=$2; shift 2
  local opts=() envs=() script="" seen=0
  while [ $# -gt 0 ]; do
    if [ "$seen" = 0 ]; then [ "$1" = "--" ] && { seen=1; shift; script=$1; shift; continue; }; opts+=("$1"); shift
    else envs+=("$1"); shift; fi
  done
  local name=ec_${TAG}_${stage} kill=""
  [ "$stage" != summarize ] && kill=" --kill-on-invalid-dep=yes"     # summarize waits with afterany (+ --after ids): never auto-killed
  local line="env $(printf '%q ' "${envs[@]}")sbatch --parsable --job-name=$name --output=$LOGS/%x-%j.out --export=ALL ${dep:+--dependency=$dep$kill }${opts[*]} $script"
  if [ "$DRY" = 1 ]; then
    log "[dry-run] $stage: $line"; JOB[$stage]="\$J_${stage^^}"
  else
    local id
    id=$(env "${envs[@]}" sbatch --parsable --job-name="$name" --output="$LOGS/%x-%j.out" --export=ALL ${dep:+--dependency="$dep"} $kill "${opts[@]}" "$script") \
      || { echo "FATAL: sbatch failed for $stage; cancelling the chain so far: ${JOB[*]:-none}"; [ -n "${JOB[*]:-}" ] && scancel ${JOB[*]}; exit 2; }
    id=${id%%;*}; JOB[$stage]=$id; log "$stage: job $id  <- $line"
  fi
  SB[$stage]="$line"
}
GPU="--partition=preempt --gres=gpu:h100:$GPUS --cpus-per-task=16 --exclude=$EXCL"
CHAIN="$RES/chain.json"
COMMON=(REPO="$REPO" OUT_DIR="$OUT_DIR" EVALCHAIN_DIR="$EC" NGPU="$GPUS" WEIGHTS="$WEIGHTS")
submit eval "" $GPU --time="$EVAL_WT_EFF" -- "$EC/eval_stage.sbatch" "${COMMON[@]}" CKPT="$CKPT" TAG="$TAG" SPECS="$OUT_DIR/specs_v2.tar.gz" N="$N" CFGS="$CFGS" SEED="$SEED" EXTRA="$ALL_EXTRA"
ALL=${JOB[eval]}; PREV=${JOB[eval]}
if [ "$FULL" = 1 ]; then
  for s in $SEEDS_FULL; do
    submit "s$s" "afterok:${JOB[eval]}" $GPU --time="$EVAL_WT_EFF" -- "$EC/eval_stage.sbatch" "${COMMON[@]}" CKPT="$CKPT" TAG="${TAG}_s$s" SPECS="$OUT_DIR/specs_v2.tar.gz" N="$N" CFGS="$CFGS" SEED="$s" EXTRA="$ALL_EXTRA"
    ALL="$ALL:${JOB[s$s]}"
  done
fi
if ! skip heldout; then
  submit heldout "afterok:$PREV" $GPU --time="$HELD_WT" -- "$EC/eval_stage.sbatch" "${COMMON[@]}" CKPT="$CKPT" TAG="${TAG}_heldout" SPECS="$HELD_SPECS" N=16 CFGS="1.5" SEED="$SEED" EXTRA="$ALL_EXTRA"
  ALL="$ALL:${JOB[heldout]}"; PREV=${JOB[heldout]}
fi
if ! skip ross; then
  submit ross "afterok:$PREV" $GPU --time="$ROSS_WT" -- "$EC/ross_panel.sbatch" "${COMMON[@]}" CKPT="$CKPT" TAG="$TAG" EXTRA="$ALL_EXTRA"
  ALL="$ALL:${JOB[ross]}"; PREV=${JOB[ross]}
fi
if ! skip design; then
  submit design "afterok:$PREV" $GPU --time="$DES_WT" -- "$REPO/scripts/eval/designability.sbatch" REPO="$REPO" OUT_DIR="$OUT_DIR" NGPU="$GPUS" TAG="$TAG" CFGS="1.5 2.0" OUT_SUFFIX="_cfg1.5_2.0"
  ALL="$ALL:${JOB[design]}"; PREV=${JOB[design]}
fi
SUMDEP="afterany:$ALL${AFTER:+:${AFTER//,/:}}"
submit summarize "$SUMDEP" --partition=preempt_cpu --cpus-per-task=16 --time="$SUM_WT" -- "$EC/summarize.sbatch" CHAIN_JSON="$CHAIN" EVALCHAIN_DIR="$EC" PROCS=16
# ---- records
if [ "$DRY" = 0 ]; then
  python3 - "$CHAIN" "$TAG" "$ITEM" "$ITEMDIR" "$RES" "$CKPT" "$WEIGHTS" "$SEED" "$CFGS" "$N" "$RECIPE" "$EXTRA" "$ALL_EXTRA" "$FULL" "$CACHE" "$OUT_DIR" \
      "$EXP" "$CAP" "$EVAL_EXP" "$HELD_EXP" "$ROSS_EXP" "$DES_EXP" "$AFTER" "$HELD_SPECS" \
      "$(for k in "${!JOB[@]}"; do printf '%s=%s\n' "$k" "${JOB[$k]}"; done)" "$(for k in "${!SB[@]}"; do printf '%s\t%s\n' "$k" "${SB[$k]}"; done)" <<'PYEOF'
import json, os, sys, time, getpass, socket
(p, tag, item, itemdir, res, ckpt, weights, seed, cfgs, n, recipe, extra, all_extra, full, cache, out_dir, exp, cap, e_exp, h_exp, r_exp, d_exp, after,
 held_specs, jobs, sbs) = sys.argv[1:27]
J = dict(l.split("=", 1) for l in jobs.splitlines() if l)
S = dict(l.split("\t", 1) for l in sbs.splitlines() if l)
st = os.stat(ckpt)
stages = {"eval": dict(job=J["eval"], tag=tag, seed=int(seed), base_tag="v2_E2_e0596_h02r1cen2m0", expected_gpuh=float(e_exp), sbatch=S["eval"])}
for k, j in J.items():
    if k.startswith("s20"):
        stages["eval_" + k] = dict(job=j, tag=f"{tag}_{k}", seed=int(k[1:]), base_tag=f"v2_E2_e0596_h02r1cen2m0_{k}", expected_gpuh=float(e_exp), sbatch=S[k])
if "heldout" in J:
    stages["heldout"] = dict(job=J["heldout"], tag=f"{tag}_heldout", specs=held_specs, base_tag="v2_E2_e0596_heldout_s1", expected_gpuh=float(h_exp), sbatch=S["heldout"])
if "ross" in J:
    stages["ross"] = dict(job=J["ross"], tar=f"{out_dir}/{tag}.ross.tar.gz", expected_gpuh=float(r_exp), sbatch=S["ross"])
if "design" in J:
    stages["design"] = dict(job=J["design"], json=f"{out_dir}/{tag}_cfg1.5_2.0.designability.json", cfgs="1.5 2.0", expected_gpuh=float(d_exp), sbatch=S["design"])
stages["summarize"] = dict(job=J["summarize"], sbatch=S["summarize"])
chain = dict(tag=tag, item=item, item_dir=itemdir, out_dir=res, ckpt=ckpt, ckpt_bytes=st.st_size,
             ckpt_mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)), weights=weights, seed=int(seed), cfgs=cfgs, n=int(n),
             recipe=recipe, extra=extra, extra_overrides=all_extra, full=bool(int(full)), cache=cache, eval_dir=out_dir, after=after,
             estimate=dict(expected_gpuh=float(exp), cap_gpuh=float(cap)), stages=stages,
             submitted=time.strftime("%Y-%m-%dT%H:%M:%S"), submitted_by=f"{getpass.getuser()}@{socket.gethostname()}")
json.dump(chain, open(p, "w"), indent=1)
json.dump({k: v["job"] for k, v in stages.items()}, open(os.path.join(res, "jobs.json"), "w"), indent=1)
PYEOF
  echo "=== submitted: $(for k in eval s20261015 s20261016 heldout ross design summarize; do [ -n "${JOB[$k]:-}" ] && printf '%s=%s ' "$k" "${JOB[$k]}"; done)"
  echo "=== records: $CHAIN, $RES/jobs.json, $SUBLOG; wait with: ~/.claude/skills/work-log/bin/wait-job.sh ${JOB[summarize]}"
  echo "=== report lands in $RES/summary.md (re-runnable any time: /path/to/env/bin/python $EC/summarize.py --chain $CHAIN)"
else
  echo "=== dry run: nothing submitted; chain.json would be $CHAIN"
fi
