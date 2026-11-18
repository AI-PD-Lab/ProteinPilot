#!/bin/bash -l
#SBATCH --job-name=harm_design
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=/path/to/work_rfd3/v2/eval/logs/%x-%j.out
# HARMONIA-v2 guardrail harness, designability half (WP-EVAL step 4). NOT RUN when written (no job was submitted);
# for FINALISTS only: it refolds the generated designs, so it costs more per design than the generation job.
#
#   1. extract the selected designs of an eval tar (<TAG>.tar.gz of eval_checkpoint.sbatch) to node-local PDBs
#   2. LigandMPNN ligand_mpnn (protein-only PDB), K=8 sequences, T=0.1, seed 37          (GPU 0, ~minutes)
#   3. ESMFold2 (pinned snapshots, ESMC-6B asserted) refold of every sequence, 4 shards in parallel (1 per GPU)
#   4. summary: designable = scRMSD < 2 A and pLDDT > 80 (70 also reported), TM >= 0.5 to the reference fold, joint rate
# Output: $OUT_DIR/<TAG>.designability.json + .md  (2 files; PDBs/fasta stay node-local)
#
# usage:  sbatch --export=ALL,TAG=v2_stageA1_e0003 scripts/eval/designability.sbatch
#   TAG        evaluation tag (tar $OUT_DIR/$TAG.tar.gz)         OUT_DIR  default /path/to/work_rfd3/v2/eval
#   OUT_SUFFIX appended to the output name (e.g. _cfg2.0 so a CFG-2.0 refold does not overwrite the default-CFG file)
#   CFGS       CFG scales to refold (default "1.0 1.5")          ARMS     (default "full sparse8 dense0_true")
#   MAXCELL    designs per (target, arm, CFG) cell (default 8)    K        sequences per design (8)
#   REPO       worktree (default the v2 worktree; only used for the scripts)   SPECS  spec pack (refs for the TM criterion)
# PLACEMENT: GPU (ESMFold2 6B + ESMC 6B bf16 per GPU; 4 shards, one per GPU). CPU is not a competitor for a 6B structure model.
# COST: default 5 targets x 3 arms x 2 CFGs x 8 designs = 240 designs x 8 sequences = 1920 folds; at ~3-5 s/fold [estimate from the MicroGreens
#   protein cells, not measured for these lengths] ~0.5 h wall on 4 GPUs = ~2-3 GPU-h; hard cap 1.5 h x 4 = 6 GPU-h (< 25 GPU-h gate).
set -uo pipefail
export LC_ALL=C
: "${TAG:?set TAG=<evaluation tag>}"
W=/path/to/work_rfd3
REPO=${REPO:-$W/harmonia_study}
OUT_DIR=${OUT_DIR:-$W/v2/eval}
SPECS=${SPECS:-$OUT_DIR/specs_v2.tar.gz}
CFGS=${CFGS:-"1.0 1.5"}; ARMS=${ARMS:-"full sparse8 dense0_true"}; MAXCELL=${MAXCELL:-8}; K=${K:-8}
NGPU=${NGPU:-4}
ES=/path/to/env/bin/python
EV=$REPO/scripts/eval
export HF_HOME=/path/to/work/cache/huggingface
TARF=$OUT_DIR/$TAG.tar.gz
[ -f "$TARF" ] || { echo "FATAL: $TARF not found"; exit 1; }
[ -f "$SPECS" ] || { echo "FATAL: $SPECS not found"; exit 1; }
T=/tmp/harmonia_design_${SLURM_JOB_ID:-local$$}; mkdir -p "$T"
trap 'rc=$?; rm -rf "$T"; exit $rc' EXIT

echo "### prepare $(date +%T)"
CUDA_VISIBLE_DEVICES=0 $ES -I "$EV/designability.py" prepare --tar "$TARF" --specs-tar "$SPECS" --work "$T" \
    --cfgs $CFGS --arms $ARMS --max-per-cell "$MAXCELL" --k "$K" --python "$ES" || { echo "FATAL: prepare failed (see $T/lmpnn.log)"; tail -20 "$T/lmpnn.log" 2>/dev/null; exit 2; }
NFA=$(ls "$T"/seqs/*.fa 2>/dev/null | wc -l); NPDB=$(ls "$T"/pdbs/*.pdb | wc -l)
[ "$NFA" -eq "$NPDB" ] || { echo "FATAL: $NFA fasta != $NPDB pdb"; exit 2; }

echo "### fold $(date +%T)"
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g $ES -I "$EV/designability.py" fold --work "$T" --shard "$g/$NGPU" --out "$T/fold_$g.json" --k "$K" > "$T/fold_$g.log" 2>&1 &
done
wait
for g in $(seq 0 $((NGPU-1))); do [ -s "$T/fold_$g.json" ] || { echo "FATAL: shard $g produced no result"; tail -20 "$T/fold_$g.log"; exit 3; }; done

echo "### summarize $(date +%T)"
$ES -I "$EV/designability.py" summarize --work "$T" --out-prefix "$OUT_DIR/$TAG${OUT_SUFFIX:-}.designability" || exit 4
echo "DONE $(date +%T)"
