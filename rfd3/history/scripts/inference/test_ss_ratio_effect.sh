#!/bin/bash
# test_ss_ratio_effect.sh
#
# Post-training effect test for the ss_ratio adapter.
# Creates de novo backbone specs at fixed ss_ratio targets, runs inference from
# the latest Phase 1 checkpoint, then summarizes achieved sheet ratio from the
# generated CIFs using biotite.structure.annotate_sse.
#
# Usage:
#   bash atomworks/Shell/test_ss_ratio_effect.sh
#
# Useful overrides:
#   CKPT=/path/to/epoch-0580.ckpt N_BATCHES=8 NUM_TIMESTEPS=120 bash atomworks/Shell/test_ss_ratio_effect.sh

set -euo pipefail

LOG_DIR=${LOG_DIR:-/path/to/work/HARMONIA/logs}
FOUNDRY=${FOUNDRY:-/path/to/work/HARMONIA/foundry}
PYTHON=${PYTHON:-/path/to/home/miniforge3/envs/atomworks/bin/python3}

TARGETS=${TARGETS:-"0.0 0.25 0.5 0.75 1.0"}
LENGTH=${LENGTH:-120}
N_BATCHES=${N_BATCHES:-4}
DIFFUSION_BATCH_SIZE=${DIFFUSION_BATCH_SIZE:-4}
NUM_TIMESTEPS=${NUM_TIMESTEPS:-120}
DEVICES_PER_NODE=${DEVICES_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-96G}
TIME=${TIME:-08:00:00}

CKPT=${CKPT:-$(find ${LOG_DIR}/train/ss-ratio-finetune-phase1 -name "*.ckpt" 2>/dev/null | sort -V | tail -1)}
if [ -z "$CKPT" ]; then
    echo "ERROR: no Phase 1 checkpoint found under ${LOG_DIR}/train/ss-ratio-finetune-phase1"
    exit 1
fi

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-${LOG_DIR}/ss_ratio_effect/${RUN_ID}}
INPUT_JSON=${OUT_ROOT}/ss_ratio_targets.json
SUMMARY_CSV=${OUT_ROOT}/summary.csv
mkdir -p "$OUT_ROOT"

${PYTHON} - "$INPUT_JSON" "$TARGETS" "$LENGTH" << 'PYJSON'
import json
import sys

out_json, targets_s, length = sys.argv[1], sys.argv[2], sys.argv[3]
targets = [float(x) for x in targets_s.split()]

specs = {}
for target in targets:
    key = f"ssr_{int(round(target * 100)):03d}"
    specs[key] = {
        "length": str(length),
        "ss_ratio": target,
        "extra": {"target_ss_ratio": target},
    }

with open(out_json, "w") as handle:
    json.dump(specs, handle, indent=4)
PYJSON

JOB_SCRIPT=$(mktemp /tmp/test_ss_ratio_effect.XXXXXX.sh)
cat > "$JOB_SCRIPT" << EOFJOB
#!/bin/bash
#SBATCH --job-name=ss_ratio_eval
#SBATCH --partition=barracuda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${DEVICES_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${OUT_ROOT}/slurm-%j.out
#SBATCH --error=${OUT_ROOT}/slurm-%j.err

set -euo pipefail
export HYDRA_FULL_ERROR=1
export DEBUG=false

echo "=== ss_ratio effect test started: \$(date) ==="
echo "Checkpoint: ${CKPT}"
echo "Input JSON: ${INPUT_JSON}"
echo "Output dir: ${OUT_ROOT}/designs"
echo "Targets: ${TARGETS}"
echo ""

cd ${FOUNDRY}

srun --cpu-bind=none ${PYTHON} models/rfd3/src/rfd3/run_inference.py \
    ckpt_path=${CKPT} \
    inputs=${INPUT_JSON} \
    out_dir=${OUT_ROOT}/designs \
    global_prefix="" \
    skip_existing=False \
    n_batches=${N_BATCHES} \
    diffusion_batch_size=${DIFFUSION_BATCH_SIZE} \
    devices_per_node=${DEVICES_PER_NODE} \
    inference_sampler.num_timesteps=${NUM_TIMESTEPS} \
    dump_prediction_metadata_json=True \
    dump_trajectories=False \
    prevalidate_inputs=True

${PYTHON} - "${OUT_ROOT}/designs" "${SUMMARY_CSV}" << 'PYSUMMARY'
import csv
import gzip
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import biotite.structure as struc
from biotite.structure.io import load_structure
from biotite.structure.io.pdbx import CIFFile, get_structure

design_dir = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
rows = []

for cif in sorted(design_dir.glob("*.cif.gz")):
    match = re.search(r"ssr_(\d{3})_", cif.name)
    if not match:
        continue
    target = int(match.group(1)) / 100.0
    if cif.name.endswith(".cif.gz"):
        with gzip.open(cif, "rt") as handle:
            atom_array = get_structure(CIFFile.read(handle), model=1)
    else:
        atom_array = load_structure(str(cif), model=1)
    protein = atom_array[struc.filter_amino_acids(atom_array)]
    if protein.array_length() == 0:
        continue
    sse = struc.annotate_sse(protein)
    helix = int(np.sum(sse == "a"))
    sheet = int(np.sum(sse == "b"))
    structured = helix + sheet
    achieved = float(sheet / structured) if structured else float("nan")
    rows.append(
        {
            "target_ss_ratio": target,
            "achieved_ss_ratio": achieved,
            "helix_residues": helix,
            "sheet_residues": sheet,
            "structured_residues": structured,
            "file": cif.name,
        }
    )

summary_csv.parent.mkdir(parents=True, exist_ok=True)
fieldnames = ["target_ss_ratio", "achieved_ss_ratio", "helix_residues", "sheet_residues", "structured_residues", "file"]
with open(summary_csv, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

by_target = defaultdict(list)
for row in rows:
    if np.isfinite(row["achieved_ss_ratio"]):
        by_target[row["target_ss_ratio"]].append(row["achieved_ss_ratio"])

print("target,n,mean,std")
for target in sorted(by_target):
    vals = np.asarray(by_target[target], dtype=float)
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    print(f"{target:.2f},{len(vals)},{float(vals.mean()):.3f},{std:.3f}")

if len(by_target) >= 2:
    targets = np.asarray(sorted(by_target), dtype=float)
    means = np.asarray([np.mean(by_target[t]) for t in targets], dtype=float)
    corr = float(np.corrcoef(targets, means)[0, 1]) if len(targets) > 1 else float("nan")
    print(f"target_mean_correlation,{corr:.3f}")
PYSUMMARY

echo ""
echo "Summary CSV: ${SUMMARY_CSV}"
echo "=== ss_ratio effect test complete: \$(date) ==="
EOFJOB

JOB=$(sbatch --parsable "$JOB_SCRIPT")
echo "Submitted ss_ratio effect test job ${JOB}"
echo "Checkpoint: ${CKPT}"
echo "Input JSON: ${INPUT_JSON}"
echo "Outputs: ${OUT_ROOT}"
echo "Monitor: tail -f ${OUT_ROOT}/slurm-${JOB}.out"
