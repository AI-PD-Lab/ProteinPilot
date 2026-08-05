#!/bin/bash
# test_phase3_comparison.sh
#
# Phase 3 inference test using a SLURM array across MIG slices on t38cn056.
# 11 array tasks (6 no-CFG + 5 CFG), max 8 concurrent, 1 MIG GPU each.
# A dependency plot job aggregates results and generates comparison figures.
#
# Usage:
#   bash atomworks/Shell/test_phase3_comparison.sh
#
# Overrides:
#   CKPT=... N_BATCHES=20 CFG_SCALE=2.0 bash atomworks/Shell/test_phase3_comparison.sh

set -euo pipefail

LOG_DIR=${LOG_DIR:-/work/gerstung/HARMONIA/logs}
FOUNDRY=${FOUNDRY:-/work/gerstung/HARMONIA/foundry}
PYTHON=${PYTHON:-/zfs/s01/z04/home/gerstung/miniforge3/envs/atomworks/bin/python3}

LENGTH=${LENGTH:-120}
N_BATCHES=${N_BATCHES:-20}
DIFFUSION_BATCH_SIZE=${DIFFUSION_BATCH_SIZE:-4}
NUM_TIMESTEPS=${NUM_TIMESTEPS:-120}
CFG_SCALE=${CFG_SCALE:-2.0}

NODELIST=${NODELIST:-t38cn056}
RESERVATION=${RESERVATION:-AIPD}
PARTITION=${PARTITION:-barracuda}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM_PER_TASK=${MEM_PER_TASK:-48G}
TIME=${TIME:-02:00:00}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}

CKPT=${CKPT:-$(find ${LOG_DIR}/train/ss-ratio-finetune-phase2 -name "*.ckpt" 2>/dev/null | sort -V | tail -1)}
if [ -z "$CKPT" ]; then
    echo "ERROR: no Phase 3 checkpoint found under ${LOG_DIR}/train/ss-ratio-finetune-phase2"
    exit 1
fi

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-${LOG_DIR}/ss_ratio_effect/phase3_${RUN_ID}}
mkdir -p "$OUT_ROOT"

# Task index → (mode, target) mapping
# 0-5: no-CFG for targets 0.0 0.25 0.5 0.75 1.0 + unconditioned
# 6-10: CFG for targets 0.0 0.25 0.5 0.75 1.0
N_TASKS=11

echo "Phase 3 array test:"
echo "  Checkpoint   : $CKPT"
echo "  Output dir   : $OUT_ROOT"
echo "  Tasks        : $N_TASKS (array 0-$((N_TASKS-1))%${MAX_CONCURRENT})"
echo "  Designs/task : ${N_BATCHES} batches × ${DIFFUSION_BATCH_SIZE} = $((N_BATCHES * DIFFUSION_BATCH_SIZE))"
echo "  Node         : $NODELIST (MIG slices, 1 per task)"
echo ""

# ── Array inference job ───────────────────────────────────────────────────────
INF_SCRIPT=$(mktemp /tmp/phase3_inf.XXXXXX.sh)
cat > "$INF_SCRIPT" << EOFINF
#!/bin/bash
#SBATCH --job-name=p3_inf
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --nodelist=${NODELIST}
#SBATCH --reservation=${RESERVATION}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM_PER_TASK}
#SBATCH --time=${TIME}
#SBATCH --array=0-$((N_TASKS-1))%${MAX_CONCURRENT}
#SBATCH --output=${OUT_ROOT}/inf_%a.out
#SBATCH --error=${OUT_ROOT}/inf_%a.err

set -euo pipefail
export HYDRA_FULL_ERROR=1
export DEBUG=false

TASK=\${SLURM_ARRAY_TASK_ID}

# Decode task → mode and target
TARGETS_NOCFG=(0.0 0.25 0.5 0.75 1.0 uncond)
TARGETS_CFG=(0.0 0.25 0.5 0.75 1.0)

if [ "\$TASK" -lt 6 ]; then
    MODE="nocfg"
    T_IDX=\$TASK
    TARGET=\${TARGETS_NOCFG[\$T_IDX]}
else
    MODE="cfg"
    T_IDX=\$(( TASK - 6 ))
    TARGET=\${TARGETS_CFG[\$T_IDX]}
fi

echo "=== Task \${TASK}: \${MODE} target=\${TARGET} started: \$(date) ==="
echo "    GPU: \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "    Checkpoint: ${CKPT}"
echo ""

TASK_OUT="${OUT_ROOT}/\${MODE}_\${TARGET}"
mkdir -p "\${TASK_OUT}"

cd ${FOUNDRY}

# Build per-task input JSON
${PYTHON} - "\${TASK_OUT}/input.json" "\${TARGET}" "${LENGTH}" "${N_BATCHES}" "${DIFFUSION_BATCH_SIZE}" "\${MODE}" << 'PYJSON'
import json, sys
out_path, target_s, length, n_batches, batch_size, mode = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
)
n = n_batches * batch_size
specs = {}
if target_s == "uncond":
    for i in range(n):
        specs[f"unc_{i:03d}"] = {"length": str(length), "extra": {"target_ss_ratio": -1.0}}
else:
    t = float(target_s)
    tstr = f"ssr_{int(round(t * 100)):03d}"
    for i in range(n):
        specs[f"{tstr}_{i:03d}"] = {"length": str(length), "ss_ratio": t, "extra": {"target_ss_ratio": t}}
with open(out_path, "w") as f:
    json.dump(specs, f, indent=2)
print(f"Built {len(specs)} designs for mode={mode} target={target_s}")
PYJSON

# CFG flags
if [ "\${MODE}" = "cfg" ]; then
    CFG_ARGS="inference_sampler.use_classifier_free_guidance=True inference_sampler.cfg_scale=${CFG_SCALE}"
else
    CFG_ARGS="inference_sampler.use_classifier_free_guidance=False"
fi

srun --cpu-bind=none ${PYTHON} models/rfd3/src/rfd3/run_inference.py \
    ckpt_path=${CKPT} \
    inputs="\${TASK_OUT}/input.json" \
    out_dir="\${TASK_OUT}/designs" \
    global_prefix="" \
    skip_existing=False \
    n_batches=${N_BATCHES} \
    diffusion_batch_size=${DIFFUSION_BATCH_SIZE} \
    devices_per_node=1 \
    inference_sampler.num_timesteps=${NUM_TIMESTEPS} \
    dump_prediction_metadata_json=True \
    dump_trajectories=False \
    prevalidate_inputs=True \
    \${CFG_ARGS}

echo ""
echo "=== Task \${TASK}: \${MODE} target=\${TARGET} done: \$(date) ==="
EOFINF

INF_JOB=$(sbatch --parsable "$INF_SCRIPT")
echo "Submitted inference array job ${INF_JOB}"

# ── Plot + aggregate job (runs after all array tasks finish) ──────────────────
PLOT_SCRIPT=$(mktemp /tmp/phase3_plot.XXXXXX.sh)
cat > "$PLOT_SCRIPT" << EOFPLOT
#!/bin/bash
#SBATCH --job-name=p3_plot
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --dependency=afterok:${INF_JOB}
#SBATCH --output=${OUT_ROOT}/plot.out
#SBATCH --error=${OUT_ROOT}/plot.err

set -euo pipefail
echo "=== Phase 3 analysis + plotting started: \$(date) ==="

${PYTHON} - "${OUT_ROOT}" "${CFG_SCALE}" << 'PYANALYZE'
import csv
import gzip
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import biotite.structure as struc
from biotite.structure.io.pdbx import CIFFile, get_structure

warnings.filterwarnings("ignore")

out_root = Path(sys.argv[1])
cfg_scale = float(sys.argv[2])


def load_cifs(design_dir: Path):
    records = []
    for cif in sorted(design_dir.glob("*.cif.gz")):
        unc_m = re.match(r"unc_", cif.name)
        ssr_m = re.search(r"ssr_(\d{3})_", cif.name)
        if unc_m:
            target = -1.0
        elif ssr_m:
            target = int(ssr_m.group(1)) / 100.0
        else:
            continue
        try:
            with gzip.open(cif, "rt") as fh:
                arr = get_structure(CIFFile.read(fh), model=1)
        except Exception:
            continue
        records.append((target, arr, cif.name))
    return records


def compute_metrics(arr):
    prot = arr[struc.filter_amino_acids(arr)]
    if prot.array_length() == 0:
        return None
    sse = struc.annotate_sse(prot)
    n_res = len(sse)
    n_helix = int(np.sum(sse == "a"))
    n_sheet = int(np.sum(sse == "b"))
    n_loop  = int(np.sum(sse == "c"))
    structured = n_helix + n_sheet
    ss_ratio = float(n_sheet / structured) if structured > 0 else float("nan")
    loop_fraction = float(n_loop / n_res)

    ss_binary = np.where((sse == "a") | (sse == "b"), 1, 0)
    n_ss_elements = int(np.sum(np.diff(np.concatenate([[0], ss_binary, [0]])) == 1))

    ca_mask   = prot.atom_name == "CA"
    ca_coords = prot.coord[ca_mask]
    if len(ca_coords) < 2:
        return None

    ca_diffs = np.linalg.norm(np.diff(ca_coords, axis=0), axis=1)
    n_chainbreaks = int(np.sum(ca_diffs > 4.2))
    max_ca_dev    = float(np.max(np.abs(ca_diffs - 3.8)))

    clashes = 0
    n_ca = len(ca_coords)
    for i in range(n_ca - 2):
        dists = np.linalg.norm(ca_coords[i+2:] - ca_coords[i], axis=1)
        clashes += int(np.sum(dists < 3.5))

    centroid = ca_coords.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((ca_coords - centroid) ** 2, axis=1))))

    res_names = prot.res_name[ca_mask]
    ala_fraction = float(np.sum(res_names == "ALA") / len(res_names))

    return {
        "ss_ratio": ss_ratio,
        "loop_fraction": loop_fraction,
        "non_loop_fraction": 1.0 - loop_fraction,
        "n_chainbreaks": n_chainbreaks,
        "backbone_clashes": clashes,
        "max_ca_deviation": max_ca_dev,
        "num_ss_elements": n_ss_elements,
        "radius_of_gyration": rg,
        "alanine_content": ala_fraction,
    }


def analyze_records(records):
    by_target = defaultdict(list)
    all_metrics = []
    for target, arr, fname in records:
        m = compute_metrics(arr)
        if m is None:
            continue
        m["target"] = target
        m["file"] = fname
        by_target[target].append(m["ss_ratio"])
        all_metrics.append(m)
    return by_target, all_metrics


# Collect all task output dirs
nocfg_records, cfg_records = [], []
for task_dir in sorted(out_root.glob("nocfg_*")):
    design_dir = task_dir / "designs"
    if design_dir.exists():
        nocfg_records.extend(load_cifs(design_dir))
for task_dir in sorted(out_root.glob("cfg_*")):
    design_dir = task_dir / "designs"
    if design_dir.exists():
        cfg_records.extend(load_cifs(design_dir))

print(f"Loaded {len(nocfg_records)} no-CFG designs, {len(cfg_records)} CFG designs")

nocfg_by_target, nocfg_metrics = analyze_records(nocfg_records)
cfg_by_target,   cfg_metrics   = analyze_records(cfg_records)

# Save summary CSVs
for label, metrics in [("nocfg", nocfg_metrics), ("cfg", cfg_metrics)]:
    if not metrics:
        continue
    csv_path = out_root / f"{label}_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(metrics[0].keys()))
        w.writeheader(); w.writerows(metrics)
    print(f"Saved {csv_path}")

# Print correlation table
print("\n=== ss_ratio Correlation ===")
for label, by_target in [("No-CFG", nocfg_by_target), (f"CFG {cfg_scale}x", cfg_by_target)]:
    t_vals = sorted(k for k in by_target if k >= 0)
    if len(t_vals) < 2:
        continue
    means = [np.mean(by_target[t]) for t in t_vals]
    r = float(np.corrcoef(t_vals, means)[0, 1])
    print(f"\n{label}  r={r:.3f}")
    for t, m in zip(t_vals, means):
        vals = np.asarray(by_target[t])
        print(f"  {t:.2f}  n={len(vals)}  mean={m:.3f}  std={vals.std(ddof=1):.3f}")

# ── Plot 1: ss_ratio correlation ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Phase 3 (epoch-0590): ss_ratio Conditioning — No-CFG vs CFG", fontsize=13, fontweight="bold")

for ax, label, by_target, color in [
    (axes[0], "No-CFG",             nocfg_by_target, "#3274A1"),
    (axes[1], f"CFG scale={cfg_scale}", cfg_by_target,   "#E1812C"),
]:
    t_vals = sorted(k for k in by_target if k >= 0)
    if not t_vals:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        continue
    means = np.asarray([np.mean(by_target[t]) for t in t_vals])
    stds  = np.asarray([np.std(by_target[t], ddof=1) if len(by_target[t]) > 1 else 0.0 for t in t_vals])
    t_arr = np.asarray(t_vals)
    r = float(np.corrcoef(t_arr, means)[0, 1]) if len(t_arr) > 1 else float("nan")

    for t in t_vals:
        ys = by_target[t]
        xs = np.full(len(ys), t) + np.random.uniform(-0.015, 0.015, len(ys))
        ax.scatter(xs, ys, alpha=0.15, s=8, color=color)

    ax.errorbar(t_arr, means, yerr=stds, fmt="o-", color=color,
                linewidth=2, markersize=7, capsize=4, label="mean ± std")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="ideal")

    all_vals = [v for t in t_vals for v in by_target[t]]
    floor = float(np.percentile(all_vals, 5))
    ceil  = float(np.percentile(all_vals, 95))
    ax.set_xlabel("Target ss_ratio", fontsize=11)
    ax.set_ylabel("Achieved ss_ratio", fontsize=11)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{label}\nr = {r:.3f}  |  floor={floor:.2f}  ceil={ceil:.2f}", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
p1 = out_root / "phase3_ss_ratio_correlation.png"
plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
print(f"\nSaved: {p1}")

# ── Plot 2: Quality metrics ───────────────────────────────────────────────────
metric_defs = [
    ("loop_fraction",       "Loop Fraction"),
    ("non_loop_fraction",   "Non-loop Fraction"),
    ("n_chainbreaks",       "Chain Breaks (count)"),
    ("backbone_clashes",    "Backbone Clashes"),
    ("max_ca_deviation",    "Max Cα Deviation (Å)"),
    ("num_ss_elements",     "# SS Elements"),
    ("radius_of_gyration",  "Radius of Gyration (Å)"),
    ("alanine_content",     "Alanine Content"),
]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Phase 3 (epoch-0590): Quality Metrics", fontsize=13, fontweight="bold")
axes = axes.flatten()

for ax, (key, title) in zip(axes, metric_defs):
    nc_vals = [m[key] for m in nocfg_metrics if np.isfinite(m.get(key, float("nan")))]
    c_vals  = [m[key] for m in cfg_metrics   if np.isfinite(m.get(key, float("nan")))]
    if not nc_vals and not c_vals:
        ax.set_title(title); continue

    parts = ax.violinplot([nc_vals, c_vals], positions=[0, 1], showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], ["#3274A1", "#E1812C"]):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    for comp in ("cbars", "cmins", "cmaxes"):
        parts[comp].set_color("gray")

    for i, (vals, col) in enumerate(zip([nc_vals, c_vals], ["#3274A1", "#E1812C"])):
        if vals:
            ax.scatter([i], [np.mean(vals)], color=col, s=50, zorder=5, marker="D")
            ylim = ax.get_ylim()
            ax.text(i, ylim[0] + 0.03 * (ylim[1] - ylim[0]),
                    f"μ={np.mean(vals):.2f}", ha="center", fontsize=8, color=col, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No-CFG", f"CFG {cfg_scale}×"], fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
p2 = out_root / "phase3_quality_metrics.png"
plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p2}")

print("\n=== Done ===")
PYANALYZE

echo "=== Plotting complete: \$(date) ==="
EOFPLOT

PLOT_JOB=$(sbatch --parsable "$PLOT_SCRIPT")
echo "Submitted plot job       ${PLOT_JOB} (depends on ${INF_JOB})"
echo ""
echo "Inference  : tail -f ${OUT_ROOT}/inf_0.out   (one per array task)"
echo "Plot log   : tail -f ${OUT_ROOT}/plot.out"
echo "Results    : ${OUT_ROOT}/"
