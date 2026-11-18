#!/bin/bash
# Harmonia-RFD3 SS/ADJ inference on the local GPU(s) (no scheduler).
#
#   CKPT=checkpoints/Harmonia-RFD3.ckpt INPUT_JSON=/path/inputs.json OUT_ROOT=out bash run_normal_ss_adj_inference.sh
#
# If INPUT_JSON is unset, a small example JSON plus tensors is generated in OUT_ROOT/input.
# Run it from anywhere; paths resolve against the repo (rfd3/). Wrap it in your own sbatch/srun header on a cluster.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RFD3_DIR=${RFD3_DIR:-$(cd "${HERE}/../../.." && pwd)}
FOUNDRY=${FOUNDRY:-${RFD3_DIR}/foundry}
PYTHON=${PYTHON:-python3}

CKPT=${CKPT:-${RFD3_DIR}/../checkpoints/Harmonia-RFD3.ckpt}
OUT_ROOT=${OUT_ROOT:-${PWD}/harmonia_rfd3_out}
OUT_DESIGNS=${OUT_DESIGNS:-${OUT_ROOT}/designs}
N_BATCHES=${N_BATCHES:-1}
DIFFUSION_BATCH_SIZE=${DIFFUSION_BATCH_SIZE:-4}
NUM_TIMESTEPS=${NUM_TIMESTEPS:-120}
USE_CFG=${USE_CFG:-True}
CFG_SCALE=${CFG_SCALE:-2.0}
DEVICES_PER_NODE=${DEVICES_PER_NODE:-1}

[ -s "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT} (python checkpoints/download.py --model rfd3, or set CKPT=)"; exit 1; }
mkdir -p "${OUT_ROOT}"
if [ -z "${INPUT_JSON:-}" ]; then
    INPUT_JSON=$("${PYTHON}" "${HERE}/make_example_inputs.py" --out-dir "${OUT_ROOT}/input")
fi

echo "Harmonia-RFD3 inference: ckpt=${CKPT} inputs=${INPUT_JSON} out=${OUT_DESIGNS} cfg=${USE_CFG}(${CFG_SCALE})"

export PYTHONPATH="${FOUNDRY}/models/rfd3/src:${FOUNDRY}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1 DEBUG=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${FOUNDRY}"
"${PYTHON}" models/rfd3/src/rfd3/run_inference.py \
    ckpt_path="${CKPT}" \
    inputs="${INPUT_JSON}" \
    out_dir="${OUT_DESIGNS}" \
    global_prefix="" \
    skip_existing=False \
    n_batches=${N_BATCHES} \
    diffusion_batch_size=${DIFFUSION_BATCH_SIZE} \
    devices_per_node=${DEVICES_PER_NODE} \
    inference_sampler.num_timesteps=${NUM_TIMESTEPS} \
    inference_sampler.use_classifier_free_guidance=${USE_CFG} \
    inference_sampler.cfg_scale=${CFG_SCALE} \
    dump_prediction_metadata_json=True \
    dump_trajectories=False \
    prevalidate_inputs=True
