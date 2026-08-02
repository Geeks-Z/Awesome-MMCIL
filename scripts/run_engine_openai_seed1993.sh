#!/usr/bin/env bash
# Evaluate ENGINE on the full native-MMCL benchmark with OpenAI CLIP ViT-B/16.
# One invocation owns exactly one physical GPU and runs its assigned protocols
# serially. Start one invocation each for GPU 1 and GPU 2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:?usage: $0 <physical-gpu-id>}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/hanlida/ld/envs/mmcl/bin/python}"
DATA_ROOT="${MMCL_DATA_ROOT:-/public/home/hanlida/Dr.1/Dataset}"
OPENAI_CHECKPOINT="${MMCL_CLIP_PRETRAINED:-/public/home/hanlida/Dr.1/pretrained_models/open_clip_pytorch_model.bin}"
SEED=1993

[[ "${GPU_ID}" == "1" || "${GPU_ID}" == "2" ]] || {
    echo "ENGINE OpenAI evaluation is assigned to physical GPU 1 or 2, got ${GPU_ID}." >&2
    exit 2
}
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing mmcl Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing dataset root: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${OPENAI_CHECKPOINT}" ]] || { echo "Missing OpenAI CLIP checkpoint: ${OPENAI_CHECKPOINT}" >&2; exit 2; }

export MMCL_DATA_ROOT="${DATA_ROOT}"
export MMCL_CLIP_PRETRAINED="${OPENAI_CHECKPOINT}"
export PYTHONUNBUFFERED=1

if [[ "${GPU_ID}" == "1" ]]; then
    CONFIGS=(
        engine_aircraft_B0_Inc10.json
        engine_aircraft_B50_Inc10.json
        engine_cifar_B0_Inc10.json
        engine_cifar_B50_Inc10.json
        engine_cars_B0_Inc10.json
        engine_cars_B50_Inc10.json
        engine_ucf_B0_Inc10.json
        engine_ucf_B50_Inc10.json
        engine_food_B0_Inc10.json
        engine_food_B50_Inc10.json
    )
else
    CONFIGS=(
        engine_inr_B0_Inc20.json
        engine_inr_B100_Inc20.json
        engine_cub_B0_Inc20.json
        engine_cub_B100_Inc20.json
        engine_sun_B0_Inc30.json
        engine_sun_B150_Inc30.json
        engine_objectnet_B0_Inc20.json
        engine_objectnet_B100_Inc20.json
    )
fi

for config in "${CONFIGS[@]}"; do
    echo "==== START ENGINE OpenAI ViT-B/16: ${config}, seed=${SEED}, GPU=${GPU_ID} ===="
    (
        cd "${ROOT}"
        "${PYTHON_BIN}" scripts/run_engine_openai.py \
            --config="configs/engine/${config}" --device="${GPU_ID}" --seed="${SEED}"
    )
    echo "==== FINISHED ENGINE OpenAI ViT-B/16: ${config}, seed=${SEED}, GPU=${GPU_ID} ===="
done

echo "==== ENGINE OpenAI ViT-B/16 queue complete: seed=${SEED}, GPU=${GPU_ID} ===="
