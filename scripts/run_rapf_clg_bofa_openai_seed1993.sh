#!/usr/bin/env bash
# Four GPU shards for RAPF, CLG-CBM, and BOFA on OpenAI CLIP ViT-B/16.
# Each shard contains every method for its assigned datasets and runs serially.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:?usage: $0 <physical-gpu-id> <shard-id>}"
SHARD_ID="${2:?usage: $0 <physical-gpu-id> <shard-id>}"
PYTHON_BIN="${PYTHON_BIN:-/public/home/hanlida/ld/envs/mmcl/bin/python}"
DATA_ROOT="${MMCL_DATA_ROOT:-/public/home/hanlida/Dr.1/Dataset}"
OPENAI_CHECKPOINT="${MMCL_CLIP_PRETRAINED:-/public/home/hanlida/Dr.1/pretrained_models/open_clip_pytorch_model.bin}"
SEED=1993

[[ "${GPU_ID}" =~ ^[0-3]$ ]] || { echo "GPU must be one of 0, 1, 2, 3." >&2; exit 2; }
[[ "${SHARD_ID}" =~ ^[0-3]$ ]] || { echo "Shard must be one of 0, 1, 2, 3." >&2; exit 2; }
[[ "${GPU_ID}" == "${SHARD_ID}" ]] || { echo "GPU and shard must match." >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing mmcl Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing dataset root: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${OPENAI_CHECKPOINT}" ]] || { echo "Missing OpenAI CLIP checkpoint: ${OPENAI_CHECKPOINT}" >&2; exit 2; }

export MMCL_DATA_ROOT="${DATA_ROOT}"
export MMCL_CLIP_PRETRAINED="${OPENAI_CHECKPOINT}"
export PYTHONUNBUFFERED=1

case "${SHARD_ID}" in
    # Aircraft, CIFAR-100, and Stanford Cars.
    0) DATASETS=(aircraft cifar cars) ;;
    # ImageNet-R and CUB-200.
    1) DATASETS=(inr cub) ;;
    # UCF101 and Food-101.
    2) DATASETS=(ucf food) ;;
    # SUN397 and ObjectNet.
    3) DATASETS=(sun objectnet) ;;
esac

declare -A CONFIG_PREFIX=(
    [rapf]=rapf
    [clg_cbm]=CLG-CBM
    [bofa]=bofa
)
declare -A CONFIG_DIR=(
    [rapf]=rapf
    [clg_cbm]=CLG-CBM
    [bofa]=bofa
)
declare -A PROTOCOLS=(
    [aircraft]='B0_Inc10 B50_Inc10'
    [cifar]='B0_Inc10 B50_Inc10'
    [cars]='B0_Inc10 B50_Inc10'
    [inr]='B0_Inc20 B100_Inc20'
    [cub]='B0_Inc20 B100_Inc20'
    [ucf]='B0_Inc10 B50_Inc10'
    [food]='B0_Inc10 B50_Inc10'
    [sun]='B0_Inc30 B150_Inc30'
    [objectnet]='B0_Inc20 B100_Inc20'
)

for method in rapf clg_cbm bofa; do
    for dataset in "${DATASETS[@]}"; do
        for protocol in ${PROTOCOLS[${dataset}]}; do
            config="${CONFIG_PREFIX[${method}]}_${dataset}_${protocol}.json"
            config_path="configs/${CONFIG_DIR[${method}]}/${config}"
            [[ -f "${ROOT}/${config_path}" ]] || { echo "Missing config: ${config_path}" >&2; exit 2; }
            echo "==== START ${method} OpenAI ViT-B/16: ${config}, seed=${SEED}, GPU=${GPU_ID} ===="
            if [[ "${MMCL_DRY_RUN:-0}" == "1" ]]; then
                continue
            fi
            (
                cd "${ROOT}"
                "${PYTHON_BIN}" scripts/run_openai_config.py \
                    --config="${config_path}" --device="${GPU_ID}" --seed="${SEED}"
            )
            echo "==== FINISHED ${method} OpenAI ViT-B/16: ${config}, seed=${SEED}, GPU=${GPU_ID} ===="
        done
    done
done

echo "==== RAPF/CLG-CBM/BOFA OpenAI queue complete: seed=${SEED}, GPU=${GPU_ID}, shard=${SHARD_ID} ===="
