#!/usr/bin/env bash
# Run the native-MMCL prompt baselines with OpenAI CLIP ViT-B/16 at seed 1993.
#
# The fixed assignments keep the requested physical GPUs isolated:
#   l2p        -> GPU 1 (or recovery shards on GPUs 5, 6, and 7)
#   dualprompt -> GPU 2
#   coda       -> GPU 5
#   proof 0    -> GPU 6 (10 configs)
#   proof 1    -> GPU 7 ( 8 configs)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${1:?usage: $0 <l2p|dualprompt|coda|proof> <physical-gpu-id> [proof-shard]}"
GPU_ID="${2:?usage: $0 <l2p|dualprompt|coda|proof> <physical-gpu-id> [proof-shard]}"
PROOF_SHARD="${3:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python}"
DATA_ROOT="${MMCL_DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
OPENAI_CHECKPOINT="${MMCL_CLIP_PRETRAINED:-/home/team/zhaohongwei/pretrained_models/open_clip_pytorch_model.bin}"
SEED="${MMCL_SEED:-1993}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing mmcl Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "Missing dataset root: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${OPENAI_CHECKPOINT}" ]] || { echo "Missing OpenAI CLIP checkpoint: ${OPENAI_CHECKPOINT}" >&2; exit 2; }

if [[ "$(ulimit -n)" -lt 65535 ]]; then
    ulimit -n 65535
fi

case "${METHOD}:${GPU_ID}:${PROOF_SHARD}" in
    l2p:1:|l2p:5:|l2p:6:|l2p:7:|dualprompt:2:)
        CONFIG_DIR="${METHOD}"
        CONFIG_PREFIX="${METHOD}"
        ;;
    coda:5:)
        CONFIG_DIR="coda_prompt"
        CONFIG_PREFIX="coda_prompt"
        ;;
    proof:6:0|proof:7:1)
        CONFIG_DIR="proof"
        CONFIG_PREFIX="proof"
        ;;
    *)
        echo "Invalid method/GPU/shard assignment: ${METHOD}:${GPU_ID}:${PROOF_SHARD}" >&2
        exit 2
        ;;
esac

COMMON_CONFIGS=(
    aircraft_B0_Inc10 aircraft_B50_Inc10
    cifar_B0_Inc10 cifar_B50_Inc10
    cars_B0_Inc10 cars_B50_Inc10
    inr_B0_Inc20 inr_B100_Inc20
    cub_B0_Inc20 cub_B100_Inc20
    ucf_B0_Inc10 ucf_B50_Inc10
    food_B0_Inc10 food_B50_Inc10
    sun_B0_Inc30 sun_B150_Inc30
    objectnet_B0_Inc20 objectnet_B100_Inc20
)

if [[ "${METHOD}" == "proof" ]]; then
    if [[ "${PROOF_SHARD}" == "0" ]]; then
        CONFIG_STEMS=("${COMMON_CONFIGS[@]:0:10}")
    else
        CONFIG_STEMS=("${COMMON_CONFIGS[@]:10}")
    fi
else
    CONFIG_STEMS=("${COMMON_CONFIGS[@]}")
fi

if [[ -n "${MMCL_CONFIG_STEMS:-}" ]]; then
    read -r -a CONFIG_STEMS <<< "${MMCL_CONFIG_STEMS}"
fi

export MMCL_DATA_ROOT="${DATA_ROOT}"
export MMCL_CLIP_PRETRAINED="${OPENAI_CHECKPOINT}"
export PYTHONUNBUFFERED=1

QUEUE_DIR="${MMCL_QUEUE_LOG_DIR:-${ROOT}/logs/OpenAI_CLIP_ViTB16/queue/seed${SEED}}"
QUEUE_TAG="${METHOD}_gpu${GPU_ID}"
if [[ -n "${PROOF_SHARD}" ]]; then
    QUEUE_TAG+="_shard${PROOF_SHARD}"
fi
QUEUE_LOG="${MMCL_QUEUE_LOG:-${QUEUE_DIR}/${QUEUE_TAG}.out}"
mkdir -p "${QUEUE_DIR}"
# Preserve the interactive stream while making every launcher invocation leave
# an auditable backbone-specific queue log.  Callers may override QUEUE_LOG.
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "Queue: method=${METHOD}, gpu=${GPU_ID}, seed=${SEED}, configs=${#CONFIG_STEMS[@]}"
echo "Checkpoint: ${OPENAI_CHECKPOINT}"
echo "Queue log: ${QUEUE_LOG}"

failed=0
for stem in "${CONFIG_STEMS[@]}"; do
    config_path="configs/${CONFIG_DIR}/${CONFIG_PREFIX}_${stem}.json"
    [[ -f "${ROOT}/${config_path}" ]] || {
        echo "Missing config: ${config_path}" >&2
        failed=1
        continue
    }

    echo "==== START ${METHOD} OpenAI ViT-B/16: ${config_path}, seed=${SEED}, GPU=${GPU_ID} ===="
    if [[ "${MMCL_DRY_RUN:-0}" == "1" ]]; then
        continue
    fi

    if (
        cd "${ROOT}"
        "${PYTHON_BIN}" scripts/run_openai_config.py \
            --config="${config_path}" --device="${GPU_ID}" --seed="${SEED}"
    ); then
        echo "==== FINISHED ${METHOD} OpenAI ViT-B/16: ${config_path}, seed=${SEED}, GPU=${GPU_ID} ===="
    else
        echo "==== FAILED ${METHOD} OpenAI ViT-B/16: ${config_path}, seed=${SEED}, GPU=${GPU_ID} ====" >&2
        failed=1
    fi
done

exit "${failed}"
