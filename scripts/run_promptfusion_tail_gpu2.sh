#!/usr/bin/env bash
# Run the tail of the remaining PromptFusion Seed2026 queue on GPU 2.
# These two protocols are reached last by the active GPU-5 queue.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_ROOT="${ROOT}/PromptFusion-main"
GPU_ID="${GPU_ID:-2}"
DATA_ROOT="${DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
PYTHON_BIN="${PYTHON_BIN:-/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python}"
SEED="2026"

[[ "${GPU_ID}" == "2" ]] || { echo "PromptFusion tail queue must use GPU 2, got ${GPU_ID}." >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing mmcl Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${DATA_ROOT}/imagenet-r/train" && -d "${DATA_ROOT}/imagenet-r/test" ]] || {
  echo "Missing ImageNet-R train/test under ${DATA_ROOT}/imagenet-r" >&2; exit 2;
}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

has_final_result() {
  local log_file="$1"
  [[ -s "${log_file}" ]] \
    && grep -q 'Average Accuracy (CNN top1):' "${log_file}" \
    && grep -q 'Last Accuracy:' "${log_file}"
}

run_experiment() {
  local config="$1" stem="$2"
  local log_file="${ROOT}/logs/OpenAI_CLIP_ViTB16/PromptFusion/${stem}_${SEED}.log"
  if has_final_result "${log_file}"; then
    echo "==== SKIP complete PromptFusion: ${config} seed=${SEED} ===="
    return
  fi

  echo "==== START PromptFusion: ${config} seed=${SEED} GPU=${GPU_ID} ===="
  (
    cd "${PROMPT_ROOT}"
    "${PYTHON_BIN}" main.py --config="./config/${config}" --seed="${SEED}" \
      --data-root="${DATA_ROOT}/imagenet-r" --file-root="${PROMPT_ROOT}"
  )
}

run_experiment pf_inr_40_40.json imagenetr_vit_b16_0_40
run_experiment pf_inr_20_20.json imagenetr_vit_b16_0_20
echo "==== PromptFusion GPU-2 tail queue finished ===="
