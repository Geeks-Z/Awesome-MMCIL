#!/usr/bin/env bash
# Resume the missing PromptFusion runs in MMCL_Baselines.xlsx sheet order:
# Seed1993, then Seed2026. Submit with GPU_ID=5 only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_ROOT="${ROOT}/PromptFusion-main"
GPU_ID="${GPU_ID:-5}"
DATA_ROOT="${DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
PYTHON_BIN="${PYTHON_BIN:-/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python}"

[[ "${GPU_ID}" == "5" ]] || { echo "PromptFusion resume must use GPU 5, got ${GPU_ID}." >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing mmcl Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${DATA_ROOT}/cifar-100-python" ]] || { echo "Missing CIFAR-100 at ${DATA_ROOT}" >&2; exit 2; }
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
  local config="$1" stem="$2" dataset_root="$3" seed="$4"
  local log_file="${ROOT}/logs/PromptFusion/${stem}_${seed}.log"
  if has_final_result "${log_file}"; then
    echo "==== SKIP complete PromptFusion: ${config} seed=${seed} ===="
    return
  fi

  echo "==== START PromptFusion: ${config} seed=${seed} GPU=${GPU_ID} ===="
  if ! (
    cd "${PROMPT_ROOT}"
    "${PYTHON_BIN}" main.py --config="./config/${config}" --seed="${seed}" \
      --data-root="${dataset_root}" --file-root="${PROMPT_ROOT}"
  ); then
    echo "==== FAILED PromptFusion: ${config} seed=${seed}; continuing queue ====" >&2
  fi
}

# Seed1993: CIFAR B0/Inc5 is complete, so start at Inc10.
for config in pf_cifar_10_10.json pf_cifar_20_20.json; do
  case "${config}" in
    pf_cifar_10_10.json) stem="cifar224_vit_b16_0_10" ;;
    pf_cifar_20_20.json) stem="cifar224_vit_b16_0_20" ;;
  esac
  run_experiment "${config}" "${stem}" "${DATA_ROOT}" 1993
done
for config in pf_inr_5_5.json pf_inr_10_10.json pf_inr_20_20.json pf_inr_40_40.json; do
  case "${config}" in
    pf_inr_5_5.json) stem="imagenetr_vit_b16_0_5" ;;
    pf_inr_10_10.json) stem="imagenetr_vit_b16_0_10" ;;
    pf_inr_20_20.json) stem="imagenetr_vit_b16_0_20" ;;
    pf_inr_40_40.json) stem="imagenetr_vit_b16_0_40" ;;
  esac
  run_experiment "${config}" "${stem}" "${DATA_ROOT}/imagenet-r" 1993
done

# Seed2026 follows the next sheet and needs all seven configurations.
for config in pf_cifar_5_5.json pf_cifar_10_10.json pf_cifar_20_20.json; do
  case "${config}" in
    pf_cifar_5_5.json) stem="cifar224_vit_b16_0_5" ;;
    pf_cifar_10_10.json) stem="cifar224_vit_b16_0_10" ;;
    pf_cifar_20_20.json) stem="cifar224_vit_b16_0_20" ;;
  esac
  run_experiment "${config}" "${stem}" "${DATA_ROOT}" 2026
done
for config in pf_inr_5_5.json pf_inr_10_10.json pf_inr_20_20.json pf_inr_40_40.json; do
  case "${config}" in
    pf_inr_5_5.json) stem="imagenetr_vit_b16_0_5" ;;
    pf_inr_10_10.json) stem="imagenetr_vit_b16_0_10" ;;
    pf_inr_20_20.json) stem="imagenetr_vit_b16_0_20" ;;
    pf_inr_40_40.json) stem="imagenetr_vit_b16_0_40" ;;
  esac
  run_experiment "${config}" "${stem}" "${DATA_ROOT}/imagenet-r" 2026
done

echo "==== PromptFusion GPU-5 resume queue finished ===="
