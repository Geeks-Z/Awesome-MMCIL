#!/usr/bin/env bash
# Run every still-queued MMCL benchmark on GPU 1, one experiment at a time.
# A failed experiment is recorded in the launcher log and does not stop the queue.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export DATA_ROOT="${DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
export PYTHON_BIN="${PYTHON_BIN:-/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python}"
export PYTHONUNBUFFERED=1

MOE_ROOT="${ROOT}/MoE-Adapters4CL-MoE-Adapters/cil"
PROMPT_ROOT="${ROOT}/PromptFusion-main"

has_moe_result() {
  local metrics="$1"
  [[ -s "$metrics" ]] && tail -n 1 "$metrics" | grep -q '"last"'
}

has_prompt_result() {
  local log="$1"
  [[ -s "$log" ]] && grep -q 'Average Accuracy (CNN top1):' "$log" && grep -q 'Last Accuracy:' "$log"
}

run_moe() {
  local config="$1" seed="$2"
  local experiment="./experiments/class/${config%.yaml}/seed_${seed}"
  local metrics="${MOE_ROOT}/${experiment}/metrics.json"
  if has_moe_result "$metrics"; then
    echo "==== SKIP complete MoE-Adapters: ${config} seed=${seed} ===="
    return
  fi

  echo "==== START MoE-Adapters: ${config} seed=${seed} GPU=${CUDA_VISIBLE_DEVICES} ===="
  if ! (
    cd "$MOE_ROOT"
    "$PYTHON_BIN" main.py --config-path configs/class --config-name="${config%.yaml}" \
      dataset_root="${DATA_ROOT}" class_order="class_orders/cifar100.yaml" +seed="$seed" \
      hydra.run.dir="$experiment"
  ); then
    echo "==== FAILED MoE-Adapters: ${config} seed=${seed}; continuing queue ====" >&2
  fi
}

run_prompt() {
  local config="$1" stem="$2" dataset_root="$3" seed="$4"
  local log="${ROOT}/logs/PromptFusion/${stem}_${seed}.log"
  if has_prompt_result "$log"; then
    echo "==== SKIP complete PromptFusion: ${config} seed=${seed} ===="
    return
  fi

  echo "==== START PromptFusion: ${config} seed=${seed} GPU=${CUDA_VISIBLE_DEVICES} ===="
  if ! (
    cd "$PROMPT_ROOT"
    "$PYTHON_BIN" main.py --config="./config/${config}" --seed="$seed" \
      --data-root="$dataset_root" --file-root="$PROMPT_ROOT"
  ); then
    echo "==== FAILED PromptFusion: ${config} seed=${seed}; continuing queue ====" >&2
  fi
}

# MoE-Adapters CIFAR-100: 10 experiments not covered by the ImageNet-R GPU-1 run.
for seed in 1993 2026; do
  for config in cifar100_5_5.yaml cifar100_10_10.yaml cifar100_20_20.yaml \
                cifar100_50_5.yaml cifar100_50_10.yaml; do
    run_moe "$config" "$seed"
  done
done

# PromptFusion: the CIFAR B0 Inc5 / seed 1993 rerun has completed already.
for config in pf_cifar_10_10.json pf_cifar_20_20.json; do
  case "$config" in
    pf_cifar_10_10.json) stem="cifar224_vit_b16_0_10" ;;
    pf_cifar_20_20.json) stem="cifar224_vit_b16_0_20" ;;
  esac
  run_prompt "$config" "$stem" "$DATA_ROOT" 1993
done
for config in pf_inr_5_5.json pf_inr_10_10.json pf_inr_20_20.json pf_inr_40_40.json; do
  case "$config" in
    pf_inr_5_5.json) stem="imagenetr_vit_b16_0_5" ;;
    pf_inr_10_10.json) stem="imagenetr_vit_b16_0_10" ;;
    pf_inr_20_20.json) stem="imagenetr_vit_b16_0_20" ;;
    pf_inr_40_40.json) stem="imagenetr_vit_b16_0_40" ;;
  esac
  run_prompt "$config" "$stem" "${DATA_ROOT}/imagenet-r" 1993
done
for config in pf_cifar_5_5.json pf_cifar_10_10.json pf_cifar_20_20.json; do
  case "$config" in
    pf_cifar_5_5.json) stem="cifar224_vit_b16_0_5" ;;
    pf_cifar_10_10.json) stem="cifar224_vit_b16_0_10" ;;
    pf_cifar_20_20.json) stem="cifar224_vit_b16_0_20" ;;
  esac
  run_prompt "$config" "$stem" "$DATA_ROOT" 2026
done
for config in pf_inr_5_5.json pf_inr_10_10.json pf_inr_20_20.json pf_inr_40_40.json; do
  case "$config" in
    pf_inr_5_5.json) stem="imagenetr_vit_b16_0_5" ;;
    pf_inr_10_10.json) stem="imagenetr_vit_b16_0_10" ;;
    pf_inr_20_20.json) stem="imagenetr_vit_b16_0_20" ;;
    pf_inr_40_40.json) stem="imagenetr_vit_b16_0_40" ;;
  esac
  run_prompt "$config" "$stem" "${DATA_ROOT}/imagenet-r" 2026
done

echo "==== MMCL GPU-1 queue finished; inspect final metrics before publishing. ===="
