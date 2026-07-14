#!/usr/bin/env bash
# MoE-Adapters supports CIFAR-100 and ImageNet-R from the shared benchmark data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/public/home/hanlida/Dr.1/Dataset}"
CONFIGS=(
  cifar100_5_5.yaml cifar100_10_10.yaml cifar100_20_20.yaml
  cifar100_50_5.yaml cifar100_50_10.yaml
  imagenet_r_10_10.yaml imagenet_r_20_20.yaml imagenet_r_40_40.yaml
  imagenet_r_100_10.yaml imagenet_r_100_20.yaml
)

[[ -d "${DATA_ROOT}/cifar-100-python" ]] || { echo "Missing CIFAR-100 at ${DATA_ROOT}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/imagenet-r/train" && -d "${DATA_ROOT}/imagenet-r/test" ]] || {
  echo "Missing ImageNet-R train/test at ${DATA_ROOT}/imagenet-r" >&2; exit 1;
}

cd "$ROOT"
run_experiment() {
  local config="$1" seed="$2" order data_root experiment
  case "$config" in
    cifar*) order="class_orders/cifar100.yaml"; data_root="$DATA_ROOT" ;;
    imagenet*) order="class_orders/imagenet_R.yaml"; data_root="${DATA_ROOT}/imagenet-r" ;;
  esac
  experiment="./experiments/class/${config%.yaml}/seed_${seed}"
  echo "==== MoE-Adapters config=${config} seed=${seed} GPU=${CUDA_VISIBLE_DEVICES:-unset} ===="
  python main.py --config-path configs/class --config-name="${config%.yaml}" \
    dataset_root="$data_root" class_order="$order" +seed="$seed" \
    hydra.run.dir="$experiment"
}

# Completed on the previous server (2026-07-14); retain as comments so they
# are never rerun when this script is moved to the new server.
# for config in "${CONFIGS[@]}"; do run_experiment "$config" 0; done
# run_experiment cifar100_5_5.yaml 42
# run_experiment cifar100_10_10.yaml 42
# run_experiment cifar100_20_20.yaml 42
# run_experiment cifar100_50_5.yaml 42
# run_experiment cifar100_50_10.yaml 42
# run_experiment imagenet_r_10_10.yaml 42

# The next seed=42 ImageNet-R configuration was interrupted and is retained.
run_experiment imagenet_r_20_20.yaml 42
run_experiment imagenet_r_40_40.yaml 42
run_experiment imagenet_r_100_10.yaml 42
run_experiment imagenet_r_100_20.yaml 42

for seed in 1993 2026; do
  for config in "${CONFIGS[@]}"; do
    run_experiment "$config" "$seed"
  done
done
