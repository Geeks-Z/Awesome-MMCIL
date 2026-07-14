#!/usr/bin/env bash
# Run one reproducible benchmark stream on the GPU selected by the launcher.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/public/home/hanlida/Dr.1/Dataset}"
CONFIGS=(
  pf_cifar_5_5.json pf_cifar_10_10.json pf_cifar_20_20.json
  pf_inr_5_5.json pf_inr_10_10.json pf_inr_20_20.json pf_inr_40_40.json
)

[[ -d "${DATA_ROOT}/cifar-100-python" ]] || { echo "Missing CIFAR-100 at ${DATA_ROOT}" >&2; exit 1; }
[[ -d "${DATA_ROOT}/imagenet-r/train" && -d "${DATA_ROOT}/imagenet-r/test" ]] || {
  echo "Missing ImageNet-R train/test at ${DATA_ROOT}/imagenet-r" >&2; exit 1;
}

cd "$ROOT"
run_experiment() {
  local config="$1" seed="$2"
  echo "==== PromptFusion config=${config} seed=${seed} GPU=${CUDA_VISIBLE_DEVICES:-unset} ===="
  python main.py --config="./config/${config}" --seed="$seed"
}

# Completed on the previous server (2026-07-14); retain as comments so they
# are never rerun when this script is moved to the new server.
# run_experiment pf_cifar_5_5.json 0
# run_experiment pf_cifar_10_10.json 0
# run_experiment pf_cifar_20_20.json 0

# The ImageNet-R 5-way run for seed 0 was interrupted and is intentionally
# retained here, followed by all configurations for the remaining seeds.
run_experiment pf_inr_5_5.json 0
run_experiment pf_inr_10_10.json 0
run_experiment pf_inr_20_20.json 0
run_experiment pf_inr_40_40.json 0

for seed in 42 1993 2026; do
  for config in "${CONFIGS[@]}"; do
    run_experiment "$config" "$seed"
  done
done
