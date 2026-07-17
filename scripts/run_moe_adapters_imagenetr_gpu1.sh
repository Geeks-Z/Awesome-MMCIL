#!/usr/bin/env bash
# Resume only unfinished MoE-Adapters ImageNet-R experiments on GPU 1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export DATA_ROOT="${DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
export PYTHON_BIN="${PYTHON_BIN:-/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python}"
export PYTHONUNBUFFERED=1

cd "$ROOT/MoE-Adapters4CL-MoE-Adapters/cil"

run_experiment() {
  local config="$1" seed="$2"
  local experiment="./experiments/class/${config%.yaml}/seed_${seed}"
  local metrics="${experiment}/metrics.json"

  if [[ -s "$metrics" ]] && tail -n 1 "$metrics" | grep -q '"last"'; then
    echo "==== SKIP complete: config=${config} seed=${seed} ===="
    return
  fi

  echo "==== MoE-Adapters ImageNet-R: config=${config} seed=${seed} GPU=${CUDA_VISIBLE_DEVICES} ===="
  "$PYTHON_BIN" main.py --config-path configs/class --config-name="${config%.yaml}" \
    dataset_root="${DATA_ROOT}/imagenet-r" class_order="class_orders/imagenet_R.yaml" +seed="$seed" \
    hydra.run.dir="$experiment"
}

# seed=42: resume after the CUDA OOM on the former 24-GB GPU.
run_experiment imagenet_r_100_10.yaml 42
run_experiment imagenet_r_100_20.yaml 42

# The previous launcher stopped before starting these seed groups.
for seed in 1993 2026; do
  for config in imagenet_r_10_10.yaml imagenet_r_20_20.yaml imagenet_r_40_40.yaml \
                imagenet_r_100_10.yaml imagenet_r_100_20.yaml; do
    run_experiment "$config" "$seed"
  done
done
