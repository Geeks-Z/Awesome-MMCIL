#!/usr/bin/env bash
# Rerun the failed PromptFusion CIFAR-100 B0 Inc5 / seed 1993 experiment on GPU 1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export DATA_ROOT="${DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
export PYTHON_BIN="${PYTHON_BIN:-/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python}"
export PYTHONUNBUFFERED=1

# Keep this rerun's log and outputs separate so the failed GPU-5 record remains intact.
RUN_ROOT="${ROOT}/reruns/PromptFusion-cifar5-seed1993-gpu1"

cd "${ROOT}/PromptFusion-main"
exec "$PYTHON_BIN" main.py --config=./config/pf_cifar_5_5.json --seed=1993 \
  --data-root="$DATA_ROOT" --file-root="${RUN_ROOT}/PromptFusion-main"
