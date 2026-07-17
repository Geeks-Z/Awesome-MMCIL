#!/usr/bin/env bash
# Submit with: nohup bash scripts/run_promptfusion.sh > results/PromptFusion-GPU5.out 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-5}"
export DATA_ROOT="${DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
export PYTHON_BIN="${PYTHON_BIN:-/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python}"

cd "$ROOT"
bash PromptFusion-main/run_mmcl_seeds.sh
