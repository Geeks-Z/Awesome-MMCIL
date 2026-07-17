#!/usr/bin/env bash
# Submit with: nohup bash scripts/run_moe_adapters.sh > results/MoE-Adapters-GPU6.out 2>&1 &
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-6}"
export DATA_ROOT="${DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
export PYTHON_BIN="${PYTHON_BIN:-/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python}"

cd "$ROOT"
bash MoE-Adapters4CL-MoE-Adapters/cil/run_mmcl_seeds.sh
