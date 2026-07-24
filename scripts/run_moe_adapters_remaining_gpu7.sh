#!/usr/bin/env bash
# Immediately resume the sole incomplete MoE-Adapters run on GPU 7.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOE_ROOT="${ROOT}/MoE-Adapters4CL-MoE-Adapters/cil"
GPU_ID="${GPU_ID:-7}"
DATA_ROOT="${DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
PYTHON_BIN="${PYTHON_BIN:-/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python}"
CONFIG="cifar100_50_10"
SEED="2026"
EXPERIMENT="./experiments/class/${CONFIG}/seed_${SEED}"
METRICS="${MOE_ROOT}/${EXPERIMENT}/metrics.json"

[[ "${GPU_ID}" == "7" ]] || { echo "MoE-Adapters resume must use GPU 7, got ${GPU_ID}." >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing mmcl Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -d "${DATA_ROOT}/cifar-100-python" ]] || { echo "Missing CIFAR-100 at ${DATA_ROOT}" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

if [[ -s "${METRICS}" ]] && tail -n 1 "${METRICS}" | grep -q '"last"'; then
  echo "==== SKIP complete MoE-Adapters: ${CONFIG} seed=${SEED} ===="
  exit 0
fi

echo "==== START MoE-Adapters: ${CONFIG} seed=${SEED} GPU=${GPU_ID} ===="
cd "${MOE_ROOT}"
"${PYTHON_BIN}" main.py --config-path configs/class --config-name="${CONFIG}" \
  dataset_root="${DATA_ROOT}" class_order="class_orders/cifar100.yaml" +seed="${SEED}" \
  hydra.run.dir="${EXPERIMENT}"

tail -n 1 "${METRICS}" | grep -q '"last"' \
  || { echo "MoE-Adapters did not write final metrics." >&2; exit 1; }
echo "==== MoE-Adapters GPU-7 resume run finished ===="
