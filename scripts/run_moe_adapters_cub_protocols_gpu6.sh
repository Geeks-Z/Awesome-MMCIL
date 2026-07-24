#!/usr/bin/env bash

# Run the two CUB-200 CIL protocols on physical GPU 6.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOE_ROOT="${REPO_ROOT}/MoE-Adapters4CL-MoE-Adapters/cil"
DATA_ROOT="/home/team/zhaohongwei/Dataset"
PYTHON_BIN="/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python"
GPU_ID="${GPU_ID:-6}"
SEEDS=(1993 2026 0 42)

if [[ "${GPU_ID}" != "6" ]]; then
    echo "Refusing to run: this launcher is restricted to physical GPU 6." >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" || ! -d "${DATA_ROOT}/CUB_200_2011" ]]; then
    echo "The mmcl environment or the CUB-200 server dataset is unavailable." >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

run_protocol() {
    local config_name="$1"
    local experiment_name="$2"

    for seed in "${SEEDS[@]}"; do
        local metrics="${MOE_ROOT}/experiments/class/${experiment_name}/seed_${seed}/metrics.json"
        if [[ -s "${metrics}" ]] && grep -q '"last"' "${metrics}"; then
            echo "$(date '+%F %T') SKIP ${config_name} seed=${seed}: complete"
            continue
        fi

        echo "$(date '+%F %T') START ${config_name} seed=${seed} on GPU ${GPU_ID}"
        if ! (
            cd "${MOE_ROOT}"
            "${PYTHON_BIN}" main.py \
                --config-path configs/class \
                --config-name="${config_name}" \
                dataset_root="${DATA_ROOT}" \
                class_order="class_orders/cub200.yaml" \
                +seed="${seed}" \
                hydra.run.dir="./experiments/class/${experiment_name}/seed_${seed}"
        ); then
            echo "$(date '+%F %T') FAILED ${config_name} seed=${seed}" >&2
        fi
    done
}

run_protocol "cub200_20_20" "cub200_20_20"
run_protocol "cub200_100_20" "cub200_100_20"
