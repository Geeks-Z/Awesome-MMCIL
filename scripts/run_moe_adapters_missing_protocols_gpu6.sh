#!/usr/bin/env bash

# Run the MoE-Adapters protocols absent from the benchmark table on physical GPU 6.
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
if [[ ! -x "${PYTHON_BIN}" || ! -d "${DATA_ROOT}/cifar-100-python" || ! -d "${DATA_ROOT}/imagenet-r" ]]; then
    echo "The mmcl environment or required server datasets are unavailable." >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

run_protocol() {
    local config_name="$1"
    local dataset_root="$2"
    local class_order="$3"
    local experiment_name="$4"

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
                dataset_root="${dataset_root}" \
                class_order="${class_order}" \
                +seed="${seed}" \
                hydra.run.dir="./experiments/class/${experiment_name}/seed_${seed}"
        ); then
            echo "$(date '+%F %T') FAILED ${config_name} seed=${seed}" >&2
        fi
    done
}

run_protocol "cifar100_0_10" "${DATA_ROOT}" "class_orders/cifar100.yaml" "cifar100_0_10"
run_protocol "imagenet_r_0_20" "${DATA_ROOT}/imagenet-r" "class_orders/imagenet_R.yaml" "imagenet_r_0_20"
