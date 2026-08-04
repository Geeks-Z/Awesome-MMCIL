#!/usr/bin/env bash

# Run the two CUB-200 CIL protocols on physical GPU 7.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_ROOT="${REPO_ROOT}/PromptFusion-main"
DATA_ROOT="/home/team/zhaohongwei/Dataset"
PYTHON_BIN="/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python"
GPU_ID="${GPU_ID:-7}"
SEEDS=(1993 2026 0 42)

if [[ "${GPU_ID}" != "7" ]]; then
    echo "Refusing to run: this launcher is restricted to physical GPU 7." >&2
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
    local log_stem="$2"

    for seed in "${SEEDS[@]}"; do
        local log_file="${REPO_ROOT}/logs/OpenAI_CLIP_ViTB16/PromptFusion/${log_stem}_${seed}.log"
        if [[ -s "${log_file}" ]] && grep -q 'Last Accuracy:' "${log_file}"; then
            echo "$(date '+%F %T') SKIP ${config_name} seed=${seed}: complete"
            continue
        fi

        echo "$(date '+%F %T') START ${config_name} seed=${seed} on GPU ${GPU_ID}"
        if ! (
            cd "${PROMPT_ROOT}"
            "${PYTHON_BIN}" main.py \
                --config="config/${config_name}.json" \
                --seed="${seed}" \
                --data-root="${DATA_ROOT}" \
                --file-root="${PROMPT_ROOT}"
        ); then
            echo "$(date '+%F %T') FAILED ${config_name} seed=${seed}" >&2
        fi
    done
}

run_protocol "pf_cub_20_20" "cub_vit_b16_0_20"
run_protocol "pf_cub_100_20" "cub_vit_b16_100_20"
