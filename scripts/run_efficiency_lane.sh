#!/usr/bin/env bash

# Run one serial efficiency-benchmark lane on a dedicated GPU.
set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <gpu> <config> [config ...]" >&2
    exit 2
fi

gpu="$1"
shift
run_suffix="${MMCL_RUN_SUFFIX:-}"

if [[ -n "${run_suffix}" && "${run_suffix}" != _* ]]; then
    echo "MMCL_RUN_SUFFIX must be empty or start with an underscore" >&2
    exit 2
fi

project_root="/public/home/hanlida/Dr.1/Code/Research/Awesome-MMCL"
data_root="/public/home/hanlida/Dr.1/Dataset"
checkpoint="/public/home/hanlida/Dr.1/pretrained_models/open_clip_pytorch_model_laion400m_e32.bin"
raw_dir="${project_root}/results/efficiency/raw"
log_dir="${project_root}/logs/efficiency/baseline"
status_file="${project_root}/results/efficiency/lane_gpu${gpu}${run_suffix}.tsv"

mkdir -p "${raw_dir}" "${log_dir}"
cd "${project_root}" || exit 1

for config in "$@"; do
    stem="$(basename "${config}" .json)"
    output="${raw_dir}/${stem}_seed1993${run_suffix}.json"
    log="${log_dir}/${stem}_seed1993${run_suffix}.log"

    if [[ -s "${output}" ]]; then
        printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "${gpu}" "SKIP" "${stem}" >> "${status_file}"
        continue
    fi

    printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "${gpu}" "START" "${stem}" >> "${status_file}"
    MMCL_DATA_ROOT="${data_root}" \
    MMCL_CLIP_PRETRAINED="${checkpoint}" \
    MMCL_DEVICE="${gpu}" \
    MMCL_SEEDS="1993" \
    MMCL_PROFILE_OUT="${output}" \
    MMCL_PROFILE_INFER_REPEATS="3" \
        python main.py --config "${config}" > "${log}" 2>&1
    status=$?

    if [[ ${status} -eq 0 && -s "${output}" ]]; then
        printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "${gpu}" "DONE" "${stem}" >> "${status_file}"
    else
        printf '%s\t%s\tFAIL(%s)\t%s\n' "$(date -Is)" "${gpu}" "${status}" "${stem}" >> "${status_file}"
    fi
done
