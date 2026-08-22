#!/usr/bin/env bash
# Run all currently blank LAION-400M workbook entries for one native method.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${METHOD:?Set METHOD to ranpac, fecam, area, or clg-cbm}"
GPU_ID="${GPU_ID:?Set GPU_ID to one of 1,2,4,5,6,7}"
JOB_SHARD_ID="${JOB_SHARD_ID:-0}"
NUM_JOB_SHARDS="${NUM_JOB_SHARDS:-1}"
JOB_CONFIG_STEMS="${JOB_CONFIG_STEMS:-}"

case "${METHOD}" in ranpac|fecam|area|clg-cbm) ;; *) echo "Unsupported method: ${METHOD}" >&2; exit 2 ;; esac
case "${GPU_ID}" in 1|2|4|5|6|7) ;; *) echo "Refusing reserved or unknown GPU ${GPU_ID}" >&2; exit 2 ;; esac
[[ "${JOB_SHARD_ID}" =~ ^[0-9]+$ && "${NUM_JOB_SHARDS}" =~ ^[1-9][0-9]*$ && "${JOB_SHARD_ID}" -lt "${NUM_JOB_SHARDS}" ]] \
  || { echo "Invalid job shard: ${JOB_SHARD_ID}/${NUM_JOB_SHARDS}" >&2; exit 2; }

source /home/team/zhaohongwei/anaconda3/etc/profile.d/conda.sh
conda activate mmcl
[[ "${CONDA_DEFAULT_ENV:-}" == "mmcl" ]] || { echo "Failed to activate mmcl" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MMCL_DEVICE=0
export MMCL_DATA_ROOT="/home/team/zhaohongwei/Dataset"
export MMCL_CLIP_PRETRAINED="/home/team/zhaohongwei/pretrained_models/open_clip_pytorch_model_laion400m_e32.bin"
export PYTHONUNBUFFERED=1

# AREA uses a high-worker dataloader; avoid exhausting the default per-process
# descriptor limit and use a sharing backend that does not retain worker FDs.
if [[ "${METHOD}" == "area" ]]; then
  ulimit -n 65535
  export MMCL_TORCH_SHARING_STRATEGY=file_system
fi

marker_root="${ROOT}/logs/OpenCLIP_LAION400M_ViTB16/queue/Workbook-Missing/${METHOD}/done"
mkdir -p "${marker_root}"
mapfile -t jobs < <(python "${ROOT}/scripts/workbook_missing_jobs.py" --method "${METHOD}")
if [[ -n "${JOB_CONFIG_STEMS}" ]]; then
  IFS=',' read -r -a requested_stems <<< "${JOB_CONFIG_STEMS}"
  filtered_jobs=()
  for job in "${jobs[@]}"; do
    IFS=$'\t' read -r _ config_path _ <<< "${job}"
    config_stem="$(basename "${config_path}" .json)"
    for requested_stem in "${requested_stems[@]}"; do
      if [[ "${config_stem}" == "${requested_stem}" ]]; then
        filtered_jobs+=("${job}")
        break
      fi
    done
  done
  jobs=("${filtered_jobs[@]}")
fi
all_job_count="${#jobs[@]}"
selected_jobs=()
for job_index in "${!jobs[@]}"; do
  if (( job_index % NUM_JOB_SHARDS == JOB_SHARD_ID )); then
    selected_jobs+=("${jobs[job_index]}")
  fi
done
jobs=("${selected_jobs[@]}")
echo "METHOD=${METHOD} physical_gpu=${GPU_ID} shard=${JOB_SHARD_ID}/${NUM_JOB_SHARDS} missing_jobs=${#jobs[@]}/${all_job_count}"
failures=0

for job in "${jobs[@]}"; do
  IFS=$'\t' read -r seed config protocol <<< "${job}"
  config_stem="$(basename "${config}" .json)"
  marker="${marker_root}/${config_stem}_seed${seed}.done"
  if [[ -s "${marker}" ]]; then
    echo "SKIP ${config_stem} seed=${seed}: marker exists"
    continue
  fi
  gpu_state="$(nvidia-smi -i "${GPU_ID}" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
  echo "START ${METHOD} ${protocol} seed=${seed} physical_gpu=${GPU_ID} state=${gpu_state}"
  if (cd "${ROOT}" && MMCL_SEEDS="${seed}" python main.py --config="${config}"); then
    printf 'completed_at=%s\nconfig=%s\nseed=%s\nphysical_gpu=%s\n' \
      "$(date --iso-8601=seconds)" "${config}" "${seed}" "${GPU_ID}" > "${marker}"
    echo "DONE ${METHOD} ${protocol} seed=${seed}"
  else
    ((failures += 1))
    echo "FAILED ${METHOD} ${protocol} seed=${seed}; continuing" >&2
  fi
done

echo "METHOD=${METHOD} finished failures=${failures}"
exit "${failures}"
