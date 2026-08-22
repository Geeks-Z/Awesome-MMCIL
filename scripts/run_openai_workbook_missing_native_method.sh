#!/usr/bin/env bash
# Run blank OpenAI CLIP ViT-B/16 workbook jobs for native MMCL methods.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:?Set GPU_ID to one of 1,2,4,5,6,7}"
JOB_SHARD_ID="${JOB_SHARD_ID:-0}"
NUM_JOB_SHARDS="${NUM_JOB_SHARDS:-1}"

case "${GPU_ID}" in 1|2|4|5|6|7) ;; *) echo "Refusing reserved or unknown GPU ${GPU_ID}" >&2; exit 2 ;; esac
[[ "${JOB_SHARD_ID}" =~ ^[0-9]+$ && "${NUM_JOB_SHARDS}" =~ ^[1-9][0-9]*$ && "${JOB_SHARD_ID}" -lt "${NUM_JOB_SHARDS}" ]] \
  || { echo "Invalid shard ${JOB_SHARD_ID}/${NUM_JOB_SHARDS}" >&2; exit 2; }

source /home/team/zhaohongwei/anaconda3/etc/profile.d/conda.sh
conda activate mmcl
[[ "${CONDA_DEFAULT_ENV:-}" == "mmcl" ]] || { echo "Failed to activate mmcl" >&2; exit 2; }

OPENAI_CHECKPOINT="/home/team/zhaohongwei/.cache/clip/ViT-B-16.pt"
[[ -f "${OPENAI_CHECKPOINT}" ]] || { echo "Missing OpenAI checkpoint: ${OPENAI_CHECKPOINT}" >&2; exit 2; }
ulimit -n 65535
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MMCL_DATA_ROOT="/home/team/zhaohongwei/Dataset"
export MMCL_CLIP_PRETRAINED="${OPENAI_CHECKPOINT}"
export PYTHONUNBUFFERED=1

marker_root="${ROOT}/logs/OpenAI_CLIP_ViTB16/queue/Workbook-Missing/native/done"
mkdir -p "${marker_root}"
mapfile -t all_jobs < <(python "${ROOT}/scripts/openai_workbook_missing_jobs.py")
jobs=()
for index in "${!all_jobs[@]}"; do
  if (( index % NUM_JOB_SHARDS == JOB_SHARD_ID )); then
    jobs+=("${all_jobs[index]}")
  fi
done
echo "METHODS=all-native physical_gpu=${GPU_ID} shard=${JOB_SHARD_ID}/${NUM_JOB_SHARDS} missing_jobs=${#jobs[@]}/${#all_jobs[@]}"

failures=0
for job in "${jobs[@]}"; do
  IFS=$'\t' read -r method seed config protocol <<< "${job}"
  config_stem="$(basename "${config}" .json)"
  marker="${marker_root}/${method}_${config_stem}_seed${seed}.done"
  if [[ -s "${marker}" ]]; then
    echo "SKIP ${config_stem} seed=${seed}: completion marker exists"
    continue
  fi
  state="$(nvidia-smi -i "${GPU_ID}" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
  echo "START ${method} ${protocol} seed=${seed} backbone=OpenAI_CLIP_ViTB16 physical_gpu=${GPU_ID} state=${state}"
  if (cd "${ROOT}" && python scripts/run_openai_config.py --config="${config}" --device=0 --seed="${seed}"); then
    printf 'completed_at=%s\nconfig=%s\nseed=%s\nphysical_gpu=%s\nbackbone=OpenAI_CLIP_ViTB16\n' \
      "$(date --iso-8601=seconds)" "${config}" "${seed}" "${GPU_ID}" > "${marker}"
    echo "DONE ${method} ${protocol} seed=${seed}"
  else
    ((failures += 1))
    echo "FAILED ${method} ${protocol} seed=${seed}; continuing" >&2
  fi
done

echo "METHODS=all-native finished failures=${failures}"
exit "${failures}"
