#!/usr/bin/env bash
# Submit all blank, executable OpenAI CLIP ViT-B/16 workbook jobs on GPUs 4--7.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS=(4 5 6 7)
QUEUE_ROOT="${ROOT}/logs/OpenAI_CLIP_ViTB16/queue/Workbook-Missing/native"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

for gpu in "${GPUS[@]}"; do
  case "${gpu}" in
    4|5|6|7) ;;
    *) echo "Refusing GPU ${gpu}; this launcher permits only 4, 5, 6, and 7." >&2; exit 2 ;;
  esac
  nvidia-smi -i "${gpu}" \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
done

mkdir -p "${QUEUE_ROOT}"
for shard in "${!GPUS[@]}"; do
  gpu="${GPUS[${shard}]}"
  output="${QUEUE_ROOT}/shard${shard}-gpu${gpu}-${TIMESTAMP}.out"
  pid_file="${QUEUE_ROOT}/shard${shard}-gpu${gpu}-${TIMESTAMP}.pid"
  nohup setsid env GPU_ID="${gpu}" JOB_SHARD_ID="${shard}" NUM_JOB_SHARDS="${#GPUS[@]}" \
    bash "${ROOT}/scripts/run_openai_workbook_missing_native_method.sh" \
    > "${output}" 2>&1 < /dev/null &
  pid="$!"
  printf '%s\n' "${pid}" > "${pid_file}"
  echo "Submitted shard=${shard} gpu=${gpu} pid=${pid} log=${output}"
done
