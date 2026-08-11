#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
read -r -a gpus <<< "${GPU_IDS:-1 2 4 5 6 7}"
num_shards="${#gpus[@]}"
queue_root="${ROOT}/logs/OpenAI_CLIP_ViTB16/queue/RanPAC-FeCAM-seed1993"
mkdir -p "${queue_root}"

for shard_id in "${!gpus[@]}"; do
  gpu_id="${gpus[${shard_id}]}"
  case "${gpu_id}" in
    1|2|4|5|6|7) ;;
    *) echo "Refusing reserved or unknown GPU ${gpu_id}; allowed: 1,2,4,5,6,7" >&2; exit 2 ;;
  esac
  nvidia-smi -i "${gpu_id}" \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
done

timestamp="$(date +%Y%m%d_%H%M%S)"
for shard_id in "${!gpus[@]}"; do
  gpu_id="${gpus[${shard_id}]}"
  output="${queue_root}/shard${shard_id}-gpu${gpu_id}-${timestamp}.out"
  pid_file="${queue_root}/shard${shard_id}-gpu${gpu_id}-${timestamp}.pid"
  nohup setsid env GPU_ID="${gpu_id}" SHARD_ID="${shard_id}" NUM_SHARDS="${num_shards}" MMCL_SEED=1993 \
    bash "${ROOT}/scripts/run_ranpac_fecam_openai_shard.sh" \
    > "${output}" 2>&1 < /dev/null &
  pid="$!"
  printf '%s\n' "${pid}" > "${pid_file}"
  echo "Submitted shard=${shard_id} gpu=${gpu_id} pid=${pid} log=${output}"
done
