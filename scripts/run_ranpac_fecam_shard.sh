#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:?Set GPU_ID to one of 1,2,4,5,6,7}"
SHARD_ID="${SHARD_ID:?Set SHARD_ID in [0, NUM_SHARDS)}"
NUM_SHARDS="${NUM_SHARDS:-6}"

case "${GPU_ID}" in
  1|2|4|5|6|7) ;;
  *) echo "Refusing reserved or unknown GPU ${GPU_ID}; allowed: 1,2,4,5,6,7" >&2; exit 2 ;;
esac
if (( SHARD_ID < 0 || SHARD_ID >= NUM_SHARDS )); then
  echo "Invalid shard ${SHARD_ID}/${NUM_SHARDS}" >&2
  exit 2
fi

source /home/team/zhaohongwei/anaconda3/etc/profile.d/conda.sh
conda activate mmcl
if [[ "${CONDA_DEFAULT_ENV:-}" != "mmcl" ]]; then
  echo "Failed to activate the mmcl conda environment" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MMCL_DEVICE=0
export MMCL_DATA_ROOT="/home/team/zhaohongwei/Dataset"
export MMCL_CLIP_PRETRAINED="/home/team/zhaohongwei/pretrained_models/open_clip_pytorch_model_laion400m_e32.bin"
export PYTHONUNBUFFERED=1

marker_root="${ROOT}/logs/OpenCLIP_LAION400M_ViTB16/queue/RanPAC-FeCAM/done"
mkdir -p "${marker_root}"

mapfile -t ranpac_configs < <(
  find "${ROOT}/configs/ranpac" -maxdepth 1 -type f -name '*.json' | sort
)
mapfile -t fecam_configs < <(
  find "${ROOT}/configs/fecam" -maxdepth 1 -type f -name '*.json' | sort
)
if (( ${#ranpac_configs[@]} != ${#fecam_configs[@]} )); then
  echo "RanPAC/FeCAM config counts differ" >&2
  exit 2
fi
configs=()
for config_index in "${!ranpac_configs[@]}"; do
  configs+=(
    "${fecam_configs[${config_index}]}"
    "${ranpac_configs[${config_index}]}"
  )
done
seeds=(0 42 1993)
job_index=0
failures=0

for config in "${configs[@]}"; do
  method="$(basename "$(dirname "${config}")")"
  config_stem="$(basename "${config}" .json)"
  for seed in "${seeds[@]}"; do
    if (( job_index % NUM_SHARDS != SHARD_ID )); then
      ((job_index += 1))
      continue
    fi

    marker="${marker_root}/${config_stem}_seed${seed}.done"
    if [[ -s "${marker}" ]]; then
      echo "SKIP ${method} ${config_stem} seed=${seed}: completion marker exists"
      ((job_index += 1))
      continue
    fi

    gpu_state="$(nvidia-smi -i "${GPU_ID}" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
    echo "START ${method} ${config_stem} seed=${seed} physical_gpu=${GPU_ID} state=${gpu_state}"
    if (
      cd "${ROOT}"
      MMCL_SEEDS="${seed}" python main.py --config="${config}"
    ); then
      printf 'completed_at=%s\nconfig=%s\nseed=%s\nphysical_gpu=%s\n' \
        "$(date --iso-8601=seconds)" "${config}" "${seed}" "${GPU_ID}" > "${marker}"
      echo "DONE ${method} ${config_stem} seed=${seed} physical_gpu=${GPU_ID}"
    else
      ((failures += 1))
      echo "FAILED ${method} ${config_stem} seed=${seed} physical_gpu=${GPU_ID}; queue continues" >&2
    fi
    ((job_index += 1))
  done
done

echo "Shard ${SHARD_ID}/${NUM_SHARDS} finished with ${failures} failures"
exit "${failures}"
