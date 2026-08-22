#!/usr/bin/env bash
# Finish the active GPU-6 AREA task, then transfer only its remaining jobs to GPU 7.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRENT_PID="${CURRENT_PID:?Set CURRENT_PID to the active AREA Python PID}"
SOURCE_PGID="${SOURCE_PGID:?Set SOURCE_PGID to the GPU-6 queue process group}"
TARGET_JOBS="${TARGET_JOBS:?Set TARGET_JOBS as config_stem:seed pairs}"
TARGET_STEMS="$(printf '%s' "${TARGET_JOBS}" | sed -E 's/:[0-9]+//g')"
POLL_SECONDS="${POLL_SECONDS:-20}"
QUEUE_ROOT="${ROOT}/logs/OpenCLIP_LAION400M_ViTB16/queue/Workbook-Missing"

echo "Waiting for active GPU-6 worker pid=${CURRENT_PID} before handoff."
while kill -0 "${CURRENT_PID}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

# Prevent the original sequential queue from starting another task during handoff.
kill -STOP -- "-${SOURCE_PGID}" 2>/dev/null || true
sleep 1

source /home/team/zhaohongwei/anaconda3/etc/profile.d/conda.sh
conda activate mmcl
[[ "${CONDA_DEFAULT_ENV:-}" == "mmcl" ]] || { echo "Failed to activate mmcl" >&2; exit 2; }

cd "${ROOT}"
python scripts/update_laion_native_baselines.py --write

quarantine_dir="${QUEUE_ROOT}/incomplete-during-gpu6-handoff-$(date +%Y%m%d_%H%M%S)"
mkdir -p "${quarantine_dir}"
mapfile -t incomplete_paths < <(TARGET_JOBS="${TARGET_JOBS}" python - <<'PY'
import json
import os
from pathlib import Path
from scripts.update_laion_native_baselines import parse_log

root = Path.cwd()
area_root = (root / 'logs/OpenCLIP_LAION400M_ViTB16/AREA').resolve()
for item in os.environ['TARGET_JOBS'].split(','):
    stem, seed = item.rsplit(':', 1)
    config = json.loads((root / 'configs/area' / f'{stem}.json').read_text())
    base = 0 if int(config['init_cls']) == int(config['increment']) else int(config['init_cls'])
    path = (area_root / f"{config['dataset']}_clip_{base}_{config['increment']}_{seed}.log").resolve()
    if path.parent != area_root:
        raise SystemExit(f'unsafe path: {path}')
    if path.exists() and parse_log('AREA', path) is None:
        print(path)
PY
)
for log_path in "${incomplete_paths[@]}"; do
  case "${log_path}" in "${ROOT}/logs/OpenCLIP_LAION400M_ViTB16/AREA/"*.log) ;; *) echo "Refusing unsafe path: ${log_path}" >&2; exit 2 ;; esac
  mv -- "${log_path}" "${quarantine_dir}/"
done

# The source queue is no longer needed. Resume it only to deliver termination.
kill -CONT -- "-${SOURCE_PGID}" 2>/dev/null || true
kill -TERM -- "-${SOURCE_PGID}" 2>/dev/null || true

handoff_log="${QUEUE_ROOT}/AREA-handoff-gpu7-$(date +%Y%m%d_%H%M%S).out"
setsid nohup env METHOD=area GPU_ID=7 JOB_CONFIG_STEMS="${TARGET_STEMS}" \
  bash scripts/run_workbook_missing_native_method.sh >"${handoff_log}" 2>&1 < /dev/null &
echo "Handoff submitted pid=$! gpu=7 targets=${TARGET_JOBS} log=${handoff_log} quarantine=${quarantine_dir}"
