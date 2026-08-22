#!/usr/bin/env bash
# Wait for paused queue parents' current jobs, then stop and reconcile results.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_PIDS=("$@")
[[ "${#QUEUE_PIDS[@]}" -gt 0 ]] || {
  echo "usage: $0 <paused-queue-pid> [<paused-queue-pid> ...]" >&2
  exit 2
}

is_current_job_running() {
  local pid
  for pid in "${QUEUE_PIDS[@]}"; do
    if pgrep -g "${pid}" -f 'scripts/run_openai_config.py' >/dev/null; then
      return 0
    fi
  done
  return 1
}

while is_current_job_running; do
  echo "$(date --iso-8601=seconds) waiting for the four in-flight jobs"
  sleep 60
done

for pid in "${QUEUE_PIDS[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    # The queue parent is SIGSTOPped.  SIGKILL cannot wake it and therefore
    # cannot let it advance to the next job.
    kill -KILL "${pid}"
    echo "$(date --iso-8601=seconds) stopped paused queue parent ${pid}"
  fi
done

source /home/team/zhaohongwei/anaconda3/etc/profile.d/conda.sh
conda activate mmcl
[[ "${CONDA_DEFAULT_ENV:-}" == "mmcl" ]] || {
  echo "Failed to activate mmcl" >&2
  exit 2
}

mkdir -p "${ROOT}/.codex_work"
flock "${ROOT}/.codex_work/openai_workbook.lock" \
  python "${ROOT}/scripts/update_openai_native_baselines.py" --write
