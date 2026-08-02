#!/usr/bin/env bash

# Run a command only after the requested physical GPU is genuinely idle.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <gpu-index> <command> [arguments...]" >&2
    exit 2
fi

gpu_index="$1"
shift

while true; do
    status=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
        --format=csv,noheader,nounits | awk -F',' -v gpu="$gpu_index" '
            $1 + 0 == gpu {
                gsub(/ /, "", $2); gsub(/ /, "", $3); print $2 "," $3
            }')
    utilization=${status%,*}
    memory=${status#*,}

    if [[ -n "$utilization" && "$utilization" -le 5 && "$memory" -le 500 ]]; then
        echo "$(date '+%F %T') GPU ${gpu_index} is idle; starting $*"
        exec "$@"
    fi

    echo "$(date '+%F %T') waiting for GPU ${gpu_index}: utilization=${utilization:-unknown}% memory=${memory:-unknown}MiB"
    sleep 300
done
