#!/usr/bin/env bash

# Delay a launcher until the current queue process has exited.
set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <pid-to-wait-for> <launcher> [launcher arguments...]" >&2
    exit 2
fi

wait_pid="$1"
shift

while kill -0 "${wait_pid}" 2>/dev/null; do
    echo "$(date '+%F %T') waiting for queue PID ${wait_pid}"
    sleep 60
done

echo "$(date '+%F %T') queue PID ${wait_pid} exited; starting $*"
exec "$@"
