#!/usr/bin/env bash
set -euo pipefail

SCRIPTS=(
  "scripts/run_TAIL_16shot.sh"
  "scripts/run_TAIL_16shot_order2.sh"
  "scripts/run_TAIL_fullshot.sh"
  "scripts/run_TAIL_fullshot_order2.sh"
)

for s in "${SCRIPTS[@]}"; do
  echo "======== Running \\\`$s\\\` ========"
  if [ ! -f "$s" ]; then
    echo "Error: \\\`$s\\\` not found." >&2
    exit 1
  fi
  if [ ! -x "$s" ]; then
    echo "Making \\\`$s\\\` executable..."
    chmod +x "$s" || true
  fi
  bash "$s"
done

echo "All runs finished."