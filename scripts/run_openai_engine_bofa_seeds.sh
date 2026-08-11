#!/usr/bin/env bash
# Run one native-MMCL method on one explicitly assigned physical GPU.
# Usage: bash scripts/run_openai_engine_bofa_seeds.sh <engine|bofa> <gpu>

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${1:?usage: $0 <engine|bofa> <gpu>}"
GPU_ID="${2:?usage: $0 <engine|bofa> <gpu>}"
DATA_ROOT="/home/team/zhaohongwei/Dataset"
OPENAI_CHECKPOINT="/home/team/zhaohongwei/pretrained_models/open_clip_pytorch_model.bin"
SEEDS=(0 42 2026)

case "${METHOD}:${GPU_ID}" in
    engine:6|bofa:7) ;;
    *)
        echo "Only ENGINE on GPU 6 and BOFA on GPU 7 are permitted." >&2
        exit 2
        ;;
esac

[[ -d "${DATA_ROOT}" ]] || { echo "Missing dataset root: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${OPENAI_CHECKPOINT}" ]] || { echo "Missing OpenAI checkpoint: ${OPENAI_CHECKPOINT}" >&2; exit 2; }

source /home/team/zhaohongwei/anaconda3/etc/profile.d/conda.sh
conda activate mmcl

cd "${ROOT}" || exit 1
unset CUDA_VISIBLE_DEVICES
export MMCL_DATA_ROOT="${DATA_ROOT}"
export MMCL_CLIP_PRETRAINED="${OPENAI_CHECKPOINT}"
export PYTHONUNBUFFERED=1

mapfile -t CONFIGS < <(
    find "configs/${METHOD}" -maxdepth 1 -type f -name "${METHOD}_*.json" \
        ! -name '*_seed*_gpu*.json' -printf '%f\n' | sort
)
[[ ${#CONFIGS[@]} -eq 18 ]] || {
    echo "Expected 18 standard ${METHOD} configurations, found ${#CONFIGS[@]}." >&2
    exit 2
}

is_complete() {
    python - "${1}" "${2}" "${METHOD^^}" <<'PY'
import ast
import json
import re
import sys
from pathlib import Path

config_path, seed_text, method = sys.argv[1:]
seed = int(seed_text)
with open(config_path) as config_file:
    config = json.load(config_file)

initial = 0 if config["init_cls"] == config["increment"] else config["init_cls"]
expected_tasks = 10 if initial == 0 else 6
log = Path("logs/OpenAI_CLIP_ViTB16") / method / (
    f"{config['dataset']}_openai_clip_{initial}_{config['increment']}_{seed}.log"
)
if not log.is_file():
    raise SystemExit(1)

content = log.read_text(errors="replace")
curves = re.findall(r"CNN top1 curve:\s*(\[[^\n]+\])", content)
logged_seeds = {int(value) for value in re.findall(r"\[trainer\.py\] => seed:\s*(\d+)", content)}
if not curves or (logged_seeds and logged_seeds != {seed}):
    raise SystemExit(1)
try:
    curve = ast.literal_eval(curves[-1])
except (SyntaxError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if isinstance(curve, list) and len(curve) == expected_tasks else 1)
PY
}

for seed in "${SEEDS[@]}"; do
    for name in "${CONFIGS[@]}"; do
        config="configs/${METHOD}/${name}"
        if is_complete "${config}" "${seed}"; then
            echo "==== SKIP complete: ${name}, seed=${seed}, GPU=${GPU_ID} ===="
            continue
        fi

        echo "==== START ${METHOD^^} OpenAI CLIP ViT-B/16: ${name}, seed=${seed}, GPU=${GPU_ID} ===="
        if python scripts/run_openai_config.py --config "${config}" --device "${GPU_ID}" --seed "${seed}"; then
            echo "==== FINISHED ${METHOD^^} OpenAI CLIP ViT-B/16: ${name}, seed=${seed}, GPU=${GPU_ID} ===="
        else
            echo "==== FAILED ${METHOD^^} OpenAI CLIP ViT-B/16: ${name}, seed=${seed}, GPU=${GPU_ID}; continuing ====" >&2
        fi
    done
done

echo "==== ${METHOD^^} OpenAI CLIP ViT-B/16 queue complete on GPU ${GPU_ID} ===="
