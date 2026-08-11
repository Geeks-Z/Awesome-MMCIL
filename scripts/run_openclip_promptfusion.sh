#!/usr/bin/env bash
# Submit with: nohup bash scripts/run_openclip_promptfusion.sh > logs/OpenCLIP_LAION400M_ViTB16/PromptFusion/queue/PromptFusion-LAION-400M.out 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-2}"
DATA_ROOT="${DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
PYTHON_BIN="${PYTHON_BIN:-/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python}"
LAION_CHECKPOINT="${LAION_CHECKPOINT:-/home/team/zhaohongwei/pretrained_models/open_clip_pytorch_model_laion400m_e32.bin}"
LOG_ROOT="OpenCLIP_LAION400M_ViTB16"
BACKBONE_LABEL="OpenCLIP ViT-B/16 LAION-400M laion400m_e32"
EXPERIMENT_TAG="openclip_laion400m_e32"
SEEDS=(1993 0 42 2026)

if [[ "$GPU_ID" != "2" ]]; then
    echo "This launcher is pinned to physical GPU 2; received GPU_ID=$GPU_ID" >&2
    exit 2
fi
[[ -x "$PYTHON_BIN" ]] || { echo "mmcl Python not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$LAION_CHECKPOINT" ]] || { echo "Checkpoint not found: $LAION_CHECKPOINT" >&2; exit 2; }
[[ -d "$DATA_ROOT/cifar-100-python" && -d "$DATA_ROOT/imagenet-r" && -d "$DATA_ROOT/cub" ]] || {
    echo "Expected CIFAR-100, ImageNet-R, and CUB datasets under $DATA_ROOT" >&2; exit 2;
}

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export DATA_ROOT PYTHON_BIN LAION_CHECKPOINT
mkdir -p "$ROOT/logs/$LOG_ROOT/PromptFusion/queue"
cd "$ROOT/PromptFusion-main"

run_one() {
    local config="$1" data_root="$2" dataset="$3" base="$4" increment="$5" seed="$6"
    local log_file="$ROOT/logs/$LOG_ROOT/PromptFusion/${dataset}_clip_${base}_${increment}_${seed}.log"
    if [[ -s "$log_file" ]] && grep -q "Finished " "$log_file"; then
        echo "[skip] $config seed=$seed has a completed log"
        return 0
    fi
    echo "[run] $config seed=$seed on physical GPU $GPU_ID"
    "$PYTHON_BIN" main.py --config="config/$config" --seed="$seed" \
        --data-root="$data_root" --file-root="$PWD" \
        --clip-model-path="$LAION_CHECKPOINT" --log-root-name="$LOG_ROOT" \
        --log-backbone-token="clip" --experiment-tag="$EXPERIMENT_TAG" \
        --backbone-label="$BACKBONE_LABEL"
}

for seed in "${SEEDS[@]}"; do
    run_one pf_cifar_10_10.json "$DATA_ROOT" cifar224 0 10 "$seed"
    run_one pf_cifar_50_10.json "$DATA_ROOT" cifar224 50 10 "$seed"
    run_one pf_inr_20_20.json "$DATA_ROOT/imagenet-r" imagenetr 0 20 "$seed"
    run_one pf_inr_100_20.json "$DATA_ROOT/imagenet-r" imagenetr 100 20 "$seed"
    run_one pf_cub_20_20.json "$DATA_ROOT" cub 0 20 "$seed"
    run_one pf_cub_100_20.json "$DATA_ROOT" cub 100 20 "$seed"
done
