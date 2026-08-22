#!/usr/bin/env bash
# Submit with: nohup bash scripts/run_openclip_moe_adapters.sh > logs/OpenCLIP_LAION400M_ViTB16/MoE-Adapters/queue/MoE-Adapters-LAION-400M.out 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-5}"
DATA_ROOT="${DATA_ROOT:-/home/team/zhaohongwei/Dataset}"
PYTHON_BIN="${PYTHON_BIN:-/home/team/zhaohongwei/anaconda3/envs/mmcl/bin/python}"
LAION_CHECKPOINT="${LAION_CHECKPOINT:-/home/team/zhaohongwei/pretrained_models/open_clip_pytorch_model_laion400m_e32.bin}"
LOG_ROOT="OpenCLIP_LAION400M_ViTB16"
BACKBONE_LABEL="OpenCLIP ViT-B/16 LAION-400M laion400m_e32"
SEEDS=(1993 0 42 2026)

case "$GPU_ID" in 1|2|4|5|6|7) ;; *) echo "Refusing reserved or unknown GPU $GPU_ID" >&2; exit 2 ;; esac
[[ -x "$PYTHON_BIN" ]] || { echo "mmcl Python not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$LAION_CHECKPOINT" ]] || { echo "Checkpoint not found: $LAION_CHECKPOINT" >&2; exit 2; }
[[ -d "$DATA_ROOT/cifar-100-python" && -d "$DATA_ROOT/imagenet-r" && -d "$DATA_ROOT/cub" ]] || {
    echo "Expected CIFAR-100, ImageNet-R, and CUB datasets under $DATA_ROOT" >&2; exit 2;
}

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export DATA_ROOT PYTHON_BIN LAION_CHECKPOINT
mkdir -p "$ROOT/logs/$LOG_ROOT/MoE-Adapters/queue"
cd "$ROOT/MoE-Adapters4CL-MoE-Adapters/cil"

run_one() {
    local config="$1" dataset_root="$2" class_order="$3" seed="$4"
    local metrics="experiments/openclip_laion400m_e32/${config}/seed_${seed}/metrics.json"
    if [[ -s "$metrics" ]] && tail -n 1 "$metrics" | grep -q '"avg"'; then
        echo "[skip] $config seed=$seed has completed metrics"
        return 0
    fi
    echo "[run] $config seed=$seed on physical GPU $GPU_ID"
    "$PYTHON_BIN" main.py --config-path configs/class --config-name="$config" \
        dataset_root="$dataset_root" class_order="$class_order" \
        model_name="$LAION_CHECKPOINT" +seed="$seed" \
        +log_root_name="$LOG_ROOT" +log_backbone_token="clip" \
        +backbone_label="$BACKBONE_LABEL" \
        hydra.run.dir="./experiments/openclip_laion400m_e32/${config}/seed_${seed}"
}

for seed in "${SEEDS[@]}"; do
    run_one cifar100_0_10 "$DATA_ROOT" class_orders/cifar100.yaml "$seed"
    run_one cifar100_50_10 "$DATA_ROOT" class_orders/cifar100.yaml "$seed"
    run_one imagenet_r_0_20 "$DATA_ROOT/imagenet-r" class_orders/imagenet_R.yaml "$seed"
    run_one imagenet_r_100_20 "$DATA_ROOT/imagenet-r" class_orders/imagenet_R.yaml "$seed"
    run_one cub200_20_20 "$DATA_ROOT" class_orders/cub200.yaml "$seed"
    run_one cub200_100_20 "$DATA_ROOT" class_orders/cub200.yaml "$seed"
done
