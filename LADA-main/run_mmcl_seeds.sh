#!/usr/bin/env bash
# LADA's published TAIL protocol is a separate 10-domain benchmark.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${LADA_DATA_ROOT:-/public/home/hanlida/Dr.1/Dataset/X-TAIL}"
SEEDS=(0 42 1993 2026)
REQUIRED=(Aircraft Caltech101 DTD EuroSAT Flowers Food MNIST Pets StanfordCars Sun397)

for dataset in "${REQUIRED[@]}"; do
  [[ -d "${DATA_ROOT}/${dataset}" ]] || {
    echo "Missing LADA X-TAIL dataset: ${DATA_ROOT}/${dataset}" >&2
    exit 1
  }
done

run_sequence() {
  local data_cfg="$1" shots="$2" label="$3"
  shift 3
  local -a sequence=("$@")

  for seed in "${SEEDS[@]}"; do
    local output="${label}_seed${seed}"
    for i in "${!sequence[@]}"; do
      local task="${sequence[$i]}" epochs
      if [[ "$shots" == "-1" ]]; then
        case "$task" in
          aircraft) epochs=60 ;; caltech101) epochs=20 ;; dtd) epochs=30 ;;
          eurosat) epochs=20 ;; flowers) epochs=40 ;; food101) epochs=10 ;;
          mnist) epochs=20 ;; oxford_pets) epochs=10 ;; stanford_cars) epochs=20 ;;
          sun397) epochs=2 ;;
        esac
      else
        case "$task" in
          aircraft) epochs=40 ;; caltech101) epochs=10 ;; dtd) epochs=30 ;;
          eurosat) epochs=100 ;; flowers) epochs=30 ;; food101) epochs=5 ;;
          mnist) epochs=200 ;; oxford_pets) epochs=10 ;; stanford_cars) epochs=30 ;;
          sun397) epochs=10 ;;
        esac
      fi
      local -a opts=(seed "$seed" num_shots "$shots" dataset "$task" num_epochs "$epochs" root "$DATA_ROOT" output_dir "$output")
      [[ "$i" == "0" ]] && opts+=(continue_train_first True)
      echo "==== LADA data=${data_cfg} shots=${shots} task=${sequence[$i]} seed=${seed} GPU=${CUDA_VISIBLE_DEVICES:-unset} ===="
      python main.py -d "$data_cfg" -m clip_vit_b16 "${opts[@]}"
    done
    python result_process.py -d "$data_cfg" --output_dir "$output"
  done
}

cd "$ROOT"
run_sequence TAIL 16 TAIL_16shot aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397
run_sequence TAIL_order2 16 TAIL_16shot_order2 stanford_cars aircraft oxford_pets food101 sun397 mnist flowers dtd caltech101 eurosat
run_sequence TAIL -1 TAIL_fullshot aircraft caltech101 dtd eurosat flowers food101 mnist oxford_pets stanford_cars sun397
run_sequence TAIL_order2 -1 TAIL_fullshot_order2 stanford_cars aircraft oxford_pets food101 sun397 mnist flowers dtd caltech101 eurosat
