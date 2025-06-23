#!/bin/bash
#CUDA_VISIBLE_DEVICES=5 python main.py \
#    --config-path configs/class \
#    --config-name cifar100_5-5-MoE-Adapters.yaml \
#    dataset_root="/home/team/zhaohongwei/Dataset" \
#    class_order="class_orders/cifar100.yaml"
#CUDA_VISIBLE_DEVICES=4 python main.py \
#    --config-path configs/class \
#    --config-name cifar100_10-10-MoE-Adapters.yaml \
#    dataset_root="/home/team/zhaohongwei/Dataset" \
#    class_order="class_orders/cifar100.yaml"
CUDA_VISIBLE_DEVICES=6 python main.py \
    --config-path configs/class \
    --config-name cifar100_20-20-MoE-Adapters.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"