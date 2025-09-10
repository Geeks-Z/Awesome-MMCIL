#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

python main.py \
    --config-path configs/class \
    --config-name cifar100_5-5-MoE-Adapters.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_10-10-MoE-Adapters.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_20-20-MoE-Adapters.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_50-5-MoE-Adapters.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_50-10-MoE-Adapters.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"

#python main.py \
#    --config-path configs/class \
#    --config-name tinyimagenet_100-10.yaml \
#    dataset_root="/home/team/zhaohongwei/Dataset" \
#    class_order="class_orders/tinyimagenet.yaml"
#python main.py \
#    --config-path configs/class \
#    --config-name tinyimagenet_100-20.yaml \
#    dataset_root="/home/team/zhaohongwei/Dataset" \
#    class_order="class_orders/tinyimagenet.yaml"
