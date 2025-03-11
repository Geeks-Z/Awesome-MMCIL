#!/bin/bash
CUDA_VISIBLE_DEVICES=3
python main.py \
    --config-path configs/class \
    --config-name cifar100_10-10-MoE-Adapters.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/cifar-100-python" \
    class_order="class_orders/cifar100.yaml"
python main.py \
    --config-path configs/class \
    --config-name imagenet_r_10-10.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/imagenet-r" \
    class_order="class_orders/imagenet_R_order.yaml"



