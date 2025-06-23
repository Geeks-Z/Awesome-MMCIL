#!bin/bash

# for imagenet-100 dataset; 10 classes/task
CUDA_VISIBLE_DEVICES=5 python main.py \
    --config-path configs/class \
    --config-name cifar100_10-10-MoE-Adapters.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"

