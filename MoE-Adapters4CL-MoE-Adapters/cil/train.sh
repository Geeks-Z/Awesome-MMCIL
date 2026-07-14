#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

python main.py \
    --config-path configs/class \
    --config-name cifar100_5_5.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_10_10.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_20_20.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_50_5.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_50_10.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100.yaml"
python main.py \
    --config-path configs/class \
    --config-name imagenet_r_10_10.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/imagenet-r" \
    class_order="class_orders/imagenet_R.yaml"
python main.py \
    --config-path configs/class \
    --config-name imagenet_r_20_20.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/imagenet-r" \
    class_order="class_orders/imagenet_R.yaml"
python main.py \
    --config-path configs/class \
    --config-name imagenet_r_40_40.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/imagenet-r" \
    class_order="class_orders/imagenet_R.yaml"
python main.py \
    --config-path configs/class \
    --config-name imagenet_r_100_10.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/imagenet-r" \
    class_order="class_orders/imagenet_R.yaml"
python main.py \
    --config-path configs/class \
    --config-name imagenet_r_100_20.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/imagenet-r" \
    class_order="class_orders/imagenet_R.yaml"
python main.py \
    --config-path configs/class \
    --config-name cub200_10_10.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cub200.yaml"
python main.py \
    --config-path configs/class \
    --config-name cub200_20_20.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cub200.yaml"
python main.py \
    --config-path configs/class \
    --config-name cub200_40_40.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cub200.yaml"
python main.py \
    --config-path configs/class \
    --config-name cub200_100_10.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cub200.yaml"
python main.py \
    --config-path configs/class \
    --config-name cub200_100_20.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cub200.yaml"
