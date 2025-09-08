#!/bin/bash
python main.py \
    --config-path configs/class \
    --config-name cifar100_5-5.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100_order.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_10-10.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100_order.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_20-20.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100_order.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_50-5.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100_order.yaml"
python main.py \
    --config-path configs/class \
    --config-name cifar100_50-10.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset" \
    class_order="class_orders/cifar100_order.yaml"
python main.py \
    --config-path configs/class \
    --config-name imagenet_r_10-10.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/imagenet-r" \
    class_order="class_orders/imagenet_R_order.yaml"
python main.py \
    --config-path configs/class \
    --config-name imagenet_r_20-20.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/imagenet-r" \
    class_order="class_orders/imagenet_R_order.yaml"
python main.py \
    --config-path configs/class \
    --config-name imagenet_r_40-40.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/imagenet-r" \
    class_order="class_orders/imagenet_R_order.yaml"
python main.py \
    --config-path configs/class \
    --config-name imagenet_r_100-10.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/imagenet-r" \
    class_order="class_orders/imagenet_R_order.yaml"
python main.py \
    --config-path configs/class \
    --config-name imagenet_r_100-20.yaml \
    dataset_root="/home/team/zhaohongwei/Dataset/imagenet-r" \
    class_order="class_orders/imagenet_R_order.yaml"
#python main.py \
#    --config-path configs/class \
#    --config-name imagenet100_10-10.yaml \
#    dataset_root="/home/team/zhaohongwei/Dataset/imagenet100" \
#    class_order="class_orders/imagenet100.yaml"

