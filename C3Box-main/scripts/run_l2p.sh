#!/bin/bash
# Aircraft
python main.py --config=./configs/engine/engine_aircraft_B0_Inc10.json
python main.py --config=./configs/engine/engine_aircraft_B50_Inc10.json

# CIFAR100
python main.py --config=./configs/engine/engine_cifar_B10_Inc10.json
python main.py --config=./configs/engine/engine_cifar_B50_Inc10.json

# Cars
python main.py --config=./configs/engine/engine_cars_B0_Inc10.json
python main.py --config=./configs/engine/engine_cars_B50_Inc10.json

# ImageNet-R
python main.py --config=./configs/engine/engine_inr_B20_Inc20.json
python main.py --config=./configs/engine/engine_inr_B100_Inc20.json

# CUB
python main.py --config=./configs/engine/engine_cub_B20_Inc20.json
python main.py --config=./configs/engine/engine_cub_B100_Inc20.json

# UCF
python main.py --config=./configs/engine/engine_ucf_B0_Inc10.json
python main.py --config=./configs/engine/engine_ucf_B50_Inc10.json

# SUN
python main.py --config=./configs/engine/engine_sun_B0_Inc30.json
python main.py --config=./configs/engine/engine_sun_B150_Inc30.json

# Food
python main.py --config=./configs/engine/engine_food_B0_Inc10.json
python main.py --config=./configs/engine/engine_food_B50_Inc10.json

# ObjectNet
python main.py --config=./configs/engine/engine_objectnet_B0_Inc20.json
python main.py --config=./configs/engine/engine_objectnet_B100_Inc20.json
