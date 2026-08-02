import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    apply_runtime_overrides(param)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def apply_runtime_overrides(param):
    """Allow launchers to select an idle GPU and a subset of configured seeds."""
    device = os.environ.get("MMCL_DEVICE")
    if device:
        param["device"] = [int(value) for value in device.split(",")]

    seeds = os.environ.get("MMCL_SEEDS")
    if seeds:
        param["seed"] = [int(value) for value in seeds.split(",")]


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./configs/engine/engine_cifar_B0_Inc10.json',
                        help='Json file of settings.')
    return parser


if __name__ == '__main__':
    main()
