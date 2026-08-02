#!/usr/bin/env python3
"""Run one ENGINE configuration with OpenAI CLIP and an explicit seed/device."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer import train


DEFAULT_OPENAI_CHECKPOINT = (
    "/public/home/hanlida/Dr.1/pretrained_models/open_clip_pytorch_model.bin"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1993)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open() as config_file:
        params = json.load(config_file)

    # Keep the original protocol hyperparameters while making the evaluated
    # backbone, seed, and physical GPU unambiguous in both the logs and output.
    params["backbone_type"] = "openai_clip"
    params["clip_pretrained"] = os.environ.get(
        "MMCL_CLIP_PRETRAINED", DEFAULT_OPENAI_CHECKPOINT
    )
    params["seed"] = [args.seed]
    params["device"] = [args.device]
    params["config"] = str(args.config)
    train(params)


if __name__ == "__main__":
    main()
