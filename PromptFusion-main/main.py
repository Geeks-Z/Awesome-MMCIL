"""
Author: Haoran Chen
Date: 2024.07.07
"""

import argparse
import os
import json
import sys
import numpy as np
import random
import logging
import time
from datetime import datetime

import torch
import torch.nn as nn
import clip
from clip.model import build_model
from dataset import gen_dataset
from continuum import ClassIncremental, rehearsal, ContinualScenario
from timm.models import create_model
from model import Clip_PF, Clip_PFLite
from train_pf import train_pf
from train_pflite import train_pflite
import utils

def set_seed(seed):
    """Match the benchmark's four reproducible seed runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def set_log():
    dataset_name = {
        "Cifar": "cifar224",
        "ImagenetR": "imagenetr",
        "CUB200": "cub",
    }.get(args["dataset"], args["dataset"].lower())
    num_classes = {
        "Cifar": 100,
        "ImagenetR": 200,
        "CUB200": 200,
    }[args["dataset"]]
    increment = int(args.get("increment", num_classes // args["step"]))
    initial_increment = int(args.get("initial_increment", increment))
    base_classes = 0 if initial_increment == increment else initial_increment

    output_parts = [
        args["file_root"], args["model_type"], args["dataset"], args["backbone"],
    ]
    if args.get("experiment_tag"):
        output_parts.append(args["experiment_tag"])
    # Keep the original B0 output location, while isolating B50/B100 runs.
    if base_classes:
        output_parts.append("base_{}_inc_{}".format(base_classes, increment))
    output_parts.append("seed_{}".format(args["seed"]))
    args["output_folder"] = os.path.join(*output_parts)
    os.makedirs(args["output_folder"], exist_ok=True)

    benchmark_root = os.path.dirname(os.path.abspath(args["file_root"]))
    log_root_name = args.get("log_root_name", "OpenAI_CLIP_ViTB16")
    log_backbone_token = args.get("log_backbone_token", "vit_b16")
    log_dir = os.path.join(
        benchmark_root, "logs", log_root_name, "PromptFusion"
    )
    os.makedirs(log_dir, exist_ok=True)
    logfilename = os.path.join(
        log_dir,
        "{}_{}_{}_{}_{}.log".format(
            dataset_name, log_backbone_token, base_classes, increment, args["seed"]
        ),
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename, mode="w"),
            logging.StreamHandler(sys.stdout),
        ], force=True,
    )


def get_clip_model(args):
    clip_model_path = args.get("clip_model_path")
    if clip_model_path:
        if not os.path.isfile(clip_model_path):
            raise FileNotFoundError("CLIP checkpoint does not exist: {}".format(clip_model_path))
        state_dict = torch.load(clip_model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        clip_model = build_model(state_dict).to(args["device"])
    else:
        clip_model, _ = clip.load("ViT-B/16", device=args["device"])

    utils.convert_models_to_fp32(clip_model)

    for name, param in clip_model.named_parameters():
        param.requires_grad_(False)

    return clip_model

def set_models(clip_model, args):
    vpt_model = create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=args["num_classes"],
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
        prompt_length=args["vpt_prompt_length"],   
        prompt_init='uniform'
    )

    vpt_model = vpt_model.to(args["device"])

    freeze = ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']
    for n, p in vpt_model.named_parameters():
        if n.startswith(tuple(freeze)):
            p.requires_grad = False
            
    vpt_model = nn.DataParallel(vpt_model)

    if args["model_type"] == 'PF':
        custom_clip_model = Clip_PF(clip_model, args).to(args["device"])
    elif args["model_type"] == 'PF_Lite':
        custom_clip_model = Clip_PFLite(clip_model, args).to(args["device"])
    else:
        raise Exception("Model type doesn't exist!")

    custom_clip_model = nn.DataParallel(custom_clip_model)
    custom_clip_model = custom_clip_model.module

    for name, param in custom_clip_model.named_parameters():
        if (not 'prompt' in name) and (not 'alpha' in name) and (not 'beta' in name) and (not 'lambda_' in name) and (not 'gumbel' in name):
            param.requires_grad_(False)

    return vpt_model, custom_clip_model

def main(args):
    train_dataset, test_dataset, classnames, transform_train, transform_test = gen_dataset(args)

    increment = int(args.get("increment", args["num_classes"] / args["step"]))
    initial_increment = int(args.get("initial_increment", increment))
    if initial_increment <= 0 or increment <= 0:
        raise ValueError("initial_increment and increment must be positive")
    if initial_increment > args["num_classes"]:
        raise ValueError("initial_increment cannot exceed the number of classes")
    if (args["num_classes"] - initial_increment) % increment:
        raise ValueError("remaining classes must be divisible by increment")

    args["increment"] = increment
    args["initial_increment"] = initial_increment
    args["base_classes"] = 0 if initial_increment == increment else initial_increment
    args["step"] = 1 + (args["num_classes"] - initial_increment) // increment
    args["task_class_counts"] = [initial_increment] + [increment] * (args["step"] - 1)

    class_mask = list()
    labels = [i for i in range(len(classnames))]

    for task_class_count in args["task_class_counts"]:
        scope = labels[:task_class_count]
        labels = labels[task_class_count:]
        class_mask.append(scope)

    scenario_train = ClassIncremental(
        train_dataset,
        initial_increment=initial_increment,
        increment=increment,
        transformations=transform_train,
    )
    scenario_test = ClassIncremental(
        test_dataset,
        initial_increment=initial_increment,
        increment=increment,
        transformations=transform_test,
    )

    memory = rehearsal.RehearsalMemory(memory_size=args["memory_size"], herding_method=args["herding_method"])
    
    clip_model = get_clip_model(args)
    vpt_model, custom_clip_model = set_models(clip_model, args)

    t = time.time()

    if args["model_type"] == "PF":
        acc_table = train_pf(clip_model, custom_clip_model, vpt_model, scenario_train, scenario_test, classnames, memory, class_mask, args)
    elif args["model_type"] == "PF_Lite":
        acc_table = train_pflite(clip_model, custom_clip_model, vpt_model, scenario_train, scenario_test, classnames, memory, class_mask, args)
    else:
        raise Exception("Model type doesn't exist!")

    stage_acc = [
        round(100 * np.mean(acc_table[: task_id + 1, task_id]), 2)
        for task_id in range(args["step"])
    ]
    average_acc = round(float(np.mean(stage_acc)), 2)
    last_acc = round(stage_acc[-1], 2)
    forgetting = round(
        100 * float(np.mean(np.max(acc_table, axis=1) - acc_table[:, -1])), 2
    )

    logging.info("CNN top1 curve: %s", stage_acc)
    logging.info("Average Accuracy (CNN top1): %s", average_acc)
    logging.info("Last Accuracy: %s", last_acc)
    logging.info(
        "Finished %s_base%s_inc%s seed=%s",
        args["dataset"], args["base_classes"], args["increment"], args["seed"],
    )
    backbone_label = args.get("backbone_label", "OpenAI CLIP ViT-B/16")
    logging.info("Backbone: %s", backbone_label)

    print(f"\n{'=' * 40}")
    print(
        "Finished {}_base{}_inc{} seed={}".format(
            args["dataset"],
            args["base_classes"], args["increment"], args["seed"]
        )
    )
    print("Backbone: {}".format(backbone_label))
    print("Average Accuracy (Top1): {:.2f}".format(average_acc))
    print("Last Accuracy: {:.2f}".format(last_acc))
    print("Forgetting: {:.2f}".format(forgetting))
    print(f'Cost:{time.time() - t:.4f}s')
    # print("Backbone: {}".format(clip_type))
    # print("Base Accuracy: {}".format(round(cnn_curve["top1"][0], 2)))
    # print("Last Accuracy: {}".format(round(cnn_curve["top1"][-1], 2)))

    print(f"{'=' * 40}\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and Evaluation Script')
    parser.add_argument('--config', type=str, default='./config/pf_inr_5_5.json', help='Json file of settings.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for one benchmark run.')
    parser.add_argument('--data-root', type=str, default=None, help='Override the dataset root from the JSON config.')
    parser.add_argument('--file-root', type=str, default=None, help='Override the PromptFusion source root from the JSON config.')
    parser.add_argument('--clip-model-path', type=str, default=None, help='Optional local CLIP-compatible checkpoint.')
    parser.add_argument('--log-root-name', type=str, default=None, help='Top-level logs directory name.')
    parser.add_argument('--log-backbone-token', type=str, default=None, help='Backbone token used in the log filename.')
    parser.add_argument('--experiment-tag', type=str, default=None, help='Optional output-directory discriminator.')
    parser.add_argument('--backbone-label', type=str, default=None, help='Human-readable backbone label for logs.')
    cli_args = parser.parse_args()

    param = load_json(cli_args.config)

    args = param
    args["config"] = cli_args.config
    args["seed"] = cli_args.seed
    if cli_args.data_root is not None:
        args["data_root"] = cli_args.data_root
    if cli_args.file_root is not None:
        args["file_root"] = cli_args.file_root
    for key in ("clip_model_path", "log_root_name", "log_backbone_token", "experiment_tag", "backbone_label"):
        value = getattr(cli_args, key)
        if value is not None:
            args[key] = value

    set_seed(args["seed"])

    set_log()

    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

    main(args)
