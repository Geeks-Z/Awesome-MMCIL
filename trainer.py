import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    backbone_name = args["backbone_type"]
    if backbone_name.startswith("pretrained_"):
        backbone_name = backbone_name[len("pretrained_") :]

    logs_name = "logs/{}".format(args["model_name"])

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}_{}_{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        backbone_name,
        init_cls,
        args["increment"],
        args["seed"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)
    model.save_dir = logs_name

    eval_metrics = ["top{}".format(k) for k in model.eval_topk]
    cnn_curve = {metric: [] for metric in eval_metrics}
    nme_curve = {metric: [] for metric in eval_metrics}
    zs_seen_curve = {metric: [] for metric in eval_metrics}
    zs_unseen_curve = {metric: [] for metric in eval_metrics}
    zs_harmonic_curve = {metric: [] for metric in eval_metrics}
    zs_total_curve = {metric: [] for metric in eval_metrics}

    for task in range(data_manager.nb_tasks):
        #  logging.info("All params: {}".format(count_parameters(model._network)))
        #  logging.info(
        #      "Trainable params: {}".format(count_parameters(model._network, True))
        #  )
        model.incremental_train(data_manager)
        # cnn_accy, nme_accy = model.eval_task()
        cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total = (
            model.eval_task()
        )
        model.after_task()

        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        for metric in eval_metrics:
            if metric in cnn_accy:
                cnn_curve[metric].append(cnn_accy[metric])
                logging.info("CNN {} curve: {}".format(metric, cnn_curve[metric]))

        logging.info(
            "Average Accuracy (CNN top1): {}".format(
                sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            )
        )

    if args["backbone_type"] == "openai_clip":
        clip_type = "OpenAI CLIP"
    elif args["backbone_type"] == "clip":
        clip_type = "OpenCLIP_LAION400M"
    else:
        clip_type = "OpenCLIP_LAION2B"

    print(f"\n{'=' * 40}")
    print(
        "Finished {}_init{}_inc{} seed={}".format(
            args["dataset"], args["init_cls"], args["increment"], args["seed"]
        )
    )
    print("Backbone: {}".format(clip_type))
    print(
        "Average Accuracy (Top1): {}".format(
            round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2)
        )
    )
    print("Last Accuracy: {}".format(round(cnn_curve["top1"][-1], 2)))
    for metric in eval_metrics:
        if metric == "top1" or len(cnn_curve[metric]) == 0:
            continue
        print(
            "Average Accuracy ({}): {}".format(
                metric.capitalize(),
                round(sum(cnn_curve[metric]) / len(cnn_curve[metric]), 2),
            )
        )
        print(
            "Last {}: {}".format(metric.capitalize(), round(cnn_curve[metric][-1], 2))
        )
    print(f"{'=' * 40}\n")


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
