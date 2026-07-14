import os
# The launcher selects the physical GPU through CUDA_VISIBLE_DEVICES.  Do not
# overwrite it here: after masking, cfg.gpu=0 correctly selects that GPU.
import torch
import random
import argparse
import numpy as np

from utils.config import _C as cfg
from utils.logger import setup_logger

from trainer import Trainer


def main(args):
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
    print("Output directory: {}".format(cfg.output_dir))
    if cfg.model_dir is None:
        cfg.model_dir = cfg.output_dir
    setup_logger(cfg.output_dir, cfg.dataset)
    
    print("** Config **")
    print(cfg)
    print("************")
    
    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    trainer = Trainer(cfg)
    
    if cfg.zero_shot:
        trainer.test_zero_shot()
        return

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, help="data config file")
    parser.add_argument("--model", "-m", type=str, help="model config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
