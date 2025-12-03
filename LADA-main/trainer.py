import os
import time
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torchvision import transforms

from clip import clip

from scenario_datasets.build_functions import build_cur_task_data_loader, build_TAIL_testloader

from models import *

from utils.meter import AverageMeter
from utils.losses import *
from utils.evaluator import Evaluator
from utils.sampler import sample_from_prototypes


def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(cfg)

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        print("mean:", mean)
        print("std:", std)

        if cfg.dataset in ["mnist", "eurosat"]:
            self.transform_train = transforms.Compose([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        self.transform_test = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        build_TAIL_testloader_return = build_TAIL_testloader(root=root, 
                                                             dataset_sequence=cfg.dataset_sequence, 
                                                             transform_test=self.transform_test,
                                                             batch_size=cfg.batch_size,
                                                             num_workers=cfg.num_workers)
        self.TAIL_test_loader, self.merged_classnames, self.indices = build_TAIL_testloader_return
        
        if not cfg.zero_shot:
            build_cur_task_data_loader_return = build_cur_task_data_loader(root=root,
                                                                        dataset_name=cfg.dataset, 
                                                                        transform_train=self.transform_train, 
                                                                        transform_test=self.transform_test, 
                                                                        num_shots=cfg.num_shots, 
                                                                        batch_size=cfg.batch_size,
                                                                        num_workers=cfg.num_workers)
            self.train_loader, self.train_loader4updating, self.test_loader, self.classnames = build_cur_task_data_loader_return

            self.num_classes = len(self.classnames)

    def build_model(self):
        cfg = self.cfg

        print("Building model")
        print(f"Loading CLIP (backbone: {cfg.backbone})")
        self.clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
        if cfg.zero_shot:
            self.model = ZeroShotCLIP(self.clip_model)
            self.model.to(self.device)

            self.model.init_text_features(self.merged_classnames, ['a photo of a {}.'], clip.tokenize, self.device)
            
        else:
            self.model = PeftModelFromCLIP(cfg, self.clip_model, self.num_classes, alpha=cfg.alpha, beta=cfg.beta)
            self.model.to(self.device)
            if cfg.continue_train and not cfg.continue_train_first:
                self.load_model4CL(cfg.model_dir)
            self.total_num_classes = self.num_classes + self.model.dpt.text_prototypes.shape[0]

            self.model.init_prompts(self.classnames, ['a photo of a {}.'], clip.tokenize, self.device)

        if not cfg.zero_shot:
            self.model.lada.build_lada(self.train_loader4updating, device=self.device, k=cfg.lada_k)
            self.build_optimizer()
            self.build_criterion()
            
            torch.cuda.empty_cache()
        
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        text_tuned_params = sum(p.numel() for p in self.model.text_tuner.parameters())
        lada_tuned_params = self.model.lada.curr_lada_features.numel()
        print(f"Total params: {total_params}")
        print(f"Tuned text encoder params: {text_tuned_params}")
        print(f"Tuned LADA params: {lada_tuned_params}")

        self.optim = torch.optim.AdamW([{"params": self.model.text_tuner.parameters()},
                                        {"params": self.model.lada.curr_lada_features}],
                                       lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optim,
                                                         max_lr=cfg.lr,
                                                         epochs=cfg.num_epochs,
                                                         steps_per_epoch=len(self.train_loader))
        self.scaler = GradScaler('cuda') if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg

        if cfg.loss_type == "CE":
            self.criterion = CELoss()
        elif cfg.loss_type == "Focal": # https://arxiv.org/abs/1708.02002
            self.criterion = FocalLoss()

    def train(self):
        cfg = self.cfg

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)

        time_start = time.time()    # Remember the starting time (for computing the elapsed time)

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            self.model.train()
            end = time.time()

            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader):
                data_time.update(time.time() - end)

                image = batch[0]
                label = batch[1]
                image = image.to(self.device)
                label = label.to(self.device)
                image_prototypes = self.model.dpt.image_prototypes
                text_prototypes = self.model.dpt.text_prototypes

                # Generate the image prototypes based on Distribution-Preserved Training method
                if cfg.continue_train and not cfg.continue_train_first:
                    image_prototypes, image_prototypes_weights = sample_from_prototypes(image_prototypes, 
                                                                                               self.model.dpt.image_prototypes_covs, 
                                                                                               self.model.dpt.image_prototypes_weights)
                    num_image_prototypes = image_prototypes.shape[0]
                    num_text_prototypes = text_prototypes.shape[0]

                    if num_image_prototypes % num_text_prototypes != 0:
                        raise ValueError("num_image_prototypes must be divisible by num_text_prototypes.")
                    repeat_factor = num_image_prototypes // num_text_prototypes
                    increment_vector = torch.arange(num_text_prototypes, device=self.device)
                    increment_vector = increment_vector.repeat_interleave(repeat_factor)
                    label = label + num_text_prototypes
                    label = torch.cat([increment_vector, label], dim=0)

                    weights = torch.cat((image_prototypes_weights * cfg.image_prototypes_weight_coef, torch.ones(image.shape[0], device=self.device)), dim=0)
                else:
                    weights = None

                # Training
                if cfg.prec == "amp":
                    with autocast('cuda'):
                        output = self.model(image,
                                            self.model.text_tuner, 
                                            image_prototypes=image_prototypes, 
                                            text_prototypes=text_prototypes,
                                            classifier=self.model.lada.joint_classifier)
                        loss = self.criterion(output, label, weights)
                        self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad()
                else:
                    raise NotImplementedError()

                # Output
                with torch.no_grad():
                    pred = output.argmax(dim=1)
                    correct = pred.eq(label).float()
                    acc = correct.mean().mul_(100.0)

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                acc_meter.update(acc.item())
                batch_time.update(time.time() - end)

                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))
                
                end = time.time()

                self.sched.step()
        torch.cuda.empty_cache()

        print("Finish training")
        # Show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")
        
        # Update the prototypes and LADA features
        self.model.eval()
        self.model.dpt.update_text_prototypes(self.model.prompts, self.model.text_tuner)
        print(f"Updated text prototypes shape: {self.model.dpt.text_prototypes.shape}")
        self.model.dpt.update_image_prototypes(self.train_loader4updating, cfg.prototype_k, self.device)
        print(f"Updated image prototypes shape: {self.model.dpt.image_prototypes.shape}")
        self.model.lada.update_lada_features()
        print(f"Updated LADA features shape: {self.model.lada.prev_lada_features.shape}")

        # Save model
        self.save_model(cfg.output_dir)

        self.test_cur_task()
        if cfg.continue_train:
            self.test_wo_selector()
            # self.test_w_selector()

    @torch.no_grad()
    def test_zero_shot(self):
        self.model.eval()
        self.evaluator.reset(self.indices)

        print("==================================================================================")
        print(f"Evaluate on the test set")
        data_loader = self.TAIL_test_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            output = self.model(image)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results

    @torch.no_grad()
    def test_cur_task(self):
        self.model.eval()
        indices = [0]
        self.evaluator.reset(indices)

        print("==================================================================================")
        print(f"Evaluate on the current task test set")
        data_loader = self.test_loader
        text_features = self.model.text_encoder(self.model.prompts, self.model.text_tuner)

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            output = self.model(image, 
                                text_features=text_features,
                                lada_features=self.model.lada.curr_lada_features,
                                classifier=self.model.lada.curr_classifier)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results

    @torch.no_grad()
    def test_w_selector(self):
        '''
        Use the original CLIP as a selector to determine whether the input image belongs to a seen class or an unseen class;
        For images predicted as seen classes by the selector, use the trained model for prediction;
        For images predicted as unseen classes by the selector, use the original CLIP for prediction.
        '''
        cfg = self.cfg
        self.model.eval()
        indices = self.indices
        self.evaluator.reset(indices)

        data_loader = self.TAIL_test_loader

        # Load zero shot CLIP
        clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
        frozen_clip = ZeroShotCLIP(clip_model)
        frozen_clip.to(self.device)
        frozen_clip.init_text_features(self.merged_classnames, ['a photo of a {}.'], clip.tokenize, self.device)
        frozen_clip.eval()

        # For seen class names(ID), use the text features from the trained model
        id_text_features = self.model.dpt.text_prototypes
        id_num = id_text_features.shape[0]

        print("==================================================================================")
        print(f"Evaluate on the X-TAIL test set")

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            zs_output = frozen_clip(image)
            zs_pred = zs_output.max(1)[1]
            
            id_indices = (zs_pred < id_num).nonzero(as_tuple=True)[0]
            ood_indices = (zs_pred >= id_num).nonzero(as_tuple=True)[0]

            if id_indices.numel() > 0:
                id_image = image[id_indices]
                id_output = self.model(id_image, 
                                       text_features=id_text_features,
                                       lada_features=self.model.lada.prev_lada_features,
                                       classifier=self.model.lada.joint_classifier)
                # Pad zeros at the end of id_output to match zs_output size
                pad_size = zs_output.shape[1] - id_output.shape[1]
                if pad_size > 0:
                    padding = torch.zeros(id_output.shape[0], pad_size).to(id_output.device)
                    id_output_padded = torch.cat([id_output, padding], dim=1)
                else:
                    id_output_padded = id_output
            else:
                id_output_padded = None

            if ood_indices.numel() > 0:
                ood_output = zs_output[ood_indices]
            else:
                ood_output = None

            # Combine id_output_padded and ood_output to form output, ensuring that the order matches the images in the batch
            output = torch.zeros_like(zs_output)
            if id_output_padded is not None:
                output[id_indices] = id_output_padded
            if ood_output is not None:
                output[ood_indices] = ood_output

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results
    
    @torch.no_grad()
    def test_wo_selector(self):
        '''
        Use text prototypes extracted by the trained model as text features for seen classes;
        Use text features extracted by the original CLIP as text features for unseen classes.
        Concatenate these as the complete set of text features, and input them together with LADA features into the trained model for prediction.
        '''
        cfg = self.cfg
        self.model.eval()
        indices = self.indices
        self.evaluator.reset(indices)

        data_loader = self.TAIL_test_loader

        # Load zero shot CLIP
        clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
        frozen_clip = ZeroShotCLIP(clip_model)
        frozen_clip.to(self.device)
        frozen_clip.init_text_features(self.merged_classnames, ['a photo of a {}.'], clip.tokenize, self.device)
        frozen_clip.eval()

        # For the seen class names(ID), use the text features from the trained model; 
        # for the unseen class names(OOD), use the text features from the original CLIP
        origin_text_features = frozen_clip.text_features
        id_text_features = self.model.dpt.text_prototypes
        id_num = id_text_features.shape[0]
        ood_text_features = origin_text_features[id_num:,:]
        text_features = torch.cat((id_text_features, ood_text_features), dim=0)

        print("==================================================================================")
        print(f"Evaluate on the X-TAIL test set")

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(self.device)
            label = label.to(self.device)

            output = self.model(image, 
                                text_features=text_features,
                                lada_features=self.model.lada.prev_lada_features,
                                classifier=self.model.lada.joint_classifier)

            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results

    @torch.no_grad()
    def save_model(self, directory):
        checkpoint = {
            "image_prototypes": self.model.dpt.image_prototypes,
            "image_prototypes_covs": self.model.dpt.image_prototypes_covs,
            "image_prototypes_weights": self.model.dpt.image_prototypes_weights,
            "text_prototypes": self.model.dpt.text_prototypes,
            "lada_features": self.model.lada.prev_lada_features,
            "classifier": self.model.lada.joint_classifier,
        }

        save_path = os.path.join(directory, "checkpoint.pth.tar")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    @torch.no_grad()
    def load_model4CL(self, directory):
        load_path = os.path.join(directory, "checkpoint.pth.tar")
        print("Loading model weights from {}".format(load_path))
        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)

        self.model.dpt.image_prototypes = checkpoint["image_prototypes"].to(self.device)
        self.model.dpt.image_prototypes_covs = checkpoint["image_prototypes_covs"].to(self.device)
        self.model.dpt.image_prototypes_weights = checkpoint["image_prototypes_weights"].to(self.device)
        self.model.dpt.text_prototypes = checkpoint["text_prototypes"].to(self.device)
        self.model.lada.prev_lada_features = checkpoint["lada_features"]
        self.model.lada.joint_classifier = checkpoint["classifier"]
