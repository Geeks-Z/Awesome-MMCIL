import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import AdapterVitNet
from utils.toolkit import get_attribute


class Learner(BaseLearner):
    """FeCAM with a LAION-400M OpenCLIP ViT-B/16 AdaptFormer backbone."""

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = AdapterVitNet(args, True)
        self.batch_size = get_attribute(args, "batch_size", 48)
        self.num_workers = get_attribute(args, "num_workers", 8)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 0.0)
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 20)
        self.optimizer_name = get_attribute(args, "optimizer", "sgd").lower()
        self.alpha = get_attribute(args, "fecam_alpha", 100.0)
        self.maha_batch_size = get_attribute(args, "maha_batch_size", 64)

        self.precision_mats = []
        self._precision_stack = None

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        proto_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        test_dataset = data_manager.get_dataset(
            np.arange(self._total_classes), source="test", mode="test"
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.train_loader_for_protonet = DataLoader(
            proto_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self._network.to(self._device)
        if self._cur_task == 0:
            self._tune_first_session()
        self._replace_fc(self.train_loader_for_protonet)

    def _make_optimizer(self):
        parameters = [p for p in self._network.parameters() if p.requires_grad]
        if self.optimizer_name == "sgd":
            return optim.SGD(
                parameters,
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.weight_decay,
            )
        if self.optimizer_name == "adam":
            return optim.AdamW(
                parameters, lr=self.init_lr, weight_decay=self.weight_decay
            )
        raise ValueError("Unsupported FeCAM optimizer: {}".format(self.optimizer_name))

    def _tune_first_session(self):
        trainable = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info("FeCAM first-session trainable parameters: %s", trainable)
        optimizer = self._make_optimizer()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.tuned_epoch), eta_min=self.min_lr
        )
        progress = tqdm(range(self.tuned_epoch), desc="FeCAM adapter tuning")
        for epoch in progress:
            self._network.train()
            losses = 0.0
            correct = 0
            total = 0
            for _, inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                correct += logits.argmax(dim=1).eq(targets).sum().item()
                total += targets.numel()
            scheduler.step()
            progress.set_description(
                "FeCAM task 0 epoch {}/{} loss {:.3f} acc {:.2f}".format(
                    epoch + 1,
                    self.tuned_epoch,
                    losses / max(1, len(self.train_loader)),
                    100.0 * correct / max(1, total),
                )
            )

    @torch.no_grad()
    def _replace_fc(self, loader):
        self._network.eval()
        features = []
        labels = []
        for _, inputs, targets in loader:
            features.append(self._network.extract_vector(inputs.to(self._device)).cpu())
            labels.append(targets.cpu())
        features = torch.cat(features).float()
        labels = torch.cat(labels)

        for class_index in labels.unique().tolist():
            class_features = features[labels == class_index].to(self._device)
            mean = class_features.mean(0)
            if class_features.shape[0] > 1:
                covariance = torch.cov(class_features.T)
            else:
                covariance = torch.zeros(
                    self._network.feature_dim,
                    self._network.feature_dim,
                    device=self._device,
                )
            shrunk = covariance + self.alpha * torch.eye(
                covariance.shape[0], device=self._device
            )
            correlation = torch.corrcoef(shrunk)
            correlation = torch.nan_to_num(correlation)
            correlation.diagonal().add_(1e-6)
            precision = torch.linalg.pinv(correlation).float()
            self._network.fc.weight.data[class_index].copy_(mean)
            self.precision_mats.append(precision.cpu())

        if len(self.precision_mats) != self._total_classes:
            raise RuntimeError(
                "FeCAM precision count {} does not match class count {}".format(
                    len(self.precision_mats), self._total_classes
                )
            )
        self._precision_stack = torch.stack(self.precision_mats).to(self._device)

    def _eval_cnn(self, loader):
        logging.info("Using FeCAM Mahalanobis classifier")
        vectors, targets = self._extract_vectors(loader)
        vectors = F.normalize(
            torch.as_tensor(vectors, device=self._device, dtype=torch.float32), dim=-1
        )
        means = F.normalize(
            self._network.fc.weight.data[: self._total_classes].float(), dim=-1
        )
        score_batches = []
        with torch.no_grad():
            for start in range(0, vectors.shape[0], self.maha_batch_size):
                batch = vectors[start : start + self.maha_batch_size]
                difference = batch[:, None, :] - means[None, :, :]
                distances = torch.einsum(
                    "bcd,cde,bce->bc",
                    difference,
                    self._precision_stack,
                    difference,
                )
                score_batches.append(distances.cpu())
        scores = torch.cat(score_batches).numpy()
        return np.argsort(scores, axis=1)[:, : self.topk], targets
