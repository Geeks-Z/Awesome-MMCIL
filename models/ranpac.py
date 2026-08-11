import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import AdapterVitNet
from utils.toolkit import get_attribute, target2onehot


class Learner(BaseLearner):
    """RanPAC with a LAION-400M OpenCLIP ViT-B/16 AdaptFormer backbone."""

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
        self.use_rp = get_attribute(args, "use_RP", True)
        self.rp_dim = get_attribute(args, "M", 10000)
        self.ridge_sample_size = get_attribute(args, "ridge_sample_size", 1024)

        self.W_rand = None
        self.Q = None
        self.G = None
        self._all_classes = None

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._all_classes = data_manager.get_total_classnum()
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
            shuffle=True,
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
            if self.use_rp:
                self._setup_random_projection()
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
        raise ValueError("Unsupported RanPAC optimizer: {}".format(self.optimizer_name))

    def _tune_first_session(self):
        trainable = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info("RanPAC first-session trainable parameters: %s", trainable)
        optimizer = self._make_optimizer()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.tuned_epoch), eta_min=self.min_lr
        )
        progress = tqdm(range(self.tuned_epoch), desc="RanPAC adapter tuning")
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
                "RanPAC task 0 epoch {}/{} loss {:.3f} acc {:.2f}".format(
                    epoch + 1,
                    self.tuned_epoch,
                    losses / max(1, len(self.train_loader)),
                    100.0 * correct / max(1, total),
                )
            )

    def _setup_random_projection(self):
        self._network.RP_dim = self.rp_dim
        self.W_rand = torch.randn(
            self._network.feature_dim, self.rp_dim, device=self._device
        )
        self._network.W_rand = self.W_rand
        self._network.fc = self._network.generate_fc(
            self.rp_dim, self._total_classes
        ).to(self._device)
        self._network.fc.weight.requires_grad_(False)
        self.Q = torch.zeros(
            self.rp_dim, self._all_classes, device=self._device
        )
        self.G = torch.zeros(self.rp_dim, self.rp_dim, device=self._device)
        logging.info(
            "Initialized RanPAC random projection: %s -> %s",
            self._network.feature_dim,
            self.rp_dim,
        )

    @torch.no_grad()
    def _collect_features(self, loader):
        self._network.eval()
        features = []
        labels = []
        for _, inputs, targets in loader:
            base_features = self._network.extract_vector(inputs.to(self._device))
            if self.W_rand is not None:
                base_features = F.relu(base_features @ self.W_rand)
            features.append(base_features)
            labels.append(targets.to(self._device))
        return torch.cat(features), torch.cat(labels)

    @torch.no_grad()
    def _replace_fc(self, loader):
        features, labels = self._collect_features(loader)
        if not self.use_rp:
            for class_index in labels.unique().tolist():
                self._network.fc.weight.data[class_index] = features[
                    labels == class_index
                ].mean(0)
            return

        targets = target2onehot(labels, self._all_classes)
        self.Q.add_(features.T @ targets)
        self.G.add_(features.T @ features)
        ridge = self._optimise_ridge_parameter(features, targets)

        system = self.G.clone()
        solved = False
        for _ in range(8):
            system.copy_(self.G)
            system.diagonal().add_(ridge)
            factor, info = torch.linalg.cholesky_ex(system)
            if int(info.max().item()) == 0:
                weights = torch.cholesky_solve(self.Q, factor).T
                solved = True
                break
            ridge *= 10.0
        if not solved:
            weights = torch.linalg.solve(system, self.Q).T
        self._network.fc.weight.data.copy_(weights[: self._total_classes])
        logging.info("RanPAC ridge parameter: %g", ridge)

    @torch.no_grad()
    def _optimise_ridge_parameter(self, features, targets):
        sample_count = min(features.shape[0], self.ridge_sample_size)
        if sample_count < 2:
            return 1.0
        features = features[:sample_count]
        targets = targets[:sample_count]
        split = min(sample_count - 1, max(1, int(sample_count * 0.8)))
        fit_features, val_features = features[:split], features[split:]
        fit_targets, val_targets = targets[:split], targets[split:]

        kernel = fit_features @ fit_features.T
        eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
        eigenvalues = eigenvalues.clamp_min_(0)
        projected_targets = eigenvectors.T @ fit_targets
        losses = []
        ridges = 10.0 ** np.arange(-8, 9)
        for ridge in ridges:
            dual = eigenvectors @ (
                projected_targets / (eigenvalues[:, None] + float(ridge))
            )
            weights = fit_features.T @ dual
            prediction = val_features @ weights
            losses.append(F.mse_loss(prediction, val_targets).item())
        return float(ridges[int(np.argmin(losses))])
