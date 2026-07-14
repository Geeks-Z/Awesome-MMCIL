import logging
import random
import re
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.data_manager import DataManager
from utils.inc_net import Area
from utils.toolkit import get_attribute


_DESCRIPTION_DIRS = {
    "aircraft": "Aircraft",
    "cars": "Car",
    "cifar224": "cifar100",
    "cub": "CUB",
    "food101": "Food",
    "imagenetr": "imagenetr",
    "objectnet": "objectnet",
    "sun": "sun",
    "ucf101": "ucf",
}


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = Area(args)

        self.init_lr = get_attribute(args, "init_lr", 0.001)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0001)
        self.milestones = get_attribute(args, "milestones", [7, 15])
        self.gamma = get_attribute(args, "scheduler_gamma", 0.1)
        self.epochs = get_attribute(args, "epochs", 20)
        self.batch_size = get_attribute(args, "batch_size", 128)
        self.samples_per_class_proto = get_attribute(args, "samples_per_class", 4)
        self.K = get_attribute(args, "K", 16)
        self.vib_lambda = get_attribute(args, "vib_lambda", 1.0)
        self.g_lambda = get_attribute(args, "g_lambda", 0.0)
        self.num_aug_views = get_attribute(args, "num_aug_views", 3)
        self.num_workers = get_attribute(args, "num_workers", 8)
        self.description_batch_size = get_attribute(
            args, "description_batch_size", 128
        )
        self.precomputed_basis_path = get_attribute(
            args, "precomputed_basis_path", None
        )

        self.text_des_path, self.occ_des_path, self.aug_des_path = (
            self._get_description_paths(args)
        )
        self._description_indices = {}
        self.classnames = None
        self.text_features = None
        self.visual_base_matrices = None
        self.textual_base_matrices = None
        self.task_sizes = [0]

    @property
    def samples_per_class(self):
        return self.samples_per_class_proto

    def after_task(self):
        self._known_classes = self._total_classes

    @staticmethod
    def _get_description_paths(args):
        description_root = args.get("description_root")
        if description_root is None:
            dataset_dir = _DESCRIPTION_DIRS.get(args["dataset"])
            if dataset_dir is None:
                raise ValueError(
                    "AREA has no bundled descriptions for dataset {!r}. "
                    "Set description_root or the three *_des_path options.".format(
                        args["dataset"]
                    )
                )
            description_root = (
                Path(__file__).resolve().parents[1]
                / "utils"
                / "area"
                / "descriptions"
                / dataset_dir
            )
        else:
            description_root = Path(description_root).expanduser()

        paths = (
            Path(args.get("text_des_path", description_root / "generated_descriptions")),
            Path(args.get("occ_des_path", description_root / "generated_descriptions_occ")),
            Path(args.get("aug_des_path", description_root / "generated_descriptions_aug")),
        )
        missing = [str(path) for path in paths if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                "AREA description directories are missing: {}".format(", ".join(missing))
            )
        return paths

    @staticmethod
    def _normalize_classname(name):
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def _description_file(self, directory, classname):
        directory = Path(directory)
        if directory not in self._description_indices:
            index = {}
            for path in directory.glob("*_descriptions.txt"):
                stem = path.name[: -len("_descriptions.txt")]
                # CUB description files use prefixes such as ``001.``.
                # Numeric model names such as ``707-320`` must remain intact.
                stem = re.sub(r"^\d{3}\.", "", stem)
                index[self._normalize_classname(stem)] = path
            self._description_indices[directory] = index

        key = self._normalize_classname(classname)
        try:
            return self._description_indices[directory][key]
        except KeyError as exc:
            raise FileNotFoundError(
                "No AREA description for class {!r} in {}".format(classname, directory)
            ) from exc

    def _read_descriptions(self, directory, classname):
        try:
            path = self._description_file(directory, classname)
        except FileNotFoundError:
            # The released corpus omits a small number of augmentation files.
            # Use the corresponding clean description instead of aborting training.
            if Path(directory) == self.text_des_path:
                raise
            path = self._description_file(self.text_des_path, classname)
        with path.open("r", encoding="utf-8") as handle:
            descriptions = [line.strip() for line in handle if line.strip()]
        if not descriptions:
            raise ValueError("AREA description file is empty: {}".format(path))
        return descriptions

    @torch.no_grad()
    def _get_class_name_features(self, data_manager):
        # DataManager already exposes labels in the shuffled incremental order.
        self.classnames = list(data_manager._class_to_label)
        template = self.args.get("class_prompt", "a good photo of a {}.")
        texts = [template.format(name.replace("_", " ")) for name in self.classnames]
        tokenized = self._network.tokenizer(texts).to(self._device)
        self.text_features = self._network.encode_text(tokenized)

    @torch.no_grad()
    def _encode_description_texts(self, descriptions):
        features = []
        for start in range(0, len(descriptions), self.description_batch_size):
            batch = descriptions[start : start + self.description_batch_size]
            tokenized = self._network.tokenizer(batch).to(self._device)
            features.append(self._network.encode_text(tokenized).cpu())
        return torch.cat(features)

    @staticmethod
    @torch.no_grad()
    def log_map(x, mu):
        cos_theta = torch.matmul(x, mu.reshape(-1)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        theta = torch.acos(cos_theta)
        sin_theta = torch.sqrt(1 - cos_theta.square())
        scale = torch.ones_like(theta)
        stable = theta > 1e-4
        scale[stable] = theta[stable] / (sin_theta[stable] + 1e-6)
        return scale.unsqueeze(1) * (
            x - cos_theta.unsqueeze(1) * mu.reshape(1, -1)
        )

    @torch.no_grad()
    def _get_visual_base_matrix(self, data_manager):
        logging.info("Computing AREA visual base matrices")
        sample_dataset = data_manager.get_dataset(
            range(data_manager.get_total_classnum()), "train", "train"
        )
        sample_loader = DataLoader(
            sample_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )
        sample_data = [[] for _ in range(data_manager.get_total_classnum())]
        for _, inputs, targets in tqdm(sample_loader, desc="Visual bases"):
            features = self._network.encode_image(inputs.to(self._device)).cpu()
            sample_data[targets.item()].append(features)

        dimension = self._network.feature_dim
        matrices = torch.zeros(
            data_manager.get_total_classnum(), dimension, self.K, device=self._device
        )
        for label, features in enumerate(sample_data):
            if not features:
                raise ValueError("No training samples found for class {}".format(label))
            data = torch.cat(features, dim=0)
            mean = F.normalize(data.mean(dim=0, keepdim=True), dim=-1)
            tangent = self.log_map(data, mean)
            _, _, vh = torch.linalg.svd(tangent, full_matrices=False)
            rank = min(self.K, vh.shape[0])
            matrices[label, :, :rank] = vh[:rank].t().to(self._device)
        self.visual_base_matrices = matrices

    @torch.no_grad()
    def _get_textual_base_matrix(self, data_manager):
        logging.info("Computing AREA textual base matrices")
        dimension = self._network.feature_dim
        matrices = torch.zeros(
            data_manager.get_total_classnum(), dimension, self.K, device=self._device
        )
        for index, classname in enumerate(tqdm(self.classnames, desc="Textual bases")):
            descriptions = self._read_descriptions(self.text_des_path, classname)
            features = self._encode_description_texts(descriptions)
            mean = F.normalize(features.mean(dim=0, keepdim=True), dim=-1)
            tangent = self.log_map(features, mean)
            _, _, vh = torch.linalg.svd(tangent, full_matrices=False)
            rank = min(self.K, vh.shape[0])
            matrices[index, :, :rank] = vh[:rank].t().to(self._device)
        self.textual_base_matrices = matrices

    def _basis_cache_file(self, name):
        if self.precomputed_basis_path is None:
            return None
        seed_dir = "seed_{}".format(self.args["seed"])
        return Path(self.precomputed_basis_path).expanduser() / seed_dir / (name + ".pth")

    def _load_basis(self, name, data_manager):
        cache_file = self._basis_cache_file(name)
        if cache_file is None or not cache_file.exists():
            return None
        logging.info("Loading AREA basis cache from %s", cache_file)
        payload = torch.load(str(cache_file), map_location="cpu")
        matrix = payload[name]
        expected = (data_manager.get_total_classnum(), self._network.feature_dim, self.K)
        if tuple(matrix.shape) != expected:
            raise ValueError(
                "AREA cache {} has shape {}, expected {}".format(
                    cache_file, tuple(matrix.shape), expected
                )
            )
        cached_names = payload.get("classnames")
        if cached_names is not None and list(cached_names) != self.classnames:
            raise ValueError(
                "AREA cache {} was created for a different class order".format(cache_file)
            )
        return matrix.to(self._device)

    def _save_basis(self, name, matrix):
        cache_file = self._basis_cache_file(name)
        if cache_file is None:
            return
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {name: matrix.cpu(), "classnames": self.classnames}, str(cache_file)
        )

    def incremental_train(self, data_manager: DataManager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self.task_sizes.append(self._total_classes)
        self._network.append_S(self._device)
        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.data_manager = data_manager
        self._network.to(self._device)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(self._total_classes), source="test", mode="test"
        )
        # AREA routes every sample to a task, so inference is intentionally single-sample.
        self.test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._network = self._network.module

        if self.text_features is None:
            self._get_class_name_features(data_manager)
            self.text_features = self.text_features.to(self._device)

        if self.textual_base_matrices is None:
            self.textual_base_matrices = self._load_basis(
                "textual_base_matrices", data_manager
            )
            if self.textual_base_matrices is None:
                self._get_textual_base_matrix(data_manager)
                self._save_basis(
                    "textual_base_matrices", self.textual_base_matrices
                )

        if self.visual_base_matrices is None:
            self.visual_base_matrices = self._load_basis(
                "visual_base_matrices", data_manager
            )
            if self.visual_base_matrices is None:
                self._get_visual_base_matrix(data_manager)
                self._save_basis("visual_base_matrices", self.visual_base_matrices)

        self._network.update_stat(
            self._known_classes,
            self._total_classes,
            self.train_loader,
            self._device,
        )
        self._train_area(self.train_loader)
        self._update_replay_statistics()

    def _train_area(self, train_loader):
        augmentation = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        )
        trainable = [parameter for parameter in self._network.parameters() if parameter.requires_grad]
        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(
                trainable,
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.weight_decay,
            )
        elif self.args["optimizer"] == "adam":
            optimizer = optim.AdamW(
                trainable, lr=self.init_lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError("Unsupported AREA optimizer: {}".format(self.args["optimizer"]))
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.gamma
        )

        progress = tqdm(range(self.epochs))
        for epoch in progress:
            self._network.train()
            old_classes = list(range(self._known_classes))
            random.shuffle(old_classes)
            last_loss = last_cls_loss = 0.0
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                real_targets = targets.clone()

                memory_data = None
                if old_classes:
                    memory_features, memory_targets = [], []
                    for label in old_classes:
                        memory_features.append(
                            self.sample(
                                self._network.class_mean_list[label],
                                self._network.class_cov_list[label],
                                int(self.samples_per_class),
                            )
                        )
                        memory_targets.append(
                            torch.full(
                                (int(self.samples_per_class),),
                                label,
                                dtype=torch.long,
                                device=self._device,
                            )
                        )
                    memory_data = torch.cat(memory_features)
                    targets = torch.cat([targets, torch.cat(memory_targets)])

                outputs = self._network(
                    inputs,
                    self.text_features[: self._total_classes],
                    self.visual_base_matrices[: self._total_classes],
                    self.textual_base_matrices[: self._total_classes],
                    self._cur_task,
                    memory_data=memory_data,
                )
                classification_loss = F.cross_entropy(outputs, targets)
                classnames = [self.classnames[index] for index in real_targets.tolist()]

                clean_visual_scores = self._network._get_visual_score(
                    inputs, self._cur_task
                )
                occluded_visual_scores = self._network._get_visual_score(
                    self._generate_occluded_inputs(inputs), self._cur_task
                )
                occluded_texts = [
                    random.choice(self._read_descriptions(self.occ_des_path, name))
                    for name in classnames
                ]
                clean_texts = ["a photo of a {}.".format(name) for name in classnames]
                occluded_text_scores = self._network._get_textual_score(
                    occluded_texts, self._cur_task
                )
                clean_text_scores = self._network._get_textual_score(
                    clean_texts, self._cur_task
                )
                mask_loss = torch.relu(
                    occluded_visual_scores - clean_visual_scores
                ).mean() + torch.relu(occluded_text_scores - clean_text_scores).mean()

                visual_views, textual_views = [], []
                for _ in range(self.num_aug_views):
                    augmented_texts = [
                        random.choice(self._read_descriptions(self.aug_des_path, name))
                        for name in classnames
                    ]
                    visual_views.append(
                        self._network._get_visual_score(
                            augmentation(inputs), self._cur_task
                        )
                    )
                    textual_views.append(
                        self._network._get_textual_score(
                            augmented_texts, self._cur_task
                        )
                    )
                mean_visual = torch.stack(visual_views).mean(dim=0)
                mean_textual = torch.stack(textual_views).mean(dim=0)
                consistency_loss = sum(
                    F.l1_loss(view, mean_visual) for view in visual_views
                ) + sum(F.l1_loss(view, mean_textual) for view in textual_views)

                loss = classification_loss + self.vib_lambda * (
                    mask_loss + consistency_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = loss.item()
                last_cls_loss = classification_loss.item()

            scheduler.step()
            progress.set_description(
                "Epoch [{}/{}] Loss: {:.4f} Cls Loss: {:.4f}".format(
                    epoch + 1, self.epochs, last_loss, last_cls_loss
                )
            )

    @torch.no_grad()
    def _update_replay_statistics(self):
        sample_loader = DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=self.num_workers,
        )
        features, targets = [], []
        for _, inputs, labels in sample_loader:
            features.append(self._network.encode_image(inputs.to(self._device)))
            targets.append(labels.to(self._device))
        self._network.analyze_mean_cov(torch.cat(features), torch.cat(targets))

    @staticmethod
    def sample(mean, cov, size):
        vectors = torch.randn(size, mean.shape[-1], device=mean.device)
        sqrt_cov = torch.linalg.cholesky(cov.cpu()).to(mean.device)
        return vectors @ sqrt_cov.t() + mean

    @torch.no_grad()
    def get_most_similar_task(self, inputs):
        if inputs.shape[0] != 1:
            raise ValueError("AREA task routing expects a single image")
        image_features = self._network.encode_image(inputs.to(self._device))
        distances = []
        for task_id in range(self._cur_task + 1):
            visual_basis = self.visual_base_matrices[
                self.task_sizes[task_id] : self.task_sizes[task_id + 1]
            ]
            task_basis = visual_basis.permute(0, 2, 1).reshape(
                -1, visual_basis.shape[1]
            )
            cost_matrix = self._compute_cost_matrix(image_features, task_basis)
            distances.append(self._sinkhorn_distance(cost_matrix))
        return torch.stack(distances).reshape(-1).argmax().item()

    @torch.no_grad()
    def _eval_cnn(self, loader):
        self._network.eval()
        predictions, ground_truth = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            task_id = self.get_most_similar_task(inputs)
            outputs = self._network.forward_inference(
                inputs,
                self.text_features[: self._total_classes],
                self.visual_base_matrices[: self._total_classes],
                self.textual_base_matrices[: self._total_classes],
                task_id,
            )
            raw_features = F.normalize(self._network.visual_forward_(inputs), dim=-1)
            gda_outputs = raw_features @ self._network.W + self._network.b
            outputs = (1 - self.g_lambda) * outputs + self.g_lambda * gda_outputs
            predictions.append(self._select_topk(outputs).cpu().numpy())
            ground_truth.append(targets.cpu().numpy())
        return np.concatenate(predictions), np.concatenate(ground_truth)

    @staticmethod
    def _generate_occluded_inputs(inputs):
        batch, channels, height, width = inputs.shape
        occluded = inputs.clone()
        for index in range(batch):
            ratio = np.random.uniform(0.1, 0.4)
            aspect = np.random.uniform(0.33, 3.0)
            area = int(ratio * height * width)
            patch_height = int(np.clip(np.sqrt(area * aspect), 1, height))
            patch_width = int(np.clip(np.sqrt(area / aspect), 1, width))
            top = np.random.randint(0, height - patch_height + 1)
            left = np.random.randint(0, width - patch_width + 1)
            occluded[
                index,
                :,
                top : top + patch_height,
                left : left + patch_width,
            ] = torch.rand(
                (channels, patch_height, patch_width), device=inputs.device
            )
        return occluded

    @staticmethod
    def _compute_cost_matrix(query_embedding, task_basis):
        query = F.normalize(query_embedding, p=2, dim=1)
        basis = F.normalize(task_basis, p=2, dim=1)
        return 1.0 - query @ basis.t()

    @staticmethod
    def _sinkhorn_distance(cost_matrix):
        batch, basis_count = cost_matrix.shape
        mu = torch.ones(batch, 1, device=cost_matrix.device)
        nu = torch.full(
            (batch, basis_count),
            1.0 / basis_count,
            device=cost_matrix.device,
        )
        f = torch.zeros_like(mu)
        g = torch.zeros_like(nu)
        log_kernel = -cost_matrix / 0.1
        for _ in range(50):
            f = torch.log(mu) - torch.logsumexp(g + log_kernel, dim=1, keepdim=True)
            g = torch.log(nu) - (f + log_kernel)
        return (f * mu).sum(dim=1) + (g * nu).sum(dim=1)
