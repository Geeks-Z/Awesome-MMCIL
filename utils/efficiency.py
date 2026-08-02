"""Reproducible efficiency measurements for MMCL experiments.

The profiler is enabled only when ``MMCL_PROFILE_OUT`` is set.  It measures
wall-clock time with CUDA synchronization, process-local CUDA peaks, and the
parameters that actually have gradients immediately before an optimizer step.
"""

import json
import os
import platform
import socket
import statistics
import time

import numpy as np
import torch


def _unique_parameters(module):
    seen = set()
    for name, parameter in module.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        yield name, parameter


def _parameter_groups(module):
    groups = {}
    for name, parameter in _unique_parameters(module):
        prefix = name.split(".", 1)[0]
        row = groups.setdefault(prefix, {"parameters": 0, "requires_grad": 0})
        row["parameters"] += parameter.numel()
        if parameter.requires_grad:
            row["requires_grad"] += parameter.numel()
    return groups


def _unique_learner_parameters(learner):
    seen = set()
    for attribute, value in vars(learner).items():
        if not isinstance(value, torch.nn.Module):
            continue
        for name, parameter in value.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            yield "{}.{}".format(attribute, name), parameter


def _learner_parameter_groups(learner):
    groups = {}
    for name, parameter in _unique_learner_parameters(learner):
        prefix = name.split(".", 1)[0]
        row = groups.setdefault(prefix, {"parameters": 0, "requires_grad": 0})
        row["parameters"] += parameter.numel()
        if parameter.requires_grad:
            row["requires_grad"] += parameter.numel()
    return groups


def _auxiliary_tensor_storage(learner):
    """Count persistent tensor/array storage outside learned parameters.

    Data loaders and datasets are deliberately not traversed: dataset samples
    are inputs, not method state. Tensor aliases are deduplicated by storage.
    """
    parameter_storages = set()
    for _, parameter in _unique_learner_parameters(learner):
        if parameter.numel():
            parameter_storages.add((str(parameter.device), parameter.data_ptr()))

    seen_objects = set()
    seen_storages = set(parameter_storages)
    totals = {"tensor_bytes": 0, "numpy_bytes": 0, "tensor_values": 0}
    excluded_attributes = {
        "data_manager",
        "train_dataset",
        "train_loader",
        "test_loader",
        "test_loader_task",
        "test_loader_list",
        "train_loader_list",
        "sample_loader",
    }

    def visit(value):
        object_id = id(value)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)

        if torch.is_tensor(value):
            if not value.numel():
                return
            storage = (str(value.device), value.data_ptr())
            if storage in seen_storages:
                return
            seen_storages.add(storage)
            totals["tensor_values"] += value.numel()
            totals["tensor_bytes"] += value.numel() * value.element_size()
            return
        if isinstance(value, np.ndarray):
            totals["numpy_bytes"] += value.nbytes
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
            return
        if isinstance(value, torch.nn.Module):
            for name, item in vars(value).items():
                if name not in excluded_attributes:
                    visit(item)

    for name, value in vars(learner).items():
        if name not in excluded_attributes:
            visit(value)

    totals["total_bytes"] = totals["tensor_bytes"] + totals["numpy_bytes"]
    totals["total_mib"] = totals["total_bytes"] / 2**20
    return totals


class EfficiencyProfiler:
    """Collect efficiency measurements without changing the training path."""

    def __init__(self, args, output_path):
        self.args = args
        self.output_path = output_path
        self.train_records = []
        self.inference_records = []
        self.parameter_snapshots = []
        self.peak_updated_parameters = 0
        self.updated_parameter_ids = {}
        self._optimizer_steps = {}
        self._install_optimizer_probes()

    def _install_optimizer_probes(self):
        # PyTorch 2.0 has global optimizer hooks, but the project must also run
        # with its older environment.  Wrapping the concrete step methods keeps
        # the measurement compatible with both versions.
        classes = []
        for name in ("SGD", "Adam", "AdamW", "RMSprop", "Adagrad"):
            cls = getattr(torch.optim, name, None)
            if cls is not None:
                classes.append(cls)

        for cls in classes:
            original = cls.step
            self._optimizer_steps[cls] = original

            def measured_step(optimizer, *args, _original=original, **kwargs):
                current = {}
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        if parameter.grad is None:
                            continue
                        current[id(parameter)] = parameter.numel()
                        self.updated_parameter_ids[id(parameter)] = parameter.numel()
                self.peak_updated_parameters = max(
                    self.peak_updated_parameters, sum(current.values())
                )
                return _original(optimizer, *args, **kwargs)

            cls.step = measured_step

    def close(self):
        for cls, original in self._optimizer_steps.items():
            cls.step = original
        self._optimizer_steps.clear()

    @staticmethod
    def _sync(device):
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(self, phase, task, device, function, *args, **kwargs):
        if torch.cuda.is_available() and device.type == "cuda":
            self._sync(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        owner = getattr(function, "__self__", None)
        patched = {}
        method_name = str(self.args["model_name"]).lower()
        # Most implementations run a full test-set pass every epoch only to
        # print accuracy. Exclude that diagnostic work from training time. CLG-
        # CBM is the exception: its validation score selects/early-stops models
        # and therefore belongs to the implemented training algorithm.
        if phase == "train" and owner is not None and method_name not in {
            "clg-cbm",
            "clg_cbm",
        }:
            for name in ("_compute_accuracy", "eval_init"):
                if hasattr(owner, name):
                    patched[name] = owner.__dict__.get(name)
                    setattr(owner, name, lambda *unused_args, **unused_kwargs: float("nan"))
        try:
            result = function(*args, **kwargs)
        finally:
            if owner is not None:
                for name, previous in patched.items():
                    if previous is None:
                        delattr(owner, name)
                    else:
                        setattr(owner, name, previous)
        self._sync(device)
        seconds = time.perf_counter() - start

        row = {"task": task, "seconds": seconds}
        if torch.cuda.is_available() and device.type == "cuda":
            row.update(
                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
            )
        if phase == "train":
            self.train_records.append(row)
        elif phase == "inference":
            self.inference_records.append(row)
        else:
            raise ValueError("Unknown efficiency phase: {}".format(phase))
        return result

    def snapshot_parameters(self, task, learner):
        parameters = list(_unique_learner_parameters(learner))
        self.parameter_snapshots.append(
            {
                "task": task,
                "all_parameters": sum(p.numel() for _, p in parameters),
                "requires_grad_parameters": sum(
                    p.numel() for _, p in parameters if p.requires_grad
                ),
                "groups": _learner_parameter_groups(learner),
                "auxiliary_storage": _auxiliary_tensor_storage(learner),
            }
        )

    def write(self, learner, test_samples):
        device = self.args["device"][0]
        inference_seconds = [row["seconds"] for row in self.inference_records]
        train_peak = max(
            [row.get("peak_allocated_mib", 0.0) for row in self.train_records]
            or [0.0]
        )
        inference_peak = max(
            [row.get("peak_allocated_mib", 0.0) for row in self.inference_records]
            or [0.0]
        )
        payload = {
            "schema_version": 1,
            "method": self.args["model_name"],
            "dataset": self.args["dataset"],
            "seed": self.args["seed"],
            "protocol": {
                "init_cls": self.args["init_cls"],
                "increment": self.args["increment"],
                "training_batch_size": getattr(
                    learner, "batch_size", self.args.get("batch_size")
                ),
                "inference_batch_size": getattr(self, "inference_batch_size", None),
            },
            "environment": {
                "host": socket.gethostname(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(device)
                    if torch.cuda.is_available() and device.type == "cuda"
                    else None
                ),
            },
            "parameters": {
                "peak_actually_updated": self.peak_updated_parameters,
                "ever_updated_union": sum(self.updated_parameter_ids.values()),
                "final_snapshot": self.parameter_snapshots[-1],
                "snapshots": self.parameter_snapshots,
            },
            "training": {
                "definition": "sum of incremental_train wall time; diagnostic epoch-level test passes are excluded, while validation/model-selection passes required by the algorithm are retained",
                "task_records": self.train_records,
                "total_seconds": sum(row["seconds"] for row in self.train_records),
                "peak_allocated_mib": train_peak,
            },
            "inference": {
                "definition": "final-stage end-to-end _eval_cnn pass, including data loading and method-specific text/statistical preparation",
                "test_samples": test_samples,
                "repetitions": self.inference_records,
                "median_seconds": (
                    statistics.median(inference_seconds) if inference_seconds else None
                ),
                "images_per_second": (
                    test_samples / statistics.median(inference_seconds)
                    if inference_seconds
                    else None
                ),
                "peak_allocated_mib": inference_peak,
            },
        }

        output_dir = os.path.dirname(os.path.abspath(self.output_path))
        os.makedirs(output_dir, exist_ok=True)
        temporary = self.output_path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, self.output_path)
