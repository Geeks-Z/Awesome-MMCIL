#!/usr/bin/env python3
"""Create paper-ready CSV/Markdown tables from efficiency profiler outputs."""

import csv
import glob
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "efficiency" / "raw"
OUT = ROOT / "results" / "efficiency"
# The common OpenCLIP text transformer's 77 x 77 float32 attention mask is an
# implementation buffer, not method-specific continual-learning state.
COMMON_CLIP_BUFFER_MIB = 77 * 77 * 4 / 2**20
FULL_OPENCLIP_PARAMETERS = 149_620_737
FULL_OPENCLIP_FP32_MIB = FULL_OPENCLIP_PARAMETERS * 4 / 2**20

METHOD_ORDER = [
    "ZS-CLIP",
    "L2P",
    "DualPrompt",
    "CODA-Prompt",
    "SimpleCIL",
    "RAPF",
    "ENGINE",
    "CLG-CBM",
    "BOFA",
]

ALIASES = {
    "zs_clip": "ZS-CLIP",
    "l2p": "L2P",
    "dualprompt": "DualPrompt",
    "coda": "CODA-Prompt",
    "simplecil": "SimpleCIL",
    "rapf": "RAPF",
    "engine": "ENGINE",
    "clg-cbm": "CLG-CBM",
    "clg_cbm": "CLG-CBM",
    "bofa": "BOFA",
}


def retained_parameters(method, classes=100, tasks=10, pool=10):
    """Final learned parameters needed by a minimal inference export.

    One common frozen CLIP is excluded. Statistical classifiers/prototypes are
    auxiliary state, not learned parameters. BOFA assumes its verified fusion
    export, but retains the task heads used by its inference rule.
    """
    return {
        "ZS-CLIP": 0,
        "L2P": 43_520 + classes * (512 + 1),
        "DualPrompt": 235_520 + classes * (512 + 1),
        "CODA-Prompt": 3_584_000 + classes * (512 + 1),
        "SimpleCIL": 0,
        "RAPF": 512 * 512,
        "ENGINE": tasks * 2 * (512 * 512 + 512),
        "CLG-CBM": (512 + classes) * (pool * classes),
        # W_fusion replaces the baseline visual projection at export, and the
        # task heads are not consumed by the final _eval_cnn path.
        "BOFA": 0,
    }[method]


def growth_rule(method):
    return {
        "ZS-CLIP": "O(1) learned; optional O(CD) text cache",
        "L2P": "fixed prompt pool + O(CD) head",
        "DualPrompt": "fixed prompt pool + O(CD) head",
        "CODA-Prompt": "fixed allocated pool + O(CD) head",
        "SimpleCIL": "O(CD) prototypes",
        "RAPF": "O(D^2) adapter + O(CD^2) statistics",
        "ENGINE": "O(TD^2) adapters + O(CV + V^2) statistics",
        "CLG-CBM": "O(pCD + pC^2) learned + O(CD^2) statistics",
        "BOFA": "O(1) extra after fusion; state O(TV^2 + TVD + CV)",
    }[method]


def export_note(method):
    return {
        "ZS-CLIP": "no optimization",
        "L2P": "prompt pool and 100-class head retained",
        "DualPrompt": "global/expert prompts and 100-class head retained",
        "CODA-Prompt": "allocated prompt pool and 100-class head retained",
        "SimpleCIL": "class prototypes are state, not gradient-learned parameters",
        "RAPF": "final 512x512 adapter retained; old adapter can be pruned",
        "ENGINE": "two adapters per task retained; unused BaseNet CLIP copy is prunable",
        "CLG-CBM": "final concept classifier retained",
        "BOFA": "fused projection replaces baseline projection; unused task heads pruned",
    }[method]


def training_schedule(method):
    return {
        "ZS-CLIP": "no optimizer",
        "L2P": "10 epochs/task",
        "DualPrompt": "10 epochs/task",
        "CODA-Prompt": "10 epochs/task",
        "SimpleCIL": "prototype estimation",
        "RAPF": "10 epochs/task",
        "ENGINE": "10 epochs/task",
        "CLG-CBM": "60 epochs/task with model selection",
        "BOFA": "15 stage-1 + 2 stage-2 epochs/task",
    }[method]


def load_profiles():
    rows = {}
    pattern = str(RAW / "*_B0_Inc10_seed1993.json")
    for filename in glob.glob(pattern):
        with open(filename) as handle:
            profile = json.load(handle)
        method = ALIASES[str(profile["method"]).lower()]
        rows[(method, profile["dataset"])] = profile
    return rows


def auxiliary_mib(method, profile):
    snapshot = profile["parameters"]["final_snapshot"]
    raw = snapshot.get("auxiliary_storage", {}).get("total_mib", 0.0)
    measured = max(0.0, raw - COMMON_CLIP_BUFFER_MIB)
    if method == "ENGINE":
        # Engine calls BaseNet.__init__, which stores an unregistered and unused
        # (model, preprocess, tokenizer) tuple, then loads the actual model a
        # second time. Classify that copy as frozen parameters, not statistics.
        measured = max(
            0.0,
            measured - FULL_OPENCLIP_FP32_MIB - COMMON_CLIP_BUFFER_MIB,
        )
    if method == "SimpleCIL":
        # Its prototype classifier is stored as an nn.Parameter but populated
        # by class means rather than gradient optimization.
        measured += 100 * 512 * 4 / 2**20
    return measured


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    profiles = load_profiles()
    parameter_rows = []
    timing_rows = []

    for method in METHOD_ORDER:
        method_profiles = [
            profiles[(method, dataset)]
            for dataset in ("cars", "aircraft")
            if (method, dataset) in profiles
        ]
        peak_updated = [
            profile["parameters"]["peak_actually_updated"]
            for profile in method_profiles
        ]
        auxiliary = [auxiliary_mib(method, profile) for profile in method_profiles]
        parameter_rows.append(
            {
                "method": method,
                "peak_updated_M": (
                    "{:.6f}".format(max(peak_updated) / 1e6) if peak_updated else ""
                ),
                "final_extra_learned_M": "{:.6f}".format(
                    retained_parameters(method) / 1e6
                ),
                "auxiliary_state_MiB": (
                    "{:.3f}".format(max(auxiliary)) if auxiliary else ""
                ),
                "extra_frozen_M": (
                    "86.192640"
                    if method in {"L2P", "DualPrompt"}
                    else "149.620737" if method == "ENGINE" else "0.000000"
                ),
                "extra_frozen_copy": (
                    "1x CLIP ViT-B/16 visual query encoder"
                    if method in {"L2P", "DualPrompt"}
                    else (
                        "1x unused full OpenCLIP (implementation artifact; prune)"
                        if method == "ENGINE"
                        else "none"
                    )
                ),
                "growth": growth_rule(method),
                "export_note": export_note(method),
            }
        )

        for dataset in ("cars", "aircraft"):
            profile = profiles.get((method, dataset))
            if profile is None:
                continue
            timing_rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "gpu": profile["environment"]["device"],
                    "train_batch": profile["protocol"]["training_batch_size"],
                    "train_schedule": training_schedule(method),
                    "train_seconds": "{:.3f}".format(
                        profile["training"]["total_seconds"]
                    ),
                    "train_minutes": "{:.2f}".format(
                        profile["training"]["total_seconds"] / 60
                    ),
                    "train_peak_MiB": "{:.1f}".format(
                        profile["training"]["peak_allocated_mib"]
                    ),
                    "infer_batch": profile["protocol"]["inference_batch_size"],
                    "infer_median_seconds": "{:.3f}".format(
                        profile["inference"]["median_seconds"]
                    ),
                    "infer_min_max_seconds": "{:.3f}-{:.3f}".format(
                        min(
                            row["seconds"]
                            for row in profile["inference"]["repetitions"]
                        ),
                        max(
                            row["seconds"]
                            for row in profile["inference"]["repetitions"]
                        ),
                    ),
                    "images_per_second": "{:.2f}".format(
                        profile["inference"]["images_per_second"]
                    ),
                    "infer_peak_MiB": "{:.1f}".format(
                        profile["inference"]["peak_allocated_mib"]
                    ),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(
        OUT / "parameter_summary.csv",
        list(parameter_rows[0]),
        parameter_rows,
    )
    if timing_rows:
        write_csv(OUT / "timing_memory_summary.csv", list(timing_rows[0]), timing_rows)

    completed = len(timing_rows)
    print("Loaded {}/18 method-dataset profiles".format(completed))
    print("Wrote {}".format(OUT / "parameter_summary.csv"))
    if timing_rows:
        print("Wrote {}".format(OUT / "timing_memory_summary.csv"))


if __name__ == "__main__":
    main()
