#!/usr/bin/env python3
"""Summarize three complete RanPAC/FeCAM efficiency runs for the paper table."""

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "efficiency" / "raw"
OUT = ROOT / "results" / "efficiency"
METHODS = ("RanPAC", "FeCAM")
DATASETS = ("cars", "aircraft")
SUFFIXES = ("", "_run2", "_run3")
STEMS = {
    (method, dataset): "{}_{}_B0_Inc10".format(method.lower(), dataset)
    for method in METHODS
    for dataset in DATASETS
}

# Both methods gradient-tune one 64-dimensional AdaptFormer in every ViT block.
ADAPTER_PARAMETERS = 12 * (
    768 * 64 + 64  # down projection
    + 64 * 768 + 768  # up projection
)
INITIAL_HEAD_PARAMETERS = 10 * 512


def sample_stats(values):
    return {
        "runs": "|".join("{:.6f}".format(value) for value in values),
        "mean": statistics.mean(values),
        "variance": statistics.variance(values),
        "std": statistics.stdev(values),
    }


def load_profiles():
    grouped = {}
    for method in METHODS:
        for dataset in DATASETS:
            stem = STEMS[(method, dataset)]
            profiles = []
            for suffix in SUFFIXES:
                path = RAW / "{}_seed1993{}.json".format(stem, suffix)
                if not path.is_file():
                    raise FileNotFoundError(path)
                profile = json.loads(path.read_text())
                if str(profile["method"]).lower() != method.lower():
                    raise RuntimeError("Method mismatch in {}".format(path))
                if str(profile["dataset"]).lower() != dataset:
                    raise RuntimeError("Dataset mismatch in {}".format(path))
                if profile["environment"]["host"] != "comput1":
                    raise RuntimeError("Unexpected host in {}".format(path))
                if "A800 80GB" not in profile["environment"]["gpu"]:
                    raise RuntimeError("Unexpected GPU in {}".format(path))
                profiles.append(profile)
            grouped[(method, dataset)] = profiles
    return grouped


def method_state_mib(method, profile):
    """Count implementation-persistent statistics and analytic caches.

    The profiler excludes nn.Parameters. The final analytic classifier is
    therefore added here, as it is populated from statistics rather than by
    gradient optimization (the same convention used for SimpleCIL).
    """
    raw = profile["parameters"]["final_snapshot"]["auxiliary_storage"][
        "total_mib"
    ]
    if method == "RanPAC":
        analytic_head = 100 * 10_000 * 4 / 2**20
    else:
        analytic_head = 100 * 512 * 4 / 2**20
    return raw + analytic_head


def summarize():
    grouped = load_profiles()
    rows = []
    for method in METHODS:
        for dataset in DATASETS:
            profiles = grouped[(method, dataset)]
            peak_updated = {
                profile["parameters"]["peak_actually_updated"]
                for profile in profiles
            }
            if peak_updated != {ADAPTER_PARAMETERS + INITIAL_HEAD_PARAMETERS}:
                raise RuntimeError(
                    "Unexpected updated-parameter counts for {}/{}: {}".format(
                        method, dataset, sorted(peak_updated)
                    )
                )
            state_values = [method_state_mib(method, profile) for profile in profiles]
            if max(state_values) - min(state_values) > 1e-6:
                raise RuntimeError(
                    "Auxiliary state changed across runs for {}/{}: {}".format(
                        method, dataset, state_values
                    )
                )

            update = sample_stats(
                [profile["training"]["total_seconds"] / 60 for profile in profiles]
            )
            inference = sample_stats(
                [profile["inference"]["median_seconds"] for profile in profiles]
            )
            update_memory = sample_stats(
                [profile["training"]["peak_allocated_mib"] / 1024 for profile in profiles]
            )
            inference_memory = sample_stats(
                [profile["inference"]["peak_allocated_mib"] / 1024 for profile in profiles]
            )
            rows.append(
                {
                    "method": method,
                    "dataset": dataset.capitalize(),
                    "n": 3,
                    "update_minutes_runs": update["runs"],
                    "update_minutes_mean": "{:.6f}".format(update["mean"]),
                    "update_minutes_sample_variance": "{:.8f}".format(
                        update["variance"]
                    ),
                    "update_minutes_sample_std": "{:.6f}".format(update["std"]),
                    "inference_seconds_runs": inference["runs"],
                    "inference_seconds_mean": "{:.6f}".format(inference["mean"]),
                    "inference_seconds_sample_variance": "{:.8f}".format(
                        inference["variance"]
                    ),
                    "inference_seconds_sample_std": "{:.6f}".format(inference["std"]),
                    "update_peak_gib_runs": update_memory["runs"],
                    "update_peak_gib_mean": "{:.6f}".format(update_memory["mean"]),
                    "update_peak_gib_sample_variance": "{:.8f}".format(
                        update_memory["variance"]
                    ),
                    "update_peak_gib_sample_std": "{:.6f}".format(
                        update_memory["std"]
                    ),
                    "inference_peak_gib_runs": inference_memory["runs"],
                    "inference_peak_gib_mean": "{:.6f}".format(
                        inference_memory["mean"]
                    ),
                    "inference_peak_gib_sample_variance": "{:.8f}".format(
                        inference_memory["variance"]
                    ),
                    "inference_peak_gib_sample_std": "{:.6f}".format(
                        inference_memory["std"]
                    ),
                    "peak_updated_M": "{:.6f}".format(
                        (ADAPTER_PARAMETERS + INITIAL_HEAD_PARAMETERS) / 1e6
                    ),
                    "final_retained_M": "{:.6f}".format(
                        ADAPTER_PARAMETERS / 1e6
                    ),
                    "extra_frozen_M": "0.000000",
                    "auxiliary_state_MiB": "{:.6f}".format(state_values[0]),
                }
            )
    return rows


def paired_latex(rows):
    by_key = {(row["method"], row["dataset"]): row for row in rows}
    lines = []
    for method in METHODS:
        cars = by_key[(method, "Cars")]
        aircraft = by_key[(method, "Aircraft")]
        lines.append(
            "{method:<12} & {updated:.2f} & {retained:.2f} & 0.00 & {state:.2f} & "
            "${cu:.2f}\\!\\pm\\!{cs:.2f}$ / ${au:.2f}\\!\\pm\\!{ass:.2f}$ & "
            "${ci:.2f}\\!\\pm\\!{cis:.2f}$ / ${ai:.2f}\\!\\pm\\!{ais:.2f}$ & "
            "{cum:.2f} / {aum:.2f} & {cim:.2f} / {aim:.2f} \\\\".format(
                method=method,
                updated=float(cars["peak_updated_M"]),
                retained=float(cars["final_retained_M"]),
                state=max(
                    float(cars["auxiliary_state_MiB"]),
                    float(aircraft["auxiliary_state_MiB"]),
                ),
                cu=float(cars["update_minutes_mean"]),
                cs=float(cars["update_minutes_sample_std"]),
                au=float(aircraft["update_minutes_mean"]),
                ass=float(aircraft["update_minutes_sample_std"]),
                ci=float(cars["inference_seconds_mean"]),
                cis=float(cars["inference_seconds_sample_std"]),
                ai=float(aircraft["inference_seconds_mean"]),
                ais=float(aircraft["inference_seconds_sample_std"]),
                cum=float(cars["update_peak_gib_mean"]),
                aum=float(aircraft["update_peak_gib_mean"]),
                cim=float(cars["inference_peak_gib_mean"]),
                aim=float(aircraft["inference_peak_gib_mean"]),
            )
        )
    return lines


def main():
    rows = summarize()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "ranpac_fecam_repeated_statistics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    latex_path = OUT / "ranpac_fecam_table_rows.tex"
    latex_path.write_text("\n".join(paired_latex(rows)) + "\n")
    print("Wrote {}".format(csv_path))
    print("Wrote {}".format(latex_path))
    print(latex_path.read_text(), end="")


if __name__ == "__main__":
    main()
