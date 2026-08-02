#!/usr/bin/env python3
"""Summarize three complete baseline runs and merge SAGLA statistics."""

import csv
import json
import statistics
from pathlib import Path

from summarize_efficiency import ALIASES, METHOD_ORDER


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "efficiency" / "raw"
OUT = ROOT / "results" / "efficiency"
SAGLA_CSV = Path(
    "/public/home/hanlida/Dr.1/Code/PaperCode/MMCL/SAGLA/"
    "results/efficiency/repeated_run_statistics.csv"
)
RUN_SUFFIXES = ("", "_run2", "_run3")
DATASETS = ("cars", "aircraft")


def sample_stats(values):
    return {
        "runs": "|".join("{:.6f}".format(value) for value in values),
        "mean": statistics.mean(values),
        "variance": statistics.variance(values),
        "std": statistics.stdev(values),
    }


def profile_path(stem, suffix):
    return RAW / "{}_seed1993{}.json".format(stem, suffix)


def load_baseline_profiles():
    grouped = {}
    for suffix in RUN_SUFFIXES:
        pattern = "*_B0_Inc10_seed1993{}.json".format(suffix)
        for path in RAW.glob(pattern):
            profile = json.loads(path.read_text())
            method = ALIASES[str(profile["method"]).lower()]
            dataset = str(profile["dataset"]).lower()
            grouped.setdefault((method, dataset), {})[suffix] = profile

    expected = {(method, dataset) for method in METHOD_ORDER for dataset in DATASETS}
    if set(grouped) != expected:
        missing = sorted(expected - set(grouped))
        extra = sorted(set(grouped) - expected)
        raise RuntimeError("Profile groups differ: missing={} extra={}".format(missing, extra))
    for key, profiles in grouped.items():
        if set(profiles) != set(RUN_SUFFIXES):
            raise RuntimeError("{} does not have three runs: {}".format(key, profiles))
    return grouped


def load_parameter_rows():
    with (OUT / "parameter_summary.csv").open() as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def baseline_rows():
    grouped = load_baseline_profiles()
    parameters = load_parameter_rows()
    rows = []
    for method in METHOD_ORDER:
        for dataset in DATASETS:
            profiles = [grouped[(method, dataset)][suffix] for suffix in RUN_SUFFIXES]
            train = sample_stats(
                [profile["training"]["total_seconds"] / 60 for profile in profiles]
            )
            inference = sample_stats(
                [profile["inference"]["median_seconds"] for profile in profiles]
            )
            throughput = sample_stats(
                [profile["inference"]["images_per_second"] for profile in profiles]
            )
            train_memory = sample_stats(
                [profile["training"]["peak_allocated_mib"] / 1024 for profile in profiles]
            )
            infer_memory = sample_stats(
                [profile["inference"]["peak_allocated_mib"] / 1024 for profile in profiles]
            )
            param = parameters[method]
            rows.append(
                {
                    "method": method,
                    "dataset": dataset.capitalize(),
                    "n": 3,
                    "train_minutes_runs": train["runs"],
                    "train_minutes_mean": "{:.6f}".format(train["mean"]),
                    "train_minutes_sample_variance": "{:.8f}".format(train["variance"]),
                    "train_minutes_sample_std": "{:.6f}".format(train["std"]),
                    "inference_seconds_runs": inference["runs"],
                    "inference_seconds_mean": "{:.6f}".format(inference["mean"]),
                    "inference_seconds_sample_variance": "{:.8f}".format(
                        inference["variance"]
                    ),
                    "inference_seconds_sample_std": "{:.6f}".format(inference["std"]),
                    "images_per_second_runs": throughput["runs"],
                    "images_per_second_mean": "{:.6f}".format(throughput["mean"]),
                    "images_per_second_sample_variance": "{:.8f}".format(
                        throughput["variance"]
                    ),
                    "images_per_second_sample_std": "{:.6f}".format(
                        throughput["std"]
                    ),
                    "train_peak_gib_runs": train_memory["runs"],
                    "train_peak_gib_mean": "{:.6f}".format(train_memory["mean"]),
                    "train_peak_gib_sample_variance": "{:.8f}".format(
                        train_memory["variance"]
                    ),
                    "train_peak_gib_sample_std": "{:.6f}".format(
                        train_memory["std"]
                    ),
                    "inference_peak_gib_runs": infer_memory["runs"],
                    "inference_peak_gib_mean": "{:.6f}".format(infer_memory["mean"]),
                    "inference_peak_gib_sample_variance": "{:.8f}".format(
                        infer_memory["variance"]
                    ),
                    "inference_peak_gib_sample_std": "{:.6f}".format(
                        infer_memory["std"]
                    ),
                    "peak_updated_M": param["peak_updated_M"],
                    "final_extra_learned_M": param["final_extra_learned_M"],
                    "auxiliary_state_MiB": param["auxiliary_state_MiB"],
                    "extra_frozen_M": param["extra_frozen_M"],
                    "source": "Awesome-MMCL three complete runs",
                }
            )
    return rows


def sagla_rows(fieldnames):
    with SAGLA_CSV.open() as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for source in source_rows:
        inference_runs = [
            float(value)
            for value in source["inference_median_seconds_runs"].split("|")
        ]
        test_samples = {"Cars": 4083, "Aircraft": 3333}[source["dataset"]]
        throughput_runs = [test_samples / seconds for seconds in inference_runs]
        train_peak_runs = [float(source["train_peak_gib_mean"])] * int(source["n"])
        inference_peak_runs = [
            float(source["inference_peak_gib_mean"])
        ] * int(source["n"])
        row = {field: "" for field in fieldnames}
        row.update(
            method=source["method"],
            dataset=source["dataset"],
            n=source["n"],
            train_minutes_runs=source["train_minutes_runs"],
            train_minutes_mean=source["train_minutes_mean"],
            train_minutes_sample_variance=source[
                "train_minutes_sample_variance"
            ],
            train_minutes_sample_std=source["train_minutes_sample_std"],
            inference_seconds_runs=source["inference_median_seconds_runs"],
            inference_seconds_mean=source["inference_median_seconds_mean"],
            inference_seconds_sample_variance=source[
                "inference_median_seconds_sample_variance"
            ],
            inference_seconds_sample_std=source[
                "inference_median_seconds_sample_std"
            ],
            images_per_second_runs="|".join(
                "{:.6f}".format(value) for value in throughput_runs
            ),
            images_per_second_mean=source["images_per_second_mean"],
            images_per_second_sample_variance=source[
                "images_per_second_sample_variance"
            ],
            images_per_second_sample_std=source["images_per_second_sample_std"],
            train_peak_gib_runs="|".join(
                "{:.6f}".format(value) for value in train_peak_runs
            ),
            train_peak_gib_mean=source["train_peak_gib_mean"],
            train_peak_gib_sample_variance=source[
                "train_peak_gib_sample_variance"
            ],
            train_peak_gib_sample_std="{:.6f}".format(
                float(source["train_peak_gib_sample_variance"]) ** 0.5
            ),
            inference_peak_gib_runs="|".join(
                "{:.6f}".format(value) for value in inference_peak_runs
            ),
            inference_peak_gib_mean=source["inference_peak_gib_mean"],
            inference_peak_gib_sample_variance=source[
                "inference_peak_gib_sample_variance"
            ],
            inference_peak_gib_sample_std="{:.6f}".format(
                float(source["inference_peak_gib_sample_variance"]) ** 0.5
            ),
            peak_updated_M="0.000000",
            final_extra_learned_M="0.000000",
            auxiliary_state_MiB="3.032",
            extra_frozen_M="0.000000",
            source="SAGLA three complete runs",
        )
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def two_decimals(value):
    return "{:.2f}".format(float(value))


def write_paper_csv(path, rows):
    fieldnames = [
        "方法",
        "数据集",
        "训练时间均值(min)",
        "训练时间标准差(min)",
        "训练时间样本方差(min^2)",
        "推理时间均值(s)",
        "推理时间标准差(s)",
        "推理时间样本方差(s^2)",
        "训练峰值显存均值(GiB)",
        "训练峰值显存标准差(GiB)",
        "训练峰值显存样本方差(GiB^2)",
        "推理峰值显存均值(GiB)",
        "推理峰值显存标准差(GiB)",
        "推理峰值显存样本方差(GiB^2)",
        "峰值更新参数(M)",
        "最终额外学习参数(M)",
        "辅助状态(MiB)",
        "额外冻结参数(M)",
    ]
    paper_rows = []
    for row in rows:
        paper_rows.append(
            {
                "方法": row["method"],
                "数据集": row["dataset"],
                "训练时间均值(min)": two_decimals(row["train_minutes_mean"]),
                "训练时间标准差(min)": two_decimals(
                    row["train_minutes_sample_std"]
                ),
                "训练时间样本方差(min^2)": two_decimals(
                    row["train_minutes_sample_variance"]
                ),
                "推理时间均值(s)": two_decimals(row["inference_seconds_mean"]),
                "推理时间标准差(s)": two_decimals(
                    row["inference_seconds_sample_std"]
                ),
                "推理时间样本方差(s^2)": two_decimals(
                    row["inference_seconds_sample_variance"]
                ),
                "训练峰值显存均值(GiB)": two_decimals(
                    row["train_peak_gib_mean"]
                ),
                "训练峰值显存标准差(GiB)": two_decimals(
                    row["train_peak_gib_sample_std"]
                ),
                "训练峰值显存样本方差(GiB^2)": two_decimals(
                    row["train_peak_gib_sample_variance"]
                ),
                "推理峰值显存均值(GiB)": two_decimals(
                    row["inference_peak_gib_mean"]
                ),
                "推理峰值显存标准差(GiB)": two_decimals(
                    row["inference_peak_gib_sample_std"]
                ),
                "推理峰值显存样本方差(GiB^2)": two_decimals(
                    row["inference_peak_gib_sample_variance"]
                ),
                "峰值更新参数(M)": two_decimals(row["peak_updated_M"]),
                "最终额外学习参数(M)": two_decimals(
                    row["final_extra_learned_M"]
                ),
                "辅助状态(MiB)": two_decimals(row["auxiliary_state_MiB"]),
                "额外冻结参数(M)": two_decimals(row["extra_frozen_M"]),
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paper_rows)


def write_report(path, rows):
    lines = [
        "# 三次独立运行的效率与参数统计",
        "",
        "所有方法均在 Cars 与 Aircraft 上采用 B0-Inc10（10 个任务、每任务 "
        "10 类）、随机种子 1993、OpenCLIP ViT-B/16 LAION-400M 权重，并在 "
        "server 204 的 NVIDIA A800 80GB PCIe 上运行。每个统计样本都是一次 "
        "完整、CUDA 同步的十任务实验。每次运行的推理值为预热后 3 次完整 "
        "最终测试集推理的中位数，而不是把这 3 次内部重复视为独立实验。",
        "",
        "最终推理统一使用 batch size 64；训练保留各方法原始配置中的 batch "
        "size 与 epoch 日程。方差为三次独立运行的样本方差 "
        "`s^2 = sum((x_i - mean)^2) / (n - 1)`（`n=3`）。显存采用进程内 "
        "峰值已分配显存。下表所有展示结果统一保留两位小数。",
        "",
        "| 方法 / 数据集 | 训练时间 min（均值 ± 标准差） | 训练方差 min² | "
        "推理时间 s（均值 ± 标准差） | 推理方差 s² | 训练峰值显存 GiB" 
        "（均值 ± 标准差；方差） | 推理峰值显存 GiB（均值 ± 标准差；方差） | "
        "峰值更新参数 M | 最终额外学习参数 M | 辅助状态 MiB | 额外冻结参数 M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} / {dataset} | {train_mean} ± {train_std} | "
            "{train_var} | {infer_mean} ± {infer_std} | {infer_var} | "
            "{train_mem_mean} ± {train_mem_std}；{train_mem_var} | "
            "{infer_mem_mean} ± {infer_mem_std}；{infer_mem_var} | "
            "{peak_updated} | {final_extra} | {state} | {extra_frozen} |".format(
                method=row["method"],
                dataset=row["dataset"],
                train_mean=two_decimals(row["train_minutes_mean"]),
                train_std=two_decimals(row["train_minutes_sample_std"]),
                train_var=two_decimals(row["train_minutes_sample_variance"]),
                infer_mean=two_decimals(row["inference_seconds_mean"]),
                infer_std=two_decimals(row["inference_seconds_sample_std"]),
                infer_var=two_decimals(row["inference_seconds_sample_variance"]),
                train_mem_mean=two_decimals(row["train_peak_gib_mean"]),
                train_mem_std=two_decimals(row["train_peak_gib_sample_std"]),
                train_mem_var=two_decimals(
                    row["train_peak_gib_sample_variance"]
                ),
                infer_mem_mean=two_decimals(row["inference_peak_gib_mean"]),
                infer_mem_std=two_decimals(row["inference_peak_gib_sample_std"]),
                infer_mem_var=two_decimals(
                    row["inference_peak_gib_sample_variance"]
                ),
                peak_updated=two_decimals(row["peak_updated_M"]),
                final_extra=two_decimals(row["final_extra_learned_M"]),
                state=two_decimals(row["auxiliary_state_MiB"]),
                extra_frozen=two_decimals(row["extra_frozen_M"]),
            )
        )
    lines.extend(
        [
            "",
            "论文展示用的两位小数数据见 `combined_repeated_table_2dp.csv`。"
            "高精度逐次观测值及派生统计保留在 "
            "`combined_repeated_statistics.csv`，基线方法的高精度统计见 "
            "`baseline_repeated_statistics.csv`，用于复核而非直接排版。",
            "",
            "说明：ZS-CLIP 不执行优化训练，其训练时间实际为初始化与特征准备 "
            "时间。ENGINE 的额外冻结参数包含一个未使用、可裁剪的实现副本。"
            "BOFA 的最终额外参数为 0，前提是采用已验证的融合投影导出并裁剪 "
            "推理不使用的任务头。SAGLA-C 为 Top-10 consensus 配置。",
            "本次实验日志统一归档在 `logs/efficiency/`。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    baseline = baseline_rows()
    fieldnames = list(baseline[0])
    sagla = sagla_rows(fieldnames)
    combined = baseline + sagla
    write_csv(OUT / "baseline_repeated_statistics.csv", baseline)
    write_csv(OUT / "combined_repeated_statistics.csv", combined)
    write_paper_csv(OUT / "combined_repeated_table_2dp.csv", combined)
    write_report(OUT / "COMBINED_REPEATED_REPORT.md", combined)
    print("Validated and summarized {} baseline + {} SAGLA rows".format(
        len(baseline), len(sagla)
    ))


if __name__ == "__main__":
    main()
