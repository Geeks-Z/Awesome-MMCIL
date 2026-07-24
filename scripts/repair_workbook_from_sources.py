#!/usr/bin/env python3
"""Rebuild MMCL_Baselines.xlsx result cells from completed source artifacts.

This script deliberately uses fixed *method positions per sheet* (not the
previous AREA writer's fixed absolute rows).  A result is written only after
its final task is present in the corresponding log/metrics file; incomplete
or missing sources leave the spreadsheet cell blank.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "MMCL_Baselines.xlsx"
BACKUP = ROOT / "MMCL_Baselines.before_log_repair.xlsx"
SEED_SHEETS = {0: "Seed0", 42: "Seed42", 1993: "Seed1993", 2026: "Seed2026"}

ROOT_METHODS = ("SimpleCIL", "ZS-CLIP", "RAPF", "ENGINE", "CLG-CBM", "BOFA", "AREA")
LOG_DIR_CANDIDATES = {
    "ENGINE": ("ENGINE", "engine"),
    "AREA": ("AREA", "area", "aera"),
}
METHOD_VENUES = {
    "SimpleCIL": None,
    "ZS-CLIP": None,
    "RAPF": "ECCV 2024",
    "ENGINE": "ICCV 2025",
    "CLG-CBM": "CVPR 2025",
    "BOFA": "AAAI 2026",
    "PromptFusion": "ECCV 2024",
    "MoE-Adapters": "CVPR 2024",
    "AREA": "ICML 2026",
}

BLOCKS = {
    "top": (
        ("aircraft_clip_0_10", "aircraft_clip_50_10", "cifar224_clip_0_10", "cifar224_clip_50_10", "cars_clip_0_10", "cars_clip_50_10"),
        ("SimpleCIL", "ZS-CLIP", "RAPF", "ENGINE", "CLG-CBM", "BOFA", "PromptFusion", "MoE-Adapters", "AREA"),
    ),
    "middle": (
        ("imagenetr_clip_0_20", "imagenetr_clip_100_20", "cub_clip_0_20", "cub_clip_100_20", "ucf101_clip_0_10", "ucf101_clip_50_10"),
        ("SimpleCIL", "ZS-CLIP", "RAPF", "ENGINE", "CLG-CBM", "BOFA", "PromptFusion", "MoE-Adapters", "AREA"),
    ),
    "bottom": (
        ("sun_clip_0_30", "sun_clip_150_30", "food101_clip_0_10", "food101_clip_50_10", "objectnet_clip_0_20", "objectnet_clip_100_20"),
        ROOT_METHODS,
    ),
}

BLOCK_ORDER = ("top", "middle", "bottom")


def root_metrics(path: Path) -> tuple[float, float] | None:
    """Return (A_bar, A_B) only for a completed root-framework run."""
    if not path.is_file():
        return None
    content = path.read_text(errors="replace").rstrip()
    averages = re.findall(r"Average Accuracy \(CNN top1\): ([0-9.]+)", content)
    curves = re.findall(r"CNN top1 curve: \[([^]]+)\]", content)
    if not averages or not curves or not content.splitlines()[-1].endswith(averages[-1]):
        return None
    curve = [float(value.strip()) for value in curves[-1].split(",")]
    return round(float(averages[-1]), 2), round(curve[-1], 2)


def root_log_path(method: str, stem: str, seed: int) -> Path:
    """Prefer canonical log folders while retaining read-only legacy fallback."""
    directories = LOG_DIR_CANDIDATES.get(method, (method,))
    for directory in directories:
        path = ROOT / "logs" / directory / f"{stem}_{seed}.log"
        if path.is_file():
            return path
    return ROOT / "logs" / directories[0] / f"{stem}_{seed}.log"


def promptfusion_metrics(path: Path) -> tuple[float, float] | None:
    """Return final PromptFusion metrics from its terminal summary."""
    if not path.is_file():
        return None
    content = path.read_text(errors="replace")
    averages = re.findall(r"Average Accuracy \((?:CNN top1|Top1)\): ([0-9.]+)", content)
    finals = re.findall(r"Last Accuracy: ([0-9.]+)", content)
    if not averages or not finals:
        return None
    return round(float(averages[-1]), 2), round(float(finals[-1]), 2)


def moe_metrics(path: Path) -> tuple[float, float] | None:
    """Read the final JSON-lines record written by MoE-Adapters."""
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        record = json.loads(lines[-1])
        return round(float(record["avg"]), 2), round(float(record["last"]), 2)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def set_value(ws, row: int, column: int, value, changes: list[str]) -> None:
    cell = ws.cell(row=row, column=column)
    if cell.value != value:
        changes.append(f"{ws.title}!{cell.coordinate}: {cell.value!r} -> {value!r}")
        cell.value = value


def write_result_row(ws, row: int, values: dict[int, tuple[float, float]], changes: list[str]) -> None:
    """Set one six-protocol row directly to its desired final state."""
    for protocol_index in range(6):
        metrics = values.get(protocol_index)
        column = 3 + protocol_index * 2
        set_value(ws, row, column, None if metrics is None else metrics[0], changes)
        set_value(ws, row, column + 1, None if metrics is None else metrics[1], changes)


def method_rows_by_block(ws) -> dict[str, dict[str, int]]:
    """Locate result rows from the live Method column, never from row numbers.

    The workbook may gain, remove, or reorder methods.  Each result block is
    delimited by its own ``Method`` header, so we first find those three
    headers and then map the actual method names below each header.
    """
    headers = [row for row in range(1, ws.max_row + 1) if ws.cell(row=row, column=1).value == "Method"]
    if len(headers) != 3:
        raise ValueError(f"{ws.title}: expected exactly three Method headers, found {len(headers)}")
    result: dict[str, dict[str, int]] = {}
    for block, header, next_header in zip(BLOCK_ORDER, headers, (*headers[1:], ws.max_row + 1)):
        rows: dict[str, int] = {}
        for row in range(header + 2, next_header):
            method = ws.cell(row=row, column=1).value
            if isinstance(method, str) and method.strip():
                if method in rows:
                    raise ValueError(f"{ws.title}: duplicate method {method!r} in {block} block")
                rows[method] = row
        result[block] = rows
    return result


def update_root_method(ws, sheet_name: str, seed: int, block: str, method: str, row: int, changes: list[str], missing: list[str]) -> None:
    stems, _ = BLOCKS[block]
    set_value(ws, row, 1, method, changes)
    set_value(ws, row, 2, METHOD_VENUES[method], changes)
    values: dict[int, tuple[float, float]] = {}
    for index, stem in enumerate(stems):
        metrics = root_metrics(root_log_path(method, stem, seed))
        if metrics is None:
            missing.append(f"{sheet_name}:{method}:{stem}:seed={seed}")
            continue
        values[index] = metrics
    write_result_row(ws, row, values, changes)


def update_promptfusion(ws, seed: int, block: str, row: int, changes: list[str], missing: list[str]) -> None:
    set_value(ws, row, 1, "PromptFusion", changes)
    set_value(ws, row, 2, METHOD_VENUES["PromptFusion"], changes)
    if block == "top":
        stem, index = "cifar224_vit_b16_0_10", 2
    else:
        stem, index = "imagenetr_vit_b16_0_20", 0
    metrics = promptfusion_metrics(ROOT / "logs" / "PromptFusion" / f"{stem}_{seed}.log")
    if metrics is None:
        missing.append(f"{ws.title}:PromptFusion:{stem}:seed={seed}")
        write_result_row(ws, row, {}, changes)
        return
    write_result_row(ws, row, {index: metrics}, changes)


def update_moe(ws, seed: int, block: str, row: int, changes: list[str], missing: list[str]) -> None:
    set_value(ws, row, 1, "MoE-Adapters", changes)
    set_value(ws, row, 2, METHOD_VENUES["MoE-Adapters"], changes)
    if block == "top":
        experiment, index = "cifar100_50_10", 3
    else:
        experiment, index = "imagenet_r_100_20", 1
    source = ROOT / "MoE-Adapters4CL-MoE-Adapters" / "cil" / "experiments" / "class" / experiment / f"seed_{seed}" / "metrics.json"
    metrics = moe_metrics(source)
    if metrics is None:
        missing.append(f"{ws.title}:MoE-Adapters:{experiment}:seed={seed}")
        write_result_row(ws, row, {}, changes)
        return
    write_result_row(ws, row, {index: metrics}, changes)


def repair(dry_run: bool) -> tuple[list[str], list[str]]:
    workbook = load_workbook(WORKBOOK)
    changes: list[str] = []
    missing: list[str] = []
    for seed, sheet_name in SEED_SHEETS.items():
        ws = workbook[sheet_name]
        rows_by_block = method_rows_by_block(ws)
        for block, (_, methods) in BLOCKS.items():
            for method in methods:
                row = rows_by_block[block].get(method)
                if row is None:
                    missing.append(f"{sheet_name}:{block}:{method}:method row not found")
                    continue
                if method in ROOT_METHODS:
                    update_root_method(ws, sheet_name, seed, block, method, row, changes, missing)
                elif method == "PromptFusion":
                    update_promptfusion(ws, seed, block, row, changes, missing)
                elif method == "MoE-Adapters":
                    update_moe(ws, seed, block, row, changes, missing)
    if not dry_run:
        if not BACKUP.exists():
            shutil.copy2(WORKBOOK, BACKUP)
        workbook.save(WORKBOOK)
    return changes, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing the workbook")
    args = parser.parse_args()
    changes, missing = repair(args.dry_run)
    print(f"{'Would change' if args.dry_run else 'Changed'} {len(changes)} cells")
    for change in changes:
        print(change)
    print(f"Incomplete or missing sources: {len(missing)}")
    for source in missing:
        print(source)
    if not args.dry_run:
        print(f"Backup: {BACKUP}")
        print(f"Updated: {WORKBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
