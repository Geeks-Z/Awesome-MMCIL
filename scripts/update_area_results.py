#!/usr/bin/env python3
"""Extract completed AREA runs from logs and update MMCL_Baselines.xlsx."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "MMCL_Baselines.xlsx"
LOG_DIR = ROOT / "logs" / "area"

# (log stem, spreadsheet column for A_bar, spreadsheet column for A_B, row)
RESULT_CELLS = (
    ("aircraft_clip_0_10", "C", "D", 11),
    ("aircraft_clip_50_10", "E", "F", 11),
    ("cifar224_clip_0_10", "G", "H", 11),
    ("cifar224_clip_50_10", "I", "J", 11),
    ("cars_clip_0_10", "K", "L", 11),
    ("cars_clip_50_10", "M", "N", 11),
    ("imagenetr_clip_0_20", "C", "D", 23),
    ("imagenetr_clip_100_20", "E", "F", 23),
    ("cub_clip_0_20", "G", "H", 23),
    ("cub_clip_100_20", "I", "J", 23),
    ("ucf101_clip_0_10", "K", "L", 23),
    ("ucf101_clip_50_10", "M", "N", 23),
    ("sun_clip_0_30", "C", "D", 32),
    ("sun_clip_150_30", "E", "F", 32),
    ("food101_clip_0_10", "G", "H", 32),
    ("food101_clip_50_10", "I", "J", 32),
    ("objectnet_clip_0_20", "K", "L", 32),
    ("objectnet_clip_100_20", "M", "N", 32),
)

SEED_SHEETS = {1993: "Seed1993", 2026: "Seed2026", 0: "Seed0", 42: "Seed42"}


def log_stem(config: Path) -> str:
    args = json.loads(config.read_text())
    initial = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    return f"{args['dataset']}_{args['backbone_type']}_{initial}_{args['increment']}"


def completed_metrics(log_path: Path) -> tuple[float, float] | None:
    """Return (A_bar, A_B) only when the final task has completed."""
    if not log_path.is_file():
        return None

    content = log_path.read_text(errors="replace").rstrip()
    if not content:
        return None

    averages = re.findall(r"Average Accuracy \(CNN top1\): ([0-9.]+)", content)
    curves = re.findall(r"CNN top1 curve: \[([^]]+)\]", content)
    if not averages or not curves or not content.splitlines()[-1].endswith(averages[-1]):
        return None

    curve = [float(value.strip()) for value in curves[-1].split(",")]
    return round(float(averages[-1]), 2), round(curve[-1], 2)


def is_complete(config: Path, seed: int) -> bool:
    return completed_metrics(LOG_DIR / f"{log_stem(config)}_{seed}.log") is not None


def update_workbook() -> list[str]:
    workbook = load_workbook(WORKBOOK)
    updates: list[str] = []

    for seed, sheet_name in SEED_SHEETS.items():
        sheet = workbook[sheet_name]
        for _, _, _, row in RESULT_CELLS:
            sheet.cell(row=row, column=1).value = "AREA"

        for stem, a_bar_column, a_b_column, row in RESULT_CELLS:
            metrics = completed_metrics(LOG_DIR / f"{stem}_{seed}.log")
            if metrics is None:
                continue
            a_bar, a_b = metrics
            a_bar_cell = f"{a_bar_column}{row}"
            a_b_cell = f"{a_b_column}{row}"
            if sheet[a_bar_cell].value != a_bar or sheet[a_b_cell].value != a_b:
                sheet[a_bar_cell] = a_bar
                sheet[a_b_cell] = a_b
                updates.append(f"{sheet_name}!{a_bar_cell}:{a_bar}, {a_b_cell}:{a_b}")

    workbook.save(WORKBOOK)
    return updates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is-complete",
        nargs=2,
        metavar=("CONFIG", "SEED"),
        help="return success when this config/seed has a completed AREA log",
    )
    args = parser.parse_args()

    if args.is_complete:
        config, seed = args.is_complete
        return 0 if is_complete(ROOT / config, int(seed)) else 1

    updates = update_workbook()
    print("AREA workbook update complete")
    for update in updates:
        print(update)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
