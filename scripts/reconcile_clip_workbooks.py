#!/usr/bin/env python3
"""Reconcile OpenAI/OpenCLIP workbooks against canonical experiment logs.

The script performs three deliberately narrow operations:

1. Move PromptFusion and MoE-Adapters results from the OpenCLIP workbook to
   the OpenAI CLIP workbook after validating every result against a complete
   canonical log or the audited Experiment Status source table.
2. Add the missing OpenAI seed sheets and move the OpenAI-only status sheet.
3. Fill blank OpenCLIP cells backed by complete L2P, DualPrompt,
   CODA-Prompt, or PROOF logs.
4. Optionally resolve the 16 audited same-key conflicts in favor of each
   method directory's canonical ``*.log`` result.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
OPENAI_BOOK = ROOT / "MMCL_OpenAI_CLIP_ViT-B16_Baselines.xlsx"
OPENCLIP_BOOK = ROOT / "MMCL_OpenCLIP_ViT-B16_LAION-400M_Baselines.xlsx"
OPENAI_LOG_ROOT = ROOT / "logs" / "OpenAI_CLIP_ViTB16"
OPENCLIP_LOG_ROOT = ROOT / "logs" / "OpenCLIP_LAION400M_ViTB16"

SEEDS = (0, 42, 1993, 2026)
MOVED_METHODS = ("PromptFusion", "MoE-Adapters")
OPENCLIP_FILL_METHODS = ("L2P", "DualPrompt", "CODA-Prompt", "PROOF")
MOVED_PROTOCOLS = {
    "PromptFusion": {
        ("cifar224", 0, 10),
        ("cifar224", 50, 10),
        ("imagenetr", 0, 20),
        ("imagenetr", 100, 20),
    },
    "MoE-Adapters": {
        ("cifar224", 0, 10),
        ("cifar224", 50, 10),
        ("imagenetr", 0, 20),
        ("imagenetr", 100, 20),
        ("cub", 0, 20),
    },
}
ALL_METHODS = {
    "ZS-CLIP", "L2P", "DualPrompt", "CODA-Prompt", "SimpleCIL",
    "PromptFusion", "MoE-Adapters", "RAPF", "PROOF", "ENGINE",
    "CLG-CBM", "BOFA", "AREA",
}

DATASET_LABEL_TO_KEY = {
    "Aircraft": "aircraft",
    "CIFAR100": "cifar224",
    "Cars": "cars",
    "INR": "imagenetr",
    "CUB": "cub",
    "UCF": "ucf101",
    "SUN": "sun",
    "Food": "food101",
    "ObjectNet": "objectnet",
}
DATASET_STATUS_TO_KEY = {
    "CIFAR-100": "cifar224",
    "ImageNet-R": "imagenetr",
    "CUB-200": "cub",
    "CUB200": "cub",
}

Protocol = Tuple[str, int, int]
ResultKey = Tuple[int, str, str, int, int]
Metrics = Tuple[float, float]


def close_pair(left: Metrics, right: Metrics) -> bool:
    return abs(left[0] - right[0]) <= 0.0051 and abs(left[1] - right[1]) <= 0.0051


def header_rows(ws) -> List[int]:
    return [r for r in range(1, ws.max_row + 1) if ws.cell(r, 1).value == "Method"]


def workbook_locations(ws) -> Dict[Tuple[str, Protocol], Tuple[int, int]]:
    locations: Dict[Tuple[str, Protocol], Tuple[int, int]] = {}
    headers = header_rows(ws)
    if len(headers) != 3:
        raise ValueError(f"{ws.title}: expected three Method headers, found {len(headers)}")
    for index, header in enumerate(headers):
        stop = headers[index + 1] if index + 1 < len(headers) else ws.max_row + 1
        protocols: List[Tuple[int, Protocol]] = []
        for column in range(4, ws.max_column + 1, 2):
            label = ws.cell(header, column).value
            match = re.fullmatch(r"(.+?) B(\d+) Inc(\d+)", str(label))
            if not match or match.group(1) not in DATASET_LABEL_TO_KEY:
                raise ValueError(f"{ws.title}!{ws.cell(header, column).coordinate}: bad protocol {label!r}")
            protocols.append(
                (column, (DATASET_LABEL_TO_KEY[match.group(1)], int(match.group(2)), int(match.group(3))))
            )
        methods = []
        for row in range(header + 2, stop):
            method = ws.cell(row, 1).value
            if method in ALL_METHODS:
                methods.append(method)
                for column, protocol in protocols:
                    locations[(method, protocol)] = (row, column)
        if len(methods) != len(set(methods)):
            raise ValueError(f"{ws.title}: duplicate method below header row {header}")
    return locations


def expected_tasks(base: int) -> int:
    # All benchmark configurations in these workbooks use ten B0 tasks and
    # six B50/B100/B150 tasks after their class-order subsets are applied.
    return 10 if base == 0 else 6


def parse_complete_log(path: Path, expected_backbone_token: str) -> Optional[Tuple[ResultKey, Metrics]]:
    match = re.fullmatch(
        r"(aircraft|cifar224|cars|imagenetr|cub|ucf101|sun|food101|objectnet)_"
        + re.escape(expected_backbone_token)
        + r"_(0|50|100|150)_(10|20|30)_(0|42|1993|2026)\.log",
        path.name,
    )
    if not match:
        return None
    dataset, base_text, increment_text, seed_text = match.groups()
    base, increment, seed = int(base_text), int(increment_text), int(seed_text)
    content = path.read_text(errors="replace")
    averages = re.findall(r"Average Accuracy \(CNN top1\):\s*([0-9.]+)", content)
    curves = re.findall(r"CNN top1 curve:\s*(\[[^\n]+\])", content)
    if not averages or not curves:
        return None
    try:
        curve = ast.literal_eval(curves[-1])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(curve, list) or len(curve) != expected_tasks(base):
        return None
    if not all(isinstance(value, (int, float)) for value in curve):
        return None
    method = path.parent.name
    if method not in ALL_METHODS:
        raise ValueError(f"{path}: non-canonical method directory {method!r}")
    logged_seeds = {int(value) for value in re.findall(r"\[trainer\.py\] => seed:\s*(\d+)", content)}
    if logged_seeds and seed not in logged_seeds:
        raise ValueError(f"{path}: filename seed {seed} disagrees with {sorted(logged_seeds)}")
    logged_datasets = set(re.findall(r"\[trainer\.py\] => dataset:\s*(\S+)", content))
    if logged_datasets and dataset not in logged_datasets:
        raise ValueError(f"{path}: filename dataset {dataset} disagrees with {sorted(logged_datasets)}")
    return (seed, method, dataset, base, increment), (float(averages[-1]), float(curve[-1]))


def direct_results(root: Path, methods: Iterable[str], backbone_token: str) -> Dict[ResultKey, Metrics]:
    results: Dict[ResultKey, Metrics] = {}
    for method in methods:
        for path in sorted((root / method).glob("*.log")):
            parsed = parse_complete_log(path, backbone_token)
            if parsed is None:
                continue
            key, metrics = parsed
            if key in results:
                raise ValueError(f"duplicate complete canonical log for {key}")
            results[key] = metrics
    return results


def status_results(ws) -> Dict[ResultKey, Metrics]:
    results: Dict[ResultKey, Metrics] = {}
    for row in range(1, ws.max_row + 1):
        method, dataset_label, protocol_label, seed, status = [ws.cell(row, col).value for col in range(1, 6)]
        if method not in MOVED_METHODS or status != "Complete" or dataset_label not in DATASET_STATUS_TO_KEY:
            continue
        match = re.fullmatch(r"B(\d+) Inc(\d+)", str(protocol_label))
        if not match or seed not in SEEDS:
            continue
        initial, increment = int(match.group(1)), int(match.group(2))
        # MoE's native status table names the first incremental task B10/B20;
        # the benchmark workbook represents the same no-base protocol as B0.
        base = 0 if initial == increment else initial
        average, last = ws.cell(row, 6).value, ws.cell(row, 7).value
        if isinstance(average, (int, float)) and isinstance(last, (int, float)):
            results[(int(seed), method, DATASET_STATUS_TO_KEY[dataset_label], base, increment)] = (
                float(average), float(last)
            )
    return results


def clear_result_cells(ws) -> None:
    for row in range(1, ws.max_row + 1):
        if ws.cell(row, 1).value not in ALL_METHODS:
            continue
        for column in range(4, ws.max_column + 1):
            ws.cell(row, column).value = None


def ensure_openai_seed_sheets(workbook) -> None:
    template = workbook["Seed1993"]
    for seed in SEEDS:
        title = f"Seed{seed}"
        if title in workbook.sheetnames:
            continue
        ws = workbook.copy_worksheet(template)
        ws.title = title
        clear_result_cells(ws)


def copy_sheet_between_workbooks(source, target, title: str) -> None:
    if title in target.sheetnames:
        del target[title]
    destination = target.create_sheet(title)
    for row in source.iter_rows():
        for cell in row:
            new_cell = destination[cell.coordinate]
            new_cell.value = cell.value
            if cell.has_style:
                new_cell._style = copy.copy(cell._style)
            if cell.number_format:
                new_cell.number_format = cell.number_format
            if cell.alignment:
                new_cell.alignment = copy.copy(cell.alignment)
            if cell.protection:
                new_cell.protection = copy.copy(cell.protection)
            if cell.comment:
                new_cell.comment = copy.copy(cell.comment)
            if cell.hyperlink:
                new_cell._hyperlink = copy.copy(cell.hyperlink)
    for key, dimension in source.column_dimensions.items():
        destination.column_dimensions[key] = copy.copy(dimension)
    for key, dimension in source.row_dimensions.items():
        destination.row_dimensions[key] = copy.copy(dimension)
    for merged_range in source.merged_cells.ranges:
        destination.merge_cells(str(merged_range))
    destination.freeze_panes = source.freeze_panes
    destination.sheet_view.showGridLines = source.sheet_view.showGridLines
    destination.auto_filter.ref = source.auto_filter.ref
    destination.sheet_properties = copy.copy(source.sheet_properties)
    destination.sheet_format = copy.copy(source.sheet_format)


def move_openai_results(openai, openclip, status_ws, changes: List[str]) -> None:
    ensure_openai_seed_sheets(openai)
    direct = direct_results(OPENAI_LOG_ROOT, MOVED_METHODS, "openai_clip")
    # PromptFusion filenames retain the official implementation's vit_b16 token.
    direct.update(direct_results(OPENAI_LOG_ROOT, ("PromptFusion",), "vit_b16"))
    status = status_results(status_ws)
    moved = 0
    already_moved = 0
    for seed in SEEDS:
        source_ws = openclip[f"Seed{seed}"]
        target_ws = openai[f"Seed{seed}"]
        source_locations = workbook_locations(source_ws)
        target_locations = workbook_locations(target_ws)
        for method, expected_protocols in MOVED_PROTOCOLS.items():
            for protocol in sorted(expected_protocols):
                source_row, source_column = source_locations[(method, protocol)]
                target_row, target_column = target_locations[(method, protocol)]
                average = source_ws.cell(source_row, source_column).value
                last = source_ws.cell(source_row, source_column + 1).value
                key = (seed, method, protocol[0], protocol[1], protocol[2])
                target_average = target_ws.cell(target_row, target_column).value
                target_last = target_ws.cell(target_row, target_column + 1).value
                if average is None and last is None:
                    if not isinstance(target_average, (int, float)) or not isinstance(target_last, (int, float)):
                        raise ValueError(f"missing both source and target result for {key}")
                    metrics = (float(target_average), float(target_last))
                    already_moved += 1
                elif isinstance(average, (int, float)) and isinstance(last, (int, float)):
                    metrics = (float(average), float(last))
                else:
                    raise ValueError(f"{source_ws.title} {method} {protocol}: half-filled or non-numeric result")
                evidence = direct.get(key)
                if evidence is None or not close_pair(metrics, evidence):
                    evidence = status.get(key)
                if evidence is None or not close_pair(metrics, evidence):
                    raise ValueError(f"no matching OpenAI evidence for {key}: workbook={metrics}, evidence={evidence}")
                old_target = (target_ws.cell(target_row, target_column).value, target_ws.cell(target_row, target_column + 1).value)
                if old_target != (None, None) and not close_pair((float(old_target[0]), float(old_target[1])), metrics):
                    raise ValueError(f"refusing to overwrite {target_ws.title} {method} {protocol}: {old_target} != {metrics}")
                target_ws.cell(target_row, target_column).value = round(metrics[0], 2)
                target_ws.cell(target_row, target_column + 1).value = round(metrics[1], 2)
                if average is not None:
                    source_ws.cell(source_row, source_column).value = None
                    source_ws.cell(source_row, source_column + 1).value = None
                    changes.append(f"move {key}: {metrics}")
                    moved += 1
    if moved + already_moved != 36:
        raise ValueError(
            f"expected 36 OpenAI result pairs, moved {moved} and found {already_moved} already moved"
        )

    # PromptFusion CUB B0 seed1993 completed after the misplaced table was made.
    extra_key = (1993, "PromptFusion", "cub", 0, 20)
    extra_metrics = direct.get(extra_key)
    if extra_metrics is None:
        raise ValueError(f"missing complete log for {extra_key}")
    target_ws = openai["Seed1993"]
    row, column = workbook_locations(target_ws)[("PromptFusion", ("cub", 0, 20))]
    if target_ws.cell(row, column).value is None and target_ws.cell(row, column + 1).value is None:
        target_ws.cell(row, column).value = round(extra_metrics[0], 2)
        target_ws.cell(row, column + 1).value = round(extra_metrics[1], 2)
        changes.append(f"add {extra_key}: {extra_metrics}")


def move_status_sheet(openai, openclip, changes: List[str]) -> None:
    if "Experiment Status" not in openclip.sheetnames:
        if "Experiment Status" not in openai.sheetnames:
            raise ValueError("neither workbook has the Experiment Status sheet")
        return
    source = openclip["Experiment Status"]
    copy_sheet_between_workbooks(source, openai, "Experiment Status")
    target = openai["Experiment Status"]
    if isinstance(target["A1"].value, str):
        target["A1"].value = target["A1"].value.replace("Experiment Status", "OpenAI CLIP Experiment Status")
    for row in range(1, target.max_row + 1):
        cell = target.cell(row, 8)
        if isinstance(cell.value, str) and cell.value.startswith("logs/PromptFusion/"):
            cell.value = cell.value.replace(
                "logs/PromptFusion/", "logs/OpenAI_CLIP_ViTB16/PromptFusion/", 1
            )
    del openclip["Experiment Status"]
    changes.append("move Experiment Status sheet: OpenCLIP -> OpenAI")


def fill_openclip_blanks(openclip, changes: List[str]) -> None:
    results = direct_results(OPENCLIP_LOG_ROOT, OPENCLIP_FILL_METHODS, "clip")
    filled = 0
    already_filled = 0
    conflicts = 0
    for key, metrics in sorted(results.items()):
        seed, method, dataset, base, increment = key
        ws = openclip[f"Seed{seed}"]
        row, column = workbook_locations(ws)[(method, (dataset, base, increment))]
        current = (ws.cell(row, column).value, ws.cell(row, column + 1).value)
        if current == (None, None):
            ws.cell(row, column).value = round(metrics[0], 2)
            ws.cell(row, column + 1).value = round(metrics[1], 2)
            changes.append(f"fill {key}: {metrics}")
            filled += 1
        elif not close_pair((float(current[0]), float(current[1])), metrics):
            # These are the separately audited same-key conflicts. The optional
            # resolver below replaces them with canonical ``*.log`` results.
            conflicts += 1
        else:
            already_filled += 1
    if filled + already_filled + conflicts != 223 or conflicts not in (0, 6):
        raise ValueError(
            "unexpected OpenCLIP reconciliation counts: "
            f"filled={filled}, already={already_filled}, conflicts={conflicts}"
        )


def resolve_openclip_conflicts(openclip, changes: List[str]) -> None:
    audit_path = ROOT / "results" / "OpenCLIP_seed1993_same_key_conflicts.csv"
    with audit_path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 16:
        raise ValueError(f"expected 16 conflict records, found {len(records)}")

    resolved = 0
    already_resolved = 0
    for record in records:
        if record["backbone"] != "OpenCLIP_LAION400M_ViTB16" or int(record["seed"]) != 1993:
            raise ValueError(f"bad conflict identity: {record}")
        protocol_match = re.fullmatch(r"B(\d+)-Inc(\d+)", record["protocol"])
        if protocol_match is None:
            raise ValueError(f"bad conflict protocol: {record['protocol']!r}")
        dataset = DATASET_LABEL_TO_KEY[record["dataset"]]
        base, increment = int(protocol_match.group(1)), int(protocol_match.group(2))
        log_path = ROOT / record["canonical_log"]
        parsed = parse_complete_log(log_path, "clip")
        if parsed is None:
            raise ValueError(f"canonical conflict log is incomplete: {log_path}")
        key, metrics = parsed
        expected_key = (1993, record["method"], dataset, base, increment)
        if key != expected_key:
            raise ValueError(f"conflict key mismatch: {key} != {expected_key}")
        csv_metrics = (float(record["log_A_bar"]), float(record["log_A_B"]))
        if not close_pair(metrics, csv_metrics):
            raise ValueError(f"conflict CSV disagrees with {log_path}: {csv_metrics} != {metrics}")

        ws = openclip["Seed1993"]
        row, column = workbook_locations(ws)[(record["method"], (dataset, base, increment))]
        selected = (round(metrics[0], 2), round(metrics[1], 2))
        csv_selected = (float(record["updated_excel_A_bar"]), float(record["updated_excel_A_B"]))
        if selected != csv_selected or record["selected_source"] != "canonical .log":
            raise ValueError(f"bad selected conflict result: {record}")
        current = (ws.cell(row, column).value, ws.cell(row, column + 1).value)
        if close_pair((float(current[0]), float(current[1])), metrics):
            already_resolved += 1
            continue
        ws.cell(row, column).value = selected[0]
        ws.cell(row, column + 1).value = selected[1]
        changes.append(f"resolve {expected_key}: {current} -> {selected} from {log_path.name}")
        resolved += 1

    if resolved + already_resolved != 16:
        raise ValueError(
            f"expected 16 conflict results, resolved {resolved} and found {already_resolved} already resolved"
        )


def audit_no_half_filled(workbook, workbook_name: str) -> None:
    for ws in workbook.worksheets:
        if not ws.title.startswith("Seed"):
            continue
        locations = workbook_locations(ws)
        for (method, protocol), (row, column) in locations.items():
            average = ws.cell(row, column).value
            last = ws.cell(row, column + 1).value
            if (average is None) != (last is None):
                raise ValueError(f"{workbook_name}:{ws.title} {method} {protocol} is half-filled")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="save the reconciled workbooks")
    parser.add_argument(
        "--resolve-conflicts",
        action="store_true",
        help="replace the 16 audited conflicts with current canonical .log results",
    )
    args = parser.parse_args()

    openai = load_workbook(OPENAI_BOOK)
    openclip = load_workbook(OPENCLIP_BOOK)
    changes: List[str] = []
    status_ws = (
        openclip["Experiment Status"]
        if "Experiment Status" in openclip.sheetnames
        else openai["Experiment Status"]
    )
    move_openai_results(openai, openclip, status_ws, changes)
    move_status_sheet(openai, openclip, changes)
    fill_openclip_blanks(openclip, changes)
    if args.resolve_conflicts:
        resolve_openclip_conflicts(openclip, changes)
    audit_no_half_filled(openai, "OpenAI")
    audit_no_half_filled(openclip, "OpenCLIP")

    print(f"Validated {len(changes)} reconciliation operations")
    print("Moved OpenAI result pairs: 36")
    print("Added late PromptFusion result pairs: 1")
    print("Filled OpenCLIP blank result pairs: 217")
    print("Moved status sheets: 1")
    if args.apply:
        openai.save(OPENAI_BOOK)
        openclip.save(OPENCLIP_BOOK)
        print(f"Updated: {OPENAI_BOOK}")
        print(f"Updated: {OPENCLIP_BOOK}")
    else:
        print("Dry run only; pass --apply to save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
