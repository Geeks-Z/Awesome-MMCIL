#!/usr/bin/env python3
"""Migrate legacy logs into ``logs/<model>/<paper method>`` directories.

The migration is conservative: when two different runs have the same filename,
the run with the newest timestamp remains at the canonical path and the other
one is retained below that method's ``archive/`` directory.
"""

import argparse
import filecmp
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.log_layout import backbone_log_root, canonical_method_name


OPENAI_ROOT = "OpenAI_CLIP_ViTB16"
OPENCLIP_ROOT = "OpenCLIP_LAION400M_ViTB16"
LEGACY_METHOD_DIRS = {
    "CLG-CBM": "clg-cbm",
    "ENGINE": "engine",
    "bofa": "bofa",
    "coda": "coda",
    "dualprompt": "dualprompt",
    "l2p": "l2p",
    "proof": "proof",
    "rapf": "rapf",
    "simplecil": "simplecil",
    "zs_clip": "zs_clip",
}
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
MODEL_NAME_RE = re.compile(r"=> model_name:\s*([^\s]+)")
BACKBONE_NAME_RE = re.compile(r"=> backbone_type:\s*([^\s]+)")
QUEUE_METHOD_MARKERS = {
    "moe-adapters": "MoE-Adapters",
    "promptfusion": "PromptFusion",
    "coda-prompt": "CODA-Prompt",
    "coda_prompt": "CODA-Prompt",
    "dualprompt": "DualPrompt",
    "finetune": "FineTune",
    "simplecil": "SimpleCIL",
    "zs-clip": "ZS-CLIP",
    "zs_clip": "ZS-CLIP",
    "clg-cbm": "CLG-CBM",
    "clg_cbm": "CLG-CBM",
    "engine": "ENGINE",
    "bofa": "BOFA",
    "l2p": "L2P",
    "proof": "PROOF",
    "rapf": "RAPF",
    "area": "AREA",
}


def _last_timestamp(path):
    latest = None
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = TIMESTAMP_RE.match(line)
            if match:
                latest = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    return latest or datetime.min


def _archive_path(path, timestamp):
    stamp = timestamp.strftime("%Y%m%d_%H%M%S") if timestamp != datetime.min else "undated"
    archive_dir = path.parent / "archive"
    candidate = archive_dir / f"{path.stem}__{stamp}{path.suffix}"
    index = 2
    while candidate.exists():
        candidate = archive_dir / f"{path.stem}__{stamp}_{index}{path.suffix}"
        index += 1
    return candidate


def _move(source, destination, apply):
    print(f"MOVE {source} -> {destination}")
    if apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def _merge_log(source, destination, apply):
    if not destination.exists():
        _move(source, destination, apply)
        return

    if filecmp.cmp(source, destination, shallow=False):
        print(f"DROP identical duplicate {source}")
        if apply:
            source.unlink()
        return

    source_time = _last_timestamp(source)
    destination_time = _last_timestamp(destination)
    if source_time >= destination_time:
        archived = _archive_path(destination, destination_time)
        _move(destination, archived, apply)
        _move(source, destination, apply)
    else:
        archived = _archive_path(destination, source_time)
        _move(source, archived, apply)


def _model_root_from_filename(filename):
    if "_openai_clip_" in filename:
        return OPENAI_ROOT
    if "_clip_" in filename:
        return OPENCLIP_ROOT
    raise ValueError(f"Cannot infer model family from legacy log {filename!r}")


def _method_from_log(path):
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file):
            match = MODEL_NAME_RE.search(line)
            if match:
                return canonical_method_name(match.group(1))
            if line_number >= 100:
                break

    lowered = path.name.lower()
    filename_aliases = {
        "clg-cbm": "CLG-CBM",
        "coda_prompt": "CODA-Prompt",
        "coda": "CODA-Prompt",
        "dualprompt": "DualPrompt",
        "engine": "ENGINE",
        "simplecil": "SimpleCIL",
        "zs_clip": "ZS-CLIP",
        "sagla": "SAGLA",
    }
    for prefix, method in filename_aliases.items():
        if lowered.startswith(prefix):
            return method
    for prefix in ("bofa", "l2p", "proof", "rapf"):
        if lowered.startswith(prefix):
            return canonical_method_name(prefix)
    raise ValueError(f"Cannot infer method from efficiency log {path}")


def _migrate_legacy_method_dirs(logs_root, apply):
    for legacy_dir, method_identifier in LEGACY_METHOD_DIRS.items():
        source_dir = logs_root / legacy_dir
        if not source_dir.is_dir():
            continue
        canonical = canonical_method_name(method_identifier)
        for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
            model_root = _model_root_from_filename(source.name)
            destination = logs_root / model_root / canonical / source.name
            _merge_log(source, destination, apply)


def _migrate_area_legacy(logs_root, apply):
    source_dir = logs_root / OPENCLIP_ROOT / "AREA_legacy"
    if not source_dir.is_dir():
        return
    for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(source_dir)
        destination = logs_root / OPENCLIP_ROOT / "AREA" / "legacy" / relative
        _merge_log(source, destination, apply)


def _migrate_efficiency_logs(logs_root, apply):
    source_dir = logs_root / "efficiency"
    if not source_dir.is_dir():
        return
    for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(source_dir)
        category = relative.parts[0]
        if category == "launcher":
            destination = logs_root / OPENCLIP_ROOT / "queue" / "efficiency" / source.name
        else:
            method = "SAGLA" if category == "sagla" else _method_from_log(source)
            suffix = Path(*relative.parts[1:])
            if category == "audit":
                suffix = Path("audit") / suffix
            destination = logs_root / OPENCLIP_ROOT / method / "efficiency" / suffix
        _merge_log(source, destination, apply)


def _queue_methods(filename):
    lowered = filename.lower()
    return {
        method for marker, method in QUEUE_METHOD_MARKERS.items() if marker in lowered
    }


def _queue_model_root(path):
    lowered = path.as_posix().lower()
    if any(
        marker in lowered
        for marker in ("openai-clip", "openai_clip", "vit-b-16", "vit_b16")
    ):
        return OPENAI_ROOT
    if "promptfusion" in lowered or "moe-adapters" in lowered:
        return OPENAI_ROOT
    if "laion-400m" in lowered or "_clip_" in lowered or "area_data" in lowered:
        return OPENCLIP_ROOT
    if _queue_methods(path.name):
        return OPENCLIP_ROOT
    raise ValueError(f"Cannot infer queue-log model family from {path}")


def _queue_destination(logs_root, model_root, source, suffix=None):
    methods = _queue_methods(source.name)
    if len(methods) == 1:
        method = next(iter(methods))
        root = logs_root / model_root / method / "queue"
    else:
        root = logs_root / model_root / "queue"
    return root / (suffix or Path(source.name))


def _migrate_existing_model_queues(logs_root, apply):
    for model_root in (OPENAI_ROOT, OPENCLIP_ROOT):
        queue_root = logs_root / model_root / "queue"
        if not queue_root.is_dir():
            continue
        for source in sorted(path for path in queue_root.rglob("*") if path.is_file()):
            methods = _queue_methods(source.name)
            if len(methods) != 1:
                continue
            suffix = source.relative_to(queue_root)
            destination = _queue_destination(
                logs_root, model_root, source, suffix=suffix
            )
            _merge_log(source, destination, apply)


def _migrate_result_logs(logs_root, results_root, apply):
    if not results_root.is_dir():
        return
    candidates = sorted(results_root.rglob("*.out"))
    candidates.extend(sorted(results_root.glob("*.log")))
    for source in candidates:
        model_root = _queue_model_root(source)
        destination = _queue_destination(logs_root, model_root, source)
        _merge_log(source, destination, apply)


def _migrate_moe_hydra_logs(logs_root, experiments_root, apply):
    if not experiments_root.is_dir():
        return
    for source in sorted(experiments_root.rglob("main.log")):
        relative = source.relative_to(experiments_root)
        destination = (
            logs_root
            / OPENAI_ROOT
            / "MoE-Adapters"
            / "archive"
            / "hydra"
            / relative
        )
        _merge_log(source, destination, apply)


def _remove_empty_directories(logs_root):
    for directory in sorted(
        (path for path in logs_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass


def _audit_layout(logs_root, results_root=None, experiments_root=None):
    allowed_model_roots = {
        OPENAI_ROOT,
        OPENCLIP_ROOT,
        "OpenCLIP_LAION2B_ViTB16",
    }
    errors = []
    checked_native_logs = 0

    for entry in logs_root.iterdir():
        if entry.name not in allowed_model_roots:
            errors.append(f"non-model entry at logs root: {entry}")

    for model_root in (path for path in logs_root.iterdir() if path.is_dir()):
        casefolded = {}
        for method_dir in (path for path in model_root.iterdir() if path.is_dir()):
            key = method_dir.name.casefold()
            if key in casefolded:
                errors.append(
                    f"case-duplicate method directories: {casefolded[key]} and {method_dir}"
                )
            casefolded[key] = method_dir

        for log_path in model_root.rglob("*.log"):
            relative = log_path.relative_to(logs_root)
            if len(relative.parts) < 3 or relative.parts[1] == "queue":
                continue
            method_dir = relative.parts[1]
            model_name = None
            backbone_name = None
            with log_path.open(encoding="utf-8", errors="replace") as log_file:
                for line_number, line in enumerate(log_file):
                    model_match = MODEL_NAME_RE.search(line)
                    if model_match:
                        model_name = model_match.group(1)
                    backbone_match = BACKBONE_NAME_RE.search(line)
                    if backbone_match:
                        backbone_name = backbone_match.group(1)
                    if (model_name and backbone_name) or line_number >= 100:
                        break
            if model_name is None and backbone_name is None:
                continue
            checked_native_logs += 1
            if model_name is not None:
                expected_method = canonical_method_name(model_name)
                if method_dir != expected_method:
                    errors.append(
                        f"method mismatch: {log_path} contains {model_name!r}, "
                        f"expected directory {expected_method!r}"
                    )
            if backbone_name is not None:
                expected_root = backbone_log_root(backbone_name)
                if model_root.name != expected_root:
                    errors.append(
                        f"model mismatch: {log_path} contains {backbone_name!r}, "
                        f"expected root {expected_root!r}"
                    )

    if results_root is not None and results_root.is_dir():
        outside_logs = sorted(results_root.rglob("*.out"))
        outside_logs.extend(sorted(results_root.glob("*.log")))
        errors.extend(f"runtime log remains outside logs/: {path}" for path in outside_logs)
    if experiments_root is not None and experiments_root.is_dir():
        errors.extend(
            f"Hydra duplicate log remains outside logs/: {path}"
            for path in sorted(experiments_root.rglob("main.log"))
        )

    if errors:
        raise SystemExit("Log layout audit failed:\n- " + "\n- ".join(errors))
    print(f"Log layout audit passed ({checked_native_logs} native logs checked).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--moe-experiments-root",
        type=Path,
        default=Path("MoE-Adapters4CL-MoE-Adapters/cil/experiments"),
    )
    parser.add_argument(
        "--apply", action="store_true", help="perform moves (default: dry run)"
    )
    parser.add_argument(
        "--check", action="store_true", help="audit the current layout without moving files"
    )
    args = parser.parse_args()

    if args.check:
        _audit_layout(args.logs_root, args.results_root, args.moe_experiments_root)
        return

    _migrate_legacy_method_dirs(args.logs_root, args.apply)
    _migrate_area_legacy(args.logs_root, args.apply)
    _migrate_efficiency_logs(args.logs_root, args.apply)
    _migrate_existing_model_queues(args.logs_root, args.apply)
    _migrate_result_logs(args.logs_root, args.results_root, args.apply)
    _migrate_moe_hydra_logs(args.logs_root, args.moe_experiments_root, args.apply)
    if args.apply:
        _remove_empty_directories(args.logs_root)
        _audit_layout(args.logs_root, args.results_root, args.moe_experiments_root)
    else:
        print("Dry run only; pass --apply to perform the migration.")


if __name__ == "__main__":
    main()
