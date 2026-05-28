"""Shared helpers for BlockBatch analysis scripts."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def json_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def json_count(value: Any) -> int | None:
    value = json_number(value)
    return None if value is None else int(value)


def trailing_number_label(dirname: str) -> int | str:
    match = re.search(r"(\d+)$", dirname)
    return int(match.group(1)) if match else dirname


def rank_count(run_dir: Path) -> int:
    count = 0
    for path in run_dir.glob("rank_*.jsonl"):
        with path.open(errors="ignore") as handle:
            count += sum(1 for line in handle if line.strip())
    return count


def metric_from_lm_eval_results(payload: dict[str, Any], task: str) -> float | None:
    task_key = "minerva_math" if task == "math" else task
    task_results = payload.get("results", {}).get(task_key, {})
    preferred = {
        "gsm8k": ("exact_match,flexible-extract", "exact_match,strict-match"),
        "humaneval": ("pass@1,create_test", "pass@1", "pass_at_1,none"),
        "mbpp": ("pass_at_1,none", "pass@1,create_test", "pass@1,none"),
        "math": ("math_verify,none", "exact_match,none"),
    }.get(task, ())
    for key in preferred:
        value = json_number(task_results.get(key))
        if value is not None:
            return value
    for key, raw_value in task_results.items():
        if key.startswith(("pass@1", "pass_at_1", "exact_match", "math_verify")):
            value = json_number(raw_value)
            if value is not None:
                return value
    return None


def cached_accuracy_rows(
    results_dir: Path,
    model: str,
    task: str,
    variant_glob: str,
    label_fn=trailing_number_label,
) -> list[tuple[int | str, int, float]]:
    rows = []
    task_dir = results_dir / model / task
    for run_dir in sorted(path for path in task_dir.glob(variant_glob) if path.is_dir()):
        result_files = sorted(run_dir.glob("lm_eval/**/results_*.json"))
        if not result_files:
            continue
        try:
            payload = json.loads(result_files[-1].read_text())
        except json.JSONDecodeError:
            continue
        accuracy = metric_from_lm_eval_results(payload, task)
        if accuracy is not None:
            rows.append((label_fn(run_dir.name), rank_count(run_dir), accuracy))
    return rows


def load_analyzer():
    sys.path.insert(0, str(PROJECT_ROOT / "blockBatching_ablation"))
    import analyze as analyze_mod  # noqa: PLC0415

    return analyze_mod


def accuracy_rows(results_dir: Path, model: str, task: str, variant_glob: str, label_fn=trailing_number_label):
    cached_rows = cached_accuracy_rows(results_dir, model, task, variant_glob, label_fn=label_fn)
    if cached_rows:
        return cached_rows
    analyze_mod = load_analyzer()
    if task == "humaneval":
        return analyze_mod.evaluate_humaneval_accu(str(results_dir), model, task=task, variant=variant_glob)
    if task == "mbpp":
        return analyze_mod.evaluate_mbpp_accu(str(results_dir), model, variant=variant_glob)
    if task == "gsm8k":
        return analyze_mod.evaluate_gsm8k_accu(str(results_dir), model, variant=variant_glob)
    if task == "math":
        return analyze_mod.evaluate_math_accu(str(results_dir), model, variant=variant_glob)
    return []
