#!/usr/bin/env python3
"""Export refresh-block-size ablation data as plot-ready JSON tables.

The output is intentionally table-first so graph scripts/notebooks can read
NFE and accuracy without re-parsing rank JSONL files:

  experiements/refresh_block_size_ablation/refresh_block_size_grid_data.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from refresh_block_size_ablation import REFRESH_BLOCK_SIZES, refresh_label, variant_name

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
OUT_JSON = EXPERIMENT_DIR / "refresh_block_size_grid_data.json"

MODELS = ["llada", "dream"]
TASKS = ["gsm8k", "humaneval", "mbpp", "math"]

sys.path.insert(0, str(PROJECT_ROOT / "blockBatching_ablation"))
import analyze as analyze_mod  # noqa: E402
from analysis_utils import accuracy_rows, json_count, json_number  # noqa: E402


def load_refresh_rows(model: str, task: str) -> list[dict]:
    """Return one row per expected refresh block size for model/task."""
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")

    variant_glob = "refresh_block_size_*"
    records = analyze_mod.load_records(str(RESULTS_DIR), model, task, variant=variant_glob)

    stats_by_size = {}
    for bl, n_samples, record_acc, avg_nfe, avg_tok, avg_lat, avg_tps in analyze_mod.compute_stats(
        records, ablation_by_dir=True
    ):
        stats_by_size[str(bl)] = {
            "n_samples": json_count(n_samples),
            "record_accuracy": json_number(record_acc),
            "avg_nfe": json_number(avg_nfe),
            "avg_tokens": json_number(avg_tok),
            "avg_latency_s": json_number(avg_lat),
            "avg_tps": json_number(avg_tps),
        }

    accuracy_by_size = {}
    for bl, n_accuracy_samples, accuracy in accuracy_rows(RESULTS_DIR, model, task, "refresh_block_size_*"):
        accuracy_by_size[str(bl)] = {
            "n_accuracy_samples": json_count(n_accuracy_samples),
            "accuracy": json_number(accuracy),
        }

    rows = []
    for refresh_block_size in REFRESH_BLOCK_SIZES:
        key = str(refresh_block_size)
        stats = stats_by_size.get(key, {})
        accu = accuracy_by_size.get(key, {})
        avg_nfe = stats.get("avg_nfe")
        accuracy = accu.get("accuracy", stats.get("record_accuracy"))
        rows.append({
            "model": model,
            "task": task,
            "method": "refresh_block_size",
            "variant": variant_name(refresh_block_size),
            "refresh_block_size": refresh_block_size,
            "x_label": refresh_label(refresh_block_size),
            "avg_nfe": avg_nfe,
            "accuracy": accuracy,
            "n_samples": stats.get("n_samples"),
            "n_accuracy_samples": accu.get("n_accuracy_samples"),
            "record_accuracy": stats.get("record_accuracy"),
            "avg_tokens": stats.get("avg_tokens"),
            "avg_latency_s": stats.get("avg_latency_s"),
            "avg_tps": stats.get("avg_tps"),
            "has_records": key in stats_by_size,
            "has_accuracy": accuracy is not None,
            "is_plotted": avg_nfe is not None and accuracy is not None,
        })
    return rows


def build_payload() -> dict:
    table = {}
    records = []
    panels = []
    for model in MODELS:
        table[model] = {}
        for task in TASKS:
            rows = load_refresh_rows(model, task)
            table[model][task] = rows
            records.extend(rows)
            available = [
                row["refresh_block_size"] for row in rows
                if row["is_plotted"]
            ]
            panels.append({
                "model": model,
                "task": task,
                "refresh_block_sizes_expected": REFRESH_BLOCK_SIZES,
                "refresh_block_sizes_available": available,
                "num_available": len(available),
                "num_expected": len(REFRESH_BLOCK_SIZES),
            })

    return {
        "description": (
            "Table data for refresh-block-size ablation. Each row is one "
            "model/task/refresh_block_size point. avg_nfe is computed from "
            "rank JSONL records; accuracy uses the same task-specific "
            "post-processing as blockBatching_ablation/analyze.py."
        ),
        "results_dir": str(RESULTS_DIR),
        "refresh_block_sizes": REFRESH_BLOCK_SIZES,
        "models": MODELS,
        "tasks": TASKS,
        "panels": panels,
        "table": table,
        "records": records,
    }


def main():
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved refresh-block-size table JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
