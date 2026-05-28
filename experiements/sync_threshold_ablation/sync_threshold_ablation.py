#!/usr/bin/env python3
"""Run and summarize the sync-threshold ablation.

This module replaces the previous collection of per-model/per-task shell
wrappers and shard runners with one entrypoint:

  python experiements/sync_threshold_ablation/sync_threshold_ablation.py run \
      --model llada --task gsm8k --sync-threshold 8

  python experiements/sync_threshold_ablation/sync_threshold_ablation.py run-shard \
      --model dream --task humaneval --shard-index 0 --shard-count 5

  python experiements/sync_threshold_ablation/sync_threshold_ablation.py plot
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets" / "ablations" / "sync_threshold"
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"
SYNC_THRESHOLDS = (4, 8, 16, 32, 64)

DEFAULT_MODEL_PATHS = {
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
    "dream": "Dream-org/Dream-v0-Base-7B",
}

sys.path.insert(0, str(PROJECT_ROOT / "blockBatching_ablation"))
import analyze as analyze_mod  # noqa: E402
from analysis_utils import cached_accuracy_rows, json_count, json_number  # noqa: E402

VANILLA = {
    "llada": {
        "gsm8k": {"accuracy": 0.7710, "avg_nfe": 256.0},
        "humaneval": {"accuracy": 0.4024, "avg_nfe": 256.0},
    },
    "dream": {
        "gsm8k": {"accuracy": 0.7513, "avg_nfe": 256.0},
        "humaneval": {"accuracy": 0.5000, "avg_nfe": 256.0},
    },
}

BLOCK_BATCHING = {
    "llada": {
        "gsm8k": {"accuracy": 0.7748, "avg_nfe": 63.2},
        "humaneval": {"accuracy": 0.3841, "avg_nfe": 64.6},
    },
    "dream": {
        "gsm8k": {"accuracy": 0.7248, "avg_nfe": 133.8},
        "humaneval": {"accuracy": 0.5244, "avg_nfe": 112.5},
    },
}

BB_COLORS = {
    "ink": "#2B2F33",
    "grid": "#C9C7BD",
    "paper": "#FFFFFF",
    "vanilla": "#8A8C7A",
    "green": "#2F8F6B",
    "green_dark": "#1F5F49",
    "accuracy": "#5AA6D6",
}

SYNC_COLORS = {
    4: "#D9E9F5",
    8: "#B6D4EA",
    16: "#7FB2D6",
    32: "#3F86B8",
    64: "#1E5F8A",
}


@dataclass(frozen=True)
class TaskConfig:
    lm_eval_task: str
    analyze_task: str
    fewshot: int | None
    default_limit: int | None
    requires_code_eval: bool


TASKS = {
    "gsm8k": TaskConfig("gsm8k", "gsm8k", 5, None, False),
    "humaneval": TaskConfig("humaneval", "humaneval", None, None, True),
    "mbpp": TaskConfig("mbpp", "mbpp", 3, None, True),
    "math": TaskConfig("minerva_math", "math", 4, 500, True),
}


def variant_name(sync_threshold: int) -> str:
    return f"sync_threshold_{int(sync_threshold)}"


def threshold_label(sync_threshold: int) -> str:
    return f"sync={int(sync_threshold)}"


def cache_dir_from_env() -> Path:
    return Path(
        os.environ.get(
            "CACHE_DIR",
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
        )
    )


def build_model_args(
    *,
    model: str,
    model_path: str,
    gen_length: int,
    threshold: float,
    block_sizes: str,
    cache_dir: Path,
    save_dir: Path,
    sync_threshold: int,
) -> str:
    common = {
        "threshold": threshold,
        "block_size_list": block_sizes,
        "cache_dir": cache_dir,
        "show_speed": "True",
        "save_dir": save_dir,
        "sync_threshold": sync_threshold,
    }
    if model == "llada":
        args = {
            "model_path": model_path,
            "gen_length": gen_length,
            "generator_variant": "generate_blockBatching_original_bulk",
            **common,
        }
    else:
        args = {
            "pretrained": model_path,
            "max_new_tokens": gen_length,
            "add_bos_token": "true",
            "escape_until": "true",
            **common,
        }
    return ",".join(f"{key}={value}" for key, value in args.items())


def build_eval_command(args: argparse.Namespace, sync_threshold: int) -> tuple[list[str], Path, Path]:
    task = TASKS[args.task]
    variant = variant_name(sync_threshold)
    save_dir = args.results_dir / args.model / task.analyze_task / variant
    model_path = args.model_path or DEFAULT_MODEL_PATHS[args.model]
    model_args = build_model_args(
        model=args.model,
        model_path=model_path,
        gen_length=args.gen_length,
        threshold=args.threshold,
        block_sizes=args.block_sizes,
        cache_dir=args.cache_dir,
        save_dir=save_dir,
        sync_threshold=sync_threshold,
    )

    if args.model == "llada":
        workdir = PROJECT_ROOT / "llada"
        lm_model = "llada_blockbatching"
    else:
        workdir = PROJECT_ROOT / "dream"
        lm_model = "dream_blockbatching"

    command = [
        "accelerate",
        "launch",
        "eval_blockBatching.py",
        "--tasks",
        task.lm_eval_task,
        "--model",
        lm_model,
        "--model_args",
        model_args,
        "--batch_size",
        "1",
        "--output_path",
        str(save_dir / "lm_eval"),
        "--log_samples",
    ]
    if task.fewshot is not None:
        command.extend(["--num_fewshot", str(args.num_fewshot or task.fewshot)])
    limit = args.limit if args.limit is not None else task.default_limit
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if task.requires_code_eval:
        command.append("--confirm_run_unsafe_code")

    return command, workdir, save_dir


def analyze_variant(args: argparse.Namespace, sync_threshold: int) -> None:
    task = TASKS[args.task]
    command = [
        sys.executable,
        str(PROJECT_ROOT / "blockBatching_ablation" / "analyze.py"),
        "--results-dir",
        str(args.results_dir),
        "--model",
        args.model,
        "--task",
        task.analyze_task,
        "--variant",
        variant_name(sync_threshold),
        "--evaluate_accu",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_one(args: argparse.Namespace, sync_threshold: int) -> None:
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(PROJECT_ROOT),
            str(PROJECT_ROOT / args.model),
            os.environ.get("PYTHONPATH", ""),
        ]
    )

    command, workdir, save_dir = build_eval_command(args, sync_threshold)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Running {args.model}/{args.task} {threshold_label(sync_threshold)} "
        f"-> {save_dir}",
        flush=True,
    )
    if args.dry_run:
        print(" ".join(command))
        return

    subprocess.run(command, cwd=workdir, check=True)
    if not args.no_analyze:
        analyze_variant(args, sync_threshold)


def thresholds_for_shard(shard_index: int, shard_count: int) -> list[int]:
    return [
        sync_threshold
        for idx, sync_threshold in enumerate(SYNC_THRESHOLDS)
        if idx % shard_count == shard_index
    ]


def load_sync_stats(results_dir: Path, model: str, task: str) -> dict[int, dict[str, Any]]:
    records = analyze_mod.load_records(str(results_dir), model, task, variant="sync_threshold_*")
    stats_rows = analyze_mod.compute_stats(records, ablation_by_dir=True)

    by_threshold: dict[str, dict[str, Any]] = {}
    for block_length, n_samples, record_acc, avg_nfe, avg_tok, avg_lat, avg_tps in stats_rows:
        by_threshold[str(block_length)] = {
            "n_samples": json_count(n_samples),
            "record_accuracy": json_number(record_acc),
            "avg_nfe": json_number(avg_nfe),
            "avg_tokens": json_number(avg_tok),
            "avg_latency_s": json_number(avg_lat),
            "avg_tps": json_number(avg_tps),
            "has_records": True,
        }

    accuracy_rows = cached_accuracy_rows(results_dir, model, task, "sync_threshold_*")
    for block_length, n_accuracy_samples, accuracy in accuracy_rows:
        point = by_threshold.setdefault(str(block_length), {})
        point.update(
            {
                "n_accuracy_samples": json_count(n_accuracy_samples),
                "accuracy": json_number(accuracy),
                "has_accuracy": True,
            }
        )

    return {int(key): value for key, value in by_threshold.items() if str(key).isdigit()}


def build_source_data(results_dir: Path) -> dict[str, Any]:
    stats: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for model in ("llada", "dream"):
        stats[model] = {}
        for task in ("gsm8k", "humaneval"):
            stats[model][task] = load_sync_stats(results_dir, model, task)

    records = []
    for model in ("llada", "dream"):
        for task in ("gsm8k", "humaneval"):
            for sync_threshold in SYNC_THRESHOLDS:
                point = stats[model][task].get(sync_threshold, {})
                records.append(
                    {
                        "model": model,
                        "task": task,
                        "method": "sync_threshold",
                        "sync_threshold": sync_threshold,
                        "avg_nfe": point.get("avg_nfe"),
                        "accuracy": point.get("accuracy"),
                        "n_samples": point.get("n_samples"),
                        "n_accuracy_samples": point.get("n_accuracy_samples"),
                        "has_records": bool(point.get("has_records")),
                        "has_accuracy": bool(point.get("has_accuracy")),
                    }
                )
            records.append(
                {
                    "model": model,
                    "task": task,
                    "method": "block_batching",
                    "sync_threshold": None,
                    **BLOCK_BATCHING[model][task],
                }
            )
            records.append(
                {
                    "model": model,
                    "task": task,
                    "method": "vanilla_reference",
                    "sync_threshold": None,
                    **VANILLA[model][task],
                }
            )

    return {
        "sync_thresholds": list(SYNC_THRESHOLDS),
        "models": ["llada", "dream"],
        "tasks": ["gsm8k", "humaneval"],
        "results_dir": str(results_dir),
        "records": records,
    }


def plot_value(value: Any, default: float = math.nan) -> float:
    value = json_number(value)
    return default if value is None else value


def draw_panel(ax: Any, model: str, payload: dict[str, Any]) -> None:
    tasks = ("gsm8k", "humaneval")
    records = payload["records"]
    ax_acc = ax.twinx()

    x_cursor = 0
    xticks: list[float] = []
    xticklabels: list[str] = []
    for task in tasks:
        xs = list(range(x_cursor, x_cursor + len(SYNC_THRESHOLDS) + 1))
        sync_points = [
            next(
                (
                    row
                    for row in records
                    if row["model"] == model
                    and row["task"] == task
                    and row["method"] == "sync_threshold"
                    and row["sync_threshold"] == threshold
                ),
                {},
            )
            for threshold in SYNC_THRESHOLDS
        ]
        bb_point = next(
            row
            for row in records
            if row["model"] == model and row["task"] == task and row["method"] == "block_batching"
        )
        vanilla = next(
            row
            for row in records
            if row["model"] == model and row["task"] == task and row["method"] == "vanilla_reference"
        )

        nfes = [plot_value(point.get("avg_nfe")) for point in sync_points] + [bb_point["avg_nfe"]]
        colors = [SYNC_COLORS[threshold] for threshold in SYNC_THRESHOLDS] + [BB_COLORS["green"]]
        ax.bar(xs, nfes, color=colors, width=0.65, edgecolor="white", linewidth=0.4)

        acc_pairs = [
            (x, plot_value(point.get("accuracy")))
            for x, point in zip(xs[:-1], sync_points)
            if point.get("accuracy") is not None
        ]
        if acc_pairs:
            lx, ly = zip(*acc_pairs)
            ax_acc.plot(lx, ly, color=BB_COLORS["accuracy"], marker="o", markersize=3.5, linewidth=1.2)
        ax_acc.scatter(
            [xs[-1]],
            [bb_point["accuracy"]],
            color=BB_COLORS["green"],
            edgecolors=BB_COLORS["green_dark"],
            linewidths=0.5,
            s=24,
            zorder=5,
        )
        ax_acc.hlines(
            vanilla["accuracy"],
            xs[0] - 0.5,
            xs[-1] + 0.5,
            colors=BB_COLORS["vanilla"],
            linestyles="--",
            linewidth=1.0,
        )
        ax.text(
            (xs[0] + xs[-1]) / 2,
            -0.16,
            "GSM8K" if task == "gsm8k" else "HumanEval",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )
        xticks.extend(xs)
        xticklabels.extend([str(threshold) for threshold in SYNC_THRESHOLDS] + ["BB"])
        x_cursor += len(SYNC_THRESHOLDS) + 2

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=7)
    ax.set_ylabel("Avg NFE", fontsize=8, color=BB_COLORS["ink"])
    ax_acc.set_ylabel("Accuracy", fontsize=8, color=BB_COLORS["accuracy"])
    ax.grid(axis="y", color=BB_COLORS["grid"], linewidth=0.3, alpha=0.5)
    ax.set_title("LLaDA" if model == "llada" else "Dream", fontsize=9, color=BB_COLORS["ink"])


def write_plot(args: argparse.Namespace) -> None:
    payload = build_source_data(args.results_dir)
    data_path = args.output.with_name(args.output.stem + "_data.json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {data_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import matplotlib.lines as mlines  # noqa: PLC0415
        import matplotlib.patches as mpatches  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        print(f"[skip plot] {exc}; source data was still written.")
        return

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": BB_COLORS["ink"],
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.1), facecolor=BB_COLORS["paper"])
    for ax, model in zip(axes, ("llada", "dream")):
        draw_panel(ax, model, payload)

    handles = [
        *[mpatches.Patch(color=SYNC_COLORS[threshold], label=f"sync={threshold}") for threshold in SYNC_THRESHOLDS],
        mpatches.Patch(color=BB_COLORS["green"], label="BlockBatch"),
        mlines.Line2D([], [], color=BB_COLORS["accuracy"], marker="o", linewidth=1.2, label="Accuracy"),
        mlines.Line2D([], [], color=BB_COLORS["vanilla"], linestyle="--", linewidth=1.0, label="Vanilla acc."),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, framealpha=0.95)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.88, bottom=0.27, wspace=0.42)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor=BB_COLORS["paper"])
    plt.close(fig)
    print(f"Wrote {args.output}")


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("llada", "dream"), required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=cache_dir_from_env())
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--block-sizes", default="4-8-16-32-64")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-analyze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one sync-threshold setting.")
    add_common_run_args(run_parser)
    run_parser.add_argument("--sync-threshold", type=int, choices=SYNC_THRESHOLDS, required=True)

    shard_parser = subparsers.add_parser("run-shard", help="Run a threshold shard.")
    add_common_run_args(shard_parser)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--shard-count", type=int, required=True)

    plot_parser = subparsers.add_parser("plot", help="Write source data and grid plot.")
    plot_parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    plot_parser.add_argument("--output", type=Path, default=ASSETS_DIR / "sync_threshold_grid.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        run_one(args, args.sync_threshold)
    elif args.command == "run-shard":
        for sync_threshold in thresholds_for_shard(args.shard_index, args.shard_count):
            run_one(args, sync_threshold)
    elif args.command == "plot":
        write_plot(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
