#!/usr/bin/env python3
"""Run refresh-block-size ablations with one public entrypoint."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"
REFRESH_BLOCK_SIZES = (4, 8, 16, 32, 64, 128)

DEFAULT_MODEL_PATHS = {
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
    "dream": "Dream-org/Dream-v0-Base-7B",
}


@dataclass(frozen=True)
class TaskConfig:
    lm_eval_task: str
    analyze_task: str
    fewshot: int | None
    default_limit: int | None
    requires_code_eval: bool


TASKS = {
    "gsm8k": TaskConfig("gsm8k", "gsm8k", 5, 500, False),
    "humaneval": TaskConfig("humaneval", "humaneval", None, None, True),
    "mbpp": TaskConfig("mbpp", "mbpp", 3, None, True),
    "math": TaskConfig("minerva_math", "math", 4, 500, False),
}


def variant_name(refresh_block_size: int) -> str:
    return f"refresh_block_size_{int(refresh_block_size)}"


def refresh_label(refresh_block_size: int) -> str:
    return str(int(refresh_block_size))


def sizes_for_shard(shard_index: int, shard_count: int) -> tuple[int, ...]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return REFRESH_BLOCK_SIZES[shard_index::shard_count]


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
    refresh_block_size: int,
) -> str:
    common = {
        "threshold": threshold,
        "block_size_list": block_sizes,
        "cache_dir": cache_dir,
        "show_speed": "True",
        "save_dir": save_dir,
        "sync_threshold": sync_threshold,
        "refresh_block_size": refresh_block_size,
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


def build_eval_command(args: argparse.Namespace, refresh_block_size: int) -> tuple[list[str], Path, Path, str]:
    task = TASKS[args.task]
    variant = variant_name(refresh_block_size)
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
        sync_threshold=args.sync_threshold,
        refresh_block_size=refresh_block_size,
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
    return command, workdir, save_dir, variant


def analyze_variant(args: argparse.Namespace, variant: str) -> None:
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
        variant,
        "--evaluate_accu",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_one(args: argparse.Namespace, refresh_block_size: int) -> None:
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            str(PROJECT_ROOT),
            str(PROJECT_ROOT / args.model),
            os.environ.get("PYTHONPATH", ""),
        ]
    )

    command, workdir, save_dir, variant = build_eval_command(args, refresh_block_size)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Running {args.model}/{args.task} {variant} "
        f"with block_sizes={args.block_sizes} -> {save_dir}",
        flush=True,
    )
    if args.dry_run:
        print(" ".join(command))
        return

    subprocess.run(command, cwd=workdir, check=True)
    if not args.no_analyze:
        analyze_variant(args, variant)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=("llada", "dream"), required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=cache_dir_from_env())
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--block-sizes", default="4-8-16-32-64-128")
    parser.add_argument("--sync-threshold", type=int, default=8)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-analyze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one refresh-block-size setting.")
    add_common_args(run_parser)
    run_parser.add_argument("--refresh-block-size", type=int, choices=REFRESH_BLOCK_SIZES, required=True)

    shard_parser = subparsers.add_parser("run-shard", help="Run a shard of refresh-block-size settings.")
    add_common_args(shard_parser)
    shard_parser.add_argument("--shard-index", type=int, default=None)
    shard_parser.add_argument("--shard-count", type=int, default=None)
    shard_parser.add_argument("--server", type=int, default=None, help="1-based shard index.")
    shard_parser.add_argument("--total-servers", type=int, default=None, help="Total shard count.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        run_one(args, args.refresh_block_size)
        return 0

    shard_index = args.shard_index
    shard_count = args.shard_count
    if args.server is not None:
        shard_index = args.server - 1
        shard_count = args.total_servers
    if shard_index is None or shard_count is None:
        raise ValueError("Provide --shard-index/--shard-count or --server/--total-servers.")

    refresh_sizes = sizes_for_shard(shard_index, shard_count)
    if not refresh_sizes:
        print(f"[skip] shard {shard_index}/{shard_count} has no refresh-block-size jobs", flush=True)
        return 0
    for index, refresh_block_size in enumerate(refresh_sizes, start=1):
        print(f"\n=== [{index}/{len(refresh_sizes)}] {variant_name(refresh_block_size)} ===", flush=True)
        run_one(args, refresh_block_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
