#!/usr/bin/env python3
"""Run release-facing BlockBatch benchmark evaluations."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "blockBatching_ablation" / "results"
DEFAULT_BLOCK_SIZES = (4, 8, 16, 32, 64, 128)

MODEL_DEFAULTS = {
    "llada": ("GSAI-ML/LLaDA-8B-Instruct", "llada"),
    "llada_1p5": ("GSAI-ML/LLaDA-1.5", "llada_1p5"),
    "llada_1p5_instruct": ("GSAI-ML/LLaDA-1.5", "llada_1p5_instruct"),
    "dream": ("Dream-org/Dream-v0-Base-7B", "dream"),
}


@dataclass(frozen=True)
class TaskConfig:
    lm_eval_task: str
    result_task: str
    fewshot: int | None
    requires_code_eval: bool


TASKS = {
    "gsm8k": TaskConfig("gsm8k", "gsm8k", 5, False),
    "humaneval": TaskConfig("humaneval", "humaneval", None, True),
    "mbpp": TaskConfig("mbpp", "mbpp", 3, True),
    "math": TaskConfig("minerva_math", "math", 4, True),
}


def cache_dir_from_env() -> Path:
    return Path(
        os.environ.get(
            "CACHE_DIR",
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
        )
    )


def parse_block_lengths(raw: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in raw.replace(",", " ").replace("-", " ").split())
    invalid = [value for value in values if value not in DEFAULT_BLOCK_SIZES]
    if not values or invalid:
        raise argparse.ArgumentTypeError("Block lengths must be drawn from 4, 8, 16, 32, 64, 128.")
    return values


def block_size_label(values: tuple[int, ...]) -> str:
    return "-".join(str(value) for value in values)


def normalize_method(method: str) -> str:
    return "confidence" if method == "fast_dllm" else method


def env_for_model(model_key: str) -> dict[str, str]:
    env = os.environ.copy()
    model_family = "dream" if model_key == "dream" else "llada"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(PROJECT_ROOT),
            str(PROJECT_ROOT / model_family),
            env.get("PYTHONPATH", ""),
        ]
    )
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
    return env


def save_dir(args: argparse.Namespace, task: TaskConfig, method: str, block_length: int | None = None) -> Path:
    if method == "confidence":
        if block_length is None:
            raise ValueError("confidence runs require block_length")
        return args.results_dir / args.result_model / task.result_task / "confidence" / f"bs{block_length}"
    return args.results_dir / args.result_model / task.result_task / method


def llada_model_args(
    args: argparse.Namespace,
    method: str,
    run_dir: Path,
    block_length: int | None = None,
) -> str:
    common = {
        "model_path": args.model_path,
        "gen_length": args.gen_length,
        "cache_dir": args.cache_dir,
        "show_speed": "True",
        "save_dir": run_dir,
    }
    if method == "block_batching":
        common.update(
            {
                "threshold": args.threshold,
                "block_size_list": block_size_label(args.block_sizes),
            }
        )
    else:
        selected_block = block_length if block_length is not None else args.block_length
        common.update(
            {
                "steps": args.gen_length,
                "block_length": selected_block,
            }
        )
        if method == "confidence":
            common.update({"use_cache": "true", "dual_cache": "true", "threshold": args.threshold})
    return ",".join(f"{key}={value}" for key, value in common.items())


def dream_model_args(
    args: argparse.Namespace,
    method: str,
    run_dir: Path,
    block_length: int | None = None,
) -> str:
    common = {
        "pretrained": args.model_path,
        "max_new_tokens": args.gen_length,
        "add_bos_token": "true",
        "cache_dir": args.cache_dir,
        "show_speed": "True",
        "save_dir": run_dir,
    }
    if args.task != "math":
        common["escape_until"] = "true"
    if method == "block_batching":
        common.update({"threshold": args.threshold, "block_size_list": block_size_label(args.block_sizes)})
    else:
        selected_block = block_length if block_length is not None else args.block_length
        diffusion_steps = args.gen_length if method == "baseline" else args.gen_length // selected_block
        common.update(
            {
                "diffusion_steps": diffusion_steps,
                "alg": "entropy" if method == "baseline" else "confidence_threshold",
            }
        )
        if method == "confidence":
            common.update(
                {
                    "threshold": args.threshold,
                    "block_length": selected_block,
                    "use_cache": "true",
                    "dual_cache": "true",
                }
            )
    return ",".join(f"{key}={value}" for key, value in common.items())


def build_command(
    args: argparse.Namespace,
    task: TaskConfig,
    method: str,
    run_dir: Path,
    block_length: int | None = None,
) -> tuple[list[str], Path]:
    model_family = "dream" if args.model == "dream" else "llada"
    workdir = PROJECT_ROOT / model_family

    if model_family == "llada":
        script = "eval_blockBatching.py" if method == "block_batching" else "eval_llada.py"
        lm_model = "llada_blockbatching" if method == "block_batching" else "llada_dist"
        model_args = llada_model_args(args, method, run_dir, block_length=block_length)
    elif args.shard_index is not None:
        script = "eval_shard.py"
        lm_model = "dream"
        model_args = dream_model_args(args, method, run_dir, block_length=block_length)
    else:
        script = "eval_blockBatching.py" if method == "block_batching" else "eval.py"
        lm_model = "dream_blockbatching" if method == "block_batching" else "dream"
        model_args = dream_model_args(args, method, run_dir, block_length=block_length)

    command = ["accelerate", "launch", script]
    if args.shard_index is not None:
        command.extend(["--shard-index", str(args.shard_index), "--shard-count", str(args.shard_count)])
    command.extend(["--model", lm_model, "--model_args", model_args, "--tasks", task.lm_eval_task])
    if task.fewshot is not None:
        command.extend(["--num_fewshot", str(args.num_fewshot or task.fewshot)])
    command.extend(["--batch_size", "1", "--output_path", str(run_dir / "lm_eval"), "--log_samples"])
    if task.requires_code_eval:
        command.append("--confirm_run_unsafe_code")
    return command, workdir


def analyze(args: argparse.Namespace, task: TaskConfig, method: str) -> None:
    variant = "confidence" if method == "confidence" else method
    command = [
        sys.executable,
        str(PROJECT_ROOT / "blockBatching_ablation" / "analyze.py"),
        "--results-dir",
        str(args.results_dir),
        "--model",
        args.result_model,
        "--task",
        task.result_task,
        "--variant",
        variant,
        "--evaluate_accu",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_eval(args: argparse.Namespace) -> None:
    method = normalize_method(args.method)
    task = TASKS[args.task]
    block_lengths = args.block_lengths if method == "confidence" else (None,)

    for block_length in block_lengths:
        run_dir = save_dir(args, task, method, block_length=block_length)
        run_dir.mkdir(parents=True, exist_ok=True)
        command, workdir = build_command(args, task, method, run_dir, block_length=block_length)
        label = f"bs{block_length}" if block_length is not None else method
        print(f"=== {args.model}/{args.task}/{label} -> {run_dir} ===", flush=True)
        if args.dry_run:
            print(" ".join(command))
            continue
        subprocess.run(command, cwd=workdir, env=env_for_model(args.model), check=True)

    if args.analyze and not args.dry_run:
        analyze(args, task, method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_DEFAULTS), required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument(
        "--method",
        choices=("baseline", "confidence", "block_batching", "fast_dllm"),
        required=True,
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=cache_dir_from_env())
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--result-model", default=None)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--block-lengths", type=parse_block_lengths, default=DEFAULT_BLOCK_SIZES)
    parser.add_argument("--block-sizes", type=parse_block_lengths, default=DEFAULT_BLOCK_SIZES)
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    default_model_path, default_result_model = MODEL_DEFAULTS[args.model]
    args.model_path = args.model_path or default_model_path
    args.result_model = args.result_model or default_result_model
    if args.method == "fast_dllm":
        args.block_lengths = (args.block_length,)
    if args.shard_index is not None and not (args.model == "dream" and args.method == "baseline"):
        parser.error("--shard-index is currently supported for Dream baseline runs only.")
    return args


def main() -> int:
    run_eval(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
