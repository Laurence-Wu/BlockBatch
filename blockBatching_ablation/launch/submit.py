#!/usr/bin/env python3
"""Submit benchmark and ablation jobs with generic Slurm settings."""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = Path(__file__).resolve().parent / "generated"


def shell_command(parts: list[str]) -> str:
    return shlex.join(str(part) for part in parts).replace("'${SLURM_ARRAY_TASK_ID}'", "${SLURM_ARRAY_TASK_ID}")


def benchmark_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "blockBatching_ablation/eval.py",
        "--model",
        args.model,
        "--task",
        args.task,
        "--method",
        args.method,
    ]
    if args.method == "fast_dllm":
        command.extend(["--block-length", str(args.block_length)])
    if args.analyze:
        command.append("--analyze")
    return command


def ablation_command(args: argparse.Namespace) -> list[str]:
    entrypoints = {
        "block_combo": "experiements/blockCombo/block_combo_ablation.py",
        "sync_threshold": "experiements/sync_threshold_ablation/sync_threshold_ablation.py",
        "refresh_block_size": "experiements/refresh_block_size_ablation/refresh_block_size_ablation.py",
    }
    command = [
        args.python,
        entrypoints[args.job],
        "run-shard",
        "--model",
        args.model,
        "--task",
        args.task,
        "--shard-index",
        "${SLURM_ARRAY_TASK_ID}",
        "--shard-count",
        str(args.shards),
    ]
    if args.no_analyze:
        command.append("--no-analyze")
    return command


def job_command(args: argparse.Namespace) -> list[str]:
    if args.job == "benchmark":
        return benchmark_command(args)
    return ablation_command(args)


def sbatch_header(args: argparse.Namespace) -> list[str]:
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={args.name}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --gres={args.gres}",
        f"#SBATCH --output={args.log_dir}/%x-%A_%a.out",
        f"#SBATCH --error={args.log_dir}/%x-%A_%a.err",
    ]
    if args.job != "benchmark":
        lines.append(f"#SBATCH --array=0-{args.shards - 1}")
    for option, value in (("partition", args.partition), ("account", args.account), ("qos", args.qos)):
        if value:
            lines.append(f"#SBATCH --{option}={value}")
    return lines


def build_script_path(args: argparse.Namespace) -> Path:
    return GENERATED_DIR / f"{args.name}.slurm.sh"


def render_script(args: argparse.Namespace) -> str:
    lines = [
        *sbatch_header(args),
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(PROJECT_ROOT))}",
        "export HF_ALLOW_CODE_EVAL=1",
        "export HF_DATASETS_TRUST_REMOTE_CODE=true",
    ]
    if args.conda_env:
        lines.extend(
            [
                "source \"${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}\"",
                f"conda activate {shlex.quote(args.conda_env)}",
            ]
        )
    lines.extend(["", shell_command(job_command(args)), ""])
    return "\n".join(lines)


def write_script(args: argparse.Namespace) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    path = build_script_path(args)
    path.write_text(render_script(args), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=("benchmark", "block_combo", "sync_threshold", "refresh_block_size"), required=True)
    parser.add_argument("--model", choices=("llada", "llada_1p5", "llada_1p5_instruct", "dream"), required=True)
    parser.add_argument("--task", choices=("gsm8k", "humaneval", "mbpp", "math"), required=True)
    parser.add_argument("--method", choices=("baseline", "confidence", "block_batching", "fast_dllm"), default="block_batching")
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--name", default=None)
    parser.add_argument("--python", default="python")
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "logs" / "slurm")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--no-analyze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.job != "benchmark" and args.model not in {"llada", "dream"}:
        parser.error("Ablation jobs currently support --model llada or dream.")
    if args.shards <= 0:
        parser.error("--shards must be positive.")
    args.name = args.name or f"{args.job}_{args.model}_{args.task}"
    return args


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(build_script_path(args))
        print(render_script(args))
        return 0
    submitted_script = write_script(args)
    subprocess.run(["sbatch", str(submitted_script)], cwd=PROJECT_ROOT, check=True)
    print(f"Submitted {submitted_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
