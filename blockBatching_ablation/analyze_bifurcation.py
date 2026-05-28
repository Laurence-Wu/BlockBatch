#!/usr/bin/env python3
"""
Analyze where single-block-size generations first bifurcate.

For every model/task pair, this script compares each pair of block sizes and
reports the average generated-token position where the two answers first differ.

By default it reads:
    results/{model}/{task}/confidence/bs*/rank_*.jsonl

Usage:
    python analyze_bifurcation.py
    python analyze_bifurcation.py --model llada --task gsm8k
    python analyze_bifurcation.py --csv bifurcation.csv
    python analyze_bifurcation.py --tokenization whitespace

Metric:
    bifurcate length = length of the common token prefix before the first
    different token appears. If the two answers are identical up to the shorter
    answer, the shorter length is used. Lower values mean earlier bifurcation.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DEFAULT_VARIANT = "confidence"
DEFAULT_MODELS = ("dream", "llada")
DEFAULT_TASKS = ("gsm8k", "humaneval", "math", "mbpp")


@dataclass
class PairStats:
    total_bifurcate_len: float = 0.0
    total_compared: int = 0
    identical: int = 0
    total_min_len: int = 0

    def add(self, bifurcate_len: int, identical: bool, min_len: int) -> None:
        self.total_bifurcate_len += bifurcate_len
        self.total_compared += 1
        self.identical += int(identical)
        self.total_min_len += min_len

    @property
    def avg_bifurcate_len(self) -> float:
        if self.total_compared == 0:
            return math.nan
        return self.total_bifurcate_len / self.total_compared

    @property
    def identical_rate(self) -> float:
        if self.total_compared == 0:
            return math.nan
        return self.identical / self.total_compared

    @property
    def avg_min_len(self) -> float:
        if self.total_compared == 0:
            return math.nan
        return self.total_min_len / self.total_compared


def regex_tokenize(text: str) -> list[str]:
    """Split into word-ish tokens plus punctuation, preserving code symbols."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def whitespace_tokenize(text: str) -> list[str]:
    return text.split()


def char_tokenize(text: str) -> list[str]:
    return list(text)


def build_hf_tokenizer(tokenizer_name: str) -> Callable[[str], list[int]]:
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return lambda text: tokenizer.encode(text, add_special_tokens=False)


def first_bifurcate_len(left: list, right: list) -> tuple[int, bool, int]:
    """Return common-prefix length, whether sequences are identical, min length."""
    min_len = min(len(left), len(right))
    for idx in range(min_len):
        if left[idx] != right[idx]:
            return idx, False, min_len
    return min_len, len(left) == len(right), min_len


def parse_rank_jsonl(path: str, tokenizer: Callable[[str], list]) -> dict[int | str, list]:
    """Load one rank_*.jsonl file and map sample id to tokenized answer."""
    samples: dict[int | str, list] = {}
    with open(path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue

            # Existing block-batching rows often have sample_id; single-block
            # ablation rows usually rely on line order. Use 1-based line order
            # so it aligns with sample_id when both are present elsewhere.
            sample_id = row.get("sample_id", line_idx)
            answer = row.get("answer", "")
            samples[sample_id] = tokenizer(str(answer))
    return samples


def discover_rank_files(results_dir: str, model: str, task: str, variant: str) -> dict[int, list[str]]:
    """Return {block_size: [rank files]} for one model/task/variant."""
    pattern = os.path.join(results_dir, model, task, variant, "bs*", "rank_*.jsonl")
    files_by_bs: dict[int, list[str]] = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        bs_dir = os.path.basename(os.path.dirname(path))
        match = re.fullmatch(r"bs(\d+)", bs_dir)
        if not match:
            continue
        files_by_bs[int(match.group(1))].append(path)
    return dict(sorted(files_by_bs.items()))


def load_block_size_outputs(
    results_dir: str,
    model: str,
    task: str,
    variant: str,
    tokenizer: Callable[[str], list],
) -> dict[int, dict[int | str, list]]:
    """Load all answers grouped as {block_size: {sample_id: tokens}}."""
    files_by_bs = discover_rank_files(results_dir, model, task, variant)
    outputs: dict[int, dict[int | str, list]] = {}
    for block_size, paths in files_by_bs.items():
        merged: dict[int | str, list] = {}
        next_auto_id = 1
        for path in paths:
            samples = parse_rank_jsonl(path, tokenizer)
            for sample_id, tokens in samples.items():
                key = sample_id
                if key in merged:
                    # Multiple rank files without global sample ids are rare in
                    # this result tree. Avoid overwriting if it happens.
                    while next_auto_id in merged:
                        next_auto_id += 1
                    key = next_auto_id
                    next_auto_id += 1
                merged[key] = tokens
        outputs[block_size] = merged
    return outputs


def compute_pair_stats(outputs_by_bs: dict[int, dict[int | str, list]]) -> dict[tuple[int, int], PairStats]:
    stats: dict[tuple[int, int], PairStats] = {}
    for left_bs, right_bs in combinations(sorted(outputs_by_bs), 2):
        left_samples = outputs_by_bs[left_bs]
        right_samples = outputs_by_bs[right_bs]
        shared_ids = sorted(set(left_samples) & set(right_samples), key=str)
        pair_stats = PairStats()
        for sample_id in shared_ids:
            bifurcate_len, identical, min_len = first_bifurcate_len(
                left_samples[sample_id],
                right_samples[sample_id],
            )
            pair_stats.add(bifurcate_len, identical, min_len)
        stats[(left_bs, right_bs)] = pair_stats
    return stats


def fmt_float(value: float, width: int = 8, precision: int = 1) -> str:
    if math.isnan(value):
        return " " * (width - 1) + "-"
    return f"{value:{width}.{precision}f}"


def format_pair(pair: tuple[int, int]) -> str:
    return f"{pair[0]}-{pair[1]}"


def print_model_table(
    model: str,
    tasks: Iterable[str],
    rows_by_task: dict[str, dict[tuple[int, int], PairStats]],
    pair_order: list[tuple[int, int]],
) -> None:
    if not pair_order:
        print(f"\nNo block-size pairs found for model={model}.")
        return

    print(f"\n{'=' * 120}")
    print(f"Model: {model} | average bifurcate length by dataset and block-size pair")
    print("Lower means the two block sizes diverge earlier in the generated answer.")
    print(f"{'=' * 120}")

    first_col = "dataset"
    col_width = max(9, max(len(format_pair(pair)) for pair in pair_order) + 2)
    header = f"{first_col:<12}" + "".join(f"{format_pair(pair):>{col_width}}" for pair in pair_order)
    print(header)
    print("-" * len(header))

    for task in tasks:
        row_stats = rows_by_task.get(task, {})
        if not row_stats:
            continue
        line = f"{task:<12}"
        for pair in pair_order:
            stat = row_stats.get(pair)
            value = stat.avg_bifurcate_len if stat else math.nan
            line += fmt_float(value, width=col_width, precision=1)
        print(line)

    print("\nEarliest average bifurcation per dataset:")
    for task in tasks:
        row_stats = rows_by_task.get(task, {})
        candidates = [
            (pair, stat)
            for pair, stat in row_stats.items()
            if stat.total_compared > 0 and not math.isnan(stat.avg_bifurcate_len)
        ]
        if not candidates:
            continue
        pair, stat = min(candidates, key=lambda item: item[1].avg_bifurcate_len)
        print(
            f"  {task:<12} best={format_pair(pair):<7} "
            f"avg={stat.avg_bifurcate_len:.1f} n={stat.total_compared} "
            f"identical={stat.identical_rate:.1%}"
        )


def write_csv(
    path: str,
    all_stats: dict[tuple[str, str], dict[tuple[int, int], PairStats]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "dataset",
            "block_size_pair",
            "left_block_size",
            "right_block_size",
            "avg_bifurcate_len",
            "n",
            "identical_rate",
            "avg_min_len",
        ])
        for (model, task), pair_stats in sorted(all_stats.items()):
            for pair, stat in sorted(pair_stats.items()):
                if stat.total_compared == 0:
                    continue
                writer.writerow([
                    model,
                    task,
                    format_pair(pair),
                    pair[0],
                    pair[1],
                    f"{stat.avg_bifurcate_len:.6f}",
                    stat.total_compared,
                    f"{stat.identical_rate:.6f}",
                    f"{stat.avg_min_len:.6f}",
                ])


def build_tokenizer(args: argparse.Namespace) -> Callable[[str], list]:
    if args.tokenization == "regex":
        return regex_tokenize
    if args.tokenization == "whitespace":
        return whitespace_tokenize
    if args.tokenization == "char":
        return char_tokenize
    if args.tokenization == "hf":
        if not args.tokenizer_name:
            raise SystemExit("--tokenizer-name is required when --tokenization hf")
        return build_hf_tokenizer(args.tokenizer_name)
    raise SystemExit(f"unknown tokenization mode: {args.tokenization}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute average first-divergence positions between block-size outputs.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Results root. Default: blockBatching_ablation/results",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=DEFAULT_MODELS,
        help="Model to include. Can be repeated. Default: dream and llada.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=DEFAULT_TASKS,
        help="Dataset/task to include. Can be repeated. Default: all known tasks.",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help="Ablation variant folder under each task. Default: confidence.",
    )
    parser.add_argument(
        "--tokenization",
        choices=("regex", "whitespace", "char", "hf"),
        default="regex",
        help="How to split generated answers before comparison. Default: regex.",
    )
    parser.add_argument(
        "--tokenizer-name",
        help="Hugging Face tokenizer name/path for --tokenization hf.",
    )
    parser.add_argument(
        "--csv",
        help="Optional CSV output path for long-form stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = build_tokenizer(args)
    models = args.model or list(DEFAULT_MODELS)
    tasks = args.task or list(DEFAULT_TASKS)

    all_stats: dict[tuple[str, str], dict[tuple[int, int], PairStats]] = {}
    for model in models:
        rows_by_task: dict[str, dict[tuple[int, int], PairStats]] = {}
        pair_set: set[tuple[int, int]] = set()
        for task in tasks:
            outputs = load_block_size_outputs(
                args.results_dir,
                model,
                task,
                args.variant,
                tokenizer,
            )
            if len(outputs) < 2:
                continue
            pair_stats = compute_pair_stats(outputs)
            rows_by_task[task] = pair_stats
            all_stats[(model, task)] = pair_stats
            pair_set.update(pair_stats)

        print_model_table(model, tasks, rows_by_task, sorted(pair_set))

    if args.csv:
        write_csv(args.csv, all_stats)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
