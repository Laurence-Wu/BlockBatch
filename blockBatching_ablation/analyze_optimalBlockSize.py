#!/usr/bin/env python3
"""Optimal Block Size ablation analysis for GSM8K, HumanEval, MBPP, and MATH.

Shows for each block size (confidence ablation) + Block Batching + Oracle:
  - Avg NFE (bars)
  - Accuracy (line)
  - Dashed reference at bs=32 confidence accuracy

Oracle: per sample, inspect all six confidence block-size outputs. The oracle
        NFE is the lowest NFE among them, while oracle accuracy is counted as
        correct if any block size produces a correct answer for that sample.
"""
from __future__ import annotations

import json
import math
import os
import sys
import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "blockBatching_ablation"))
sys.path.insert(0, str(PROJECT_ROOT / "dream"))
sys.path.insert(0, str(PROJECT_ROOT / "llada"))

import analyze as analyze_mod

RESULTS_BASE = PROJECT_ROOT / "blockBatching_ablation" / "results"

# ── Block Batching paper palette (NeurIPS-style muted green/red) ──────────────
# Role mapping per guidance:
#   BBVanilla  → Vanilla baseline / bs=32 reference line
#   BBRed      → Oracle / accuracy line
#   BBGreen    → Block Batching (our main method)
#   BBGreenDk  → Block Batching emphasized / outlines
#   BBSlate    → Secondary / muted elements
#   BBInk      → Text, axes
#   BBGrid     → Gridlines
#   BBPaper    → Figure/axes background
#   BBPanel    → Legend background
BB = {
    "ink":     "#2B2F33",   # BBInk
    "grid":    "#C9C7BD",   # BBGrid
    "paper":   "#FFFFFF",   # BBPaper   — pure white figure/axes background
    "panel":   "#FFFFFF",   # BBPanel   — pure white legend background
    "vanilla": "#8A8C7A",   # BBVanilla — bs=32 reference dashed line
    "green":   "#1F5A3D",   # deep muted green — Block Batching bar
    "green_dk":"#123C29",   # darker green outline/emphasis
    "green_lt":"#DDE8D6",   # muted light green fill
    "red":     "#7A1F1F",   # deep muted red — Oracle / accuracy
    "red_dk":  "#4F1414",   # darker red outline/emphasis
    "red_lt":  "#E7D1CF",   # muted light red fill
    "slate":   "#657489",   # BBSlate   — muted secondary
}

# Confidence block-size bars use the block-size green gradient per guidance
# (light→dark as block size increases, same scale used everywhere in paper)
CONF_BAR_COLORS = {
    4:   "#D7E6D0",   # lightest muted green
    8:   "#B8D2AC",
    16:  "#8FB987",
    32:  "#5F9463",
    64:  "#356B46",
    128: "#1F4D34",   # darkest muted green
}

LABEL_SIZE = 14
TITLE_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 11
SPINE_WIDTH = 1.6
LEGEND_FRAME_WIDTH = 1.5

# ── Coarse data from experiments (accuracy + avg NFE per block size) ───────────
# Format: {model: {task: {method_or_blocklen: {acc, nfe}}}}
COARSE = {
    "llada": {
        "gsm8k": {
            "baseline": {"acc": 0.7710, "nfe": 256.0},
            4:          {"acc": 0.7763, "nfe": 121.2},
            8:          {"acc": 0.7953, "nfe": 100.4},
            16:         {"acc": 0.7854, "nfe": 90.8},
            32:         {"acc": 0.7870, "nfe": 86.2},
            64:         {"acc": 0.7657, "nfe": 82.8},
            128:        {"acc": 0.7165, "nfe": 85.2},
            "bb":       {"acc": 0.7748, "nfe": 63.2},
        },
        "humaneval": {
            "baseline": {"acc": 0.4024, "nfe": 256.0},
            4:          {"acc": 0.4146, "nfe": 120.6},
            8:          {"acc": 0.3841, "nfe": 99.3},
            16:         {"acc": 0.3659, "nfe": 90.6},
            32:         {"acc": 0.3659, "nfe": 90.4},
            64:         {"acc": 0.3659, "nfe": 91.7},
            128:        {"acc": 0.3720, "nfe": 93.1},
            "bb":       {"acc": 0.3841, "nfe": 64.6},
        },
        "math": {
            "baseline": {"acc": None,   "nfe": 256.0},
            4:          {"acc": 0.3822, "nfe": 137.1},
            8:          {"acc": 0.3748, "nfe": 120.4},
            16:         {"acc": 0.3700, "nfe": 113.0},
            32:         {"acc": 0.3712, "nfe": 107.2},
            64:         {"acc": 0.3716, "nfe": 104.0},
            128:        {"acc": 0.3534, "nfe": 104.9},
            "bb":       {"acc": 0.3720, "nfe": 80.4},
        },
        "mbpp": {
            "baseline": {"acc": 0.4140, "nfe": 256.0},
            4:          {"acc": 0.4140, "nfe": 116.9},
            8:          {"acc": 0.4080, "nfe": 92.2},
            16:         {"acc": 0.4060, "nfe": 75.9},
            32:         {"acc": 0.3840, "nfe": 68.2},
            64:         {"acc": 0.4020, "nfe": 67.5},
            128:        {"acc": 0.4100, "nfe": 74.9},
            "bb":       {"acc": 0.3960, "nfe": 45.3},
        },
    },
    "dream": {
        "gsm8k": {
            "baseline": {"acc": 0.7513, "nfe": 256.0},
            4:          {"acc": 0.7635, "nfe": 203.2},
            8:          {"acc": 0.7627, "nfe": 180.9},
            16:         {"acc": 0.7589, "nfe": 168.1},
            32:         {"acc": 0.7362, "nfe": 162.3},
            64:         {"acc": 0.7346, "nfe": 162.5},
            128:        {"acc": 0.7271, "nfe": 159.8},
            "bb":       {"acc": 0.7248, "nfe": 133.8},
        },
        "humaneval": {
            "baseline": {"acc": 0.5000, "nfe": 256.0},
            4:          {"acc": 0.5183, "nfe": 194.0},
            8:          {"acc": 0.4817, "nfe": 162.9},
            16:         {"acc": 0.4817, "nfe": 151.0},
            32:         {"acc": 0.5244, "nfe": 156.1},
            64:         {"acc": 0.5427, "nfe": 153.3},
            128:        {"acc": 0.5427, "nfe": 142.4},
            "bb":       {"acc": 0.5244, "nfe": 112.5},
        },
        "math": {
            "baseline": {"acc": None,   "nfe": 256.0},
            4:          {"acc": 0.4000, "nfe": 185.2},
            8:          {"acc": 0.3930, "nfe": 148.9},
            16:         {"acc": 0.3938, "nfe": 131.9},
            32:         {"acc": 0.3934, "nfe": 121.9},
            64:         {"acc": 0.3946, "nfe": 120.3},
            128:        {"acc": 0.3990, "nfe": 117.5},
            "bb":       {"acc": 0.3944, "nfe": 92.8},
        },
        "mbpp": {
            "baseline": {"acc": 0.5580, "nfe": 256.0},
            4:          {"acc": 0.5760, "nfe": 173.0},
            8:          {"acc": 0.5500, "nfe": 132.8},
            16:         {"acc": 0.5420, "nfe": 119.2},
            32:         {"acc": 0.5320, "nfe": 111.9},
            64:         {"acc": 0.5460, "nfe": 105.6},
            128:        {"acc": 0.5500, "nfe": 102.2},
            "bb":       {"acc": 0.5200, "nfe": 81.1},
        },
    },
}

BLOCK_SIZES = [4, 8, 16, 32, 64, 128]
MODELS = ("llada", "dream")
TASKS = ("math", "mbpp", "gsm8k", "humaneval")
MODEL_LABELS = {"llada": "LLaDA", "dream": "Dream"}
TASK_LABELS = {
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "math": "MATH",
}


def _bold_ticklabels(*axes) -> None:
    for ax in axes:
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")


def _style_paper_axes(ax, ax_acc=None) -> None:
    """Apply the thicker square-frame style used by the paper figures."""
    ax.set_facecolor(BB["paper"])
    for side in ("left", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(BB["ink"])
        ax.spines[side].set_linewidth(SPINE_WIDTH)

    if ax_acc is None:
        ax.spines["right"].set_visible(True)
        ax.spines["right"].set_color(BB["ink"])
        ax.spines["right"].set_linewidth(SPINE_WIDTH)
    else:
        ax.spines["right"].set_visible(False)
        ax_acc.set_facecolor(BB["paper"])
        for side in ("left", "bottom", "top"):
            ax_acc.spines[side].set_visible(False)
        ax_acc.spines["right"].set_visible(True)
        ax_acc.spines["right"].set_color(BB["ink"])
        ax_acc.spines["right"].set_linewidth(SPINE_WIDTH)


# ── Oracle NFE computation ────────────────────────────────────────────────────

def _read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def _load_per_sample_confidence(results_dir, model, task):
    """Load {sample_idx: {block_size: record}} for confidence ablation."""
    per_sample = defaultdict(dict)
    for bs in BLOCK_SIZES:
        bs_dir = results_dir / model / task / "confidence" / f"bs{bs}"
        if not bs_dir.exists():
            continue
        idx = 0
        for path in sorted(bs_dir.glob("rank_*.jsonl")):
            for rec in _read_jsonl(path):
                per_sample[idx][bs] = rec
                idx += 1
    return per_sample


def _sample_files(bs_dir: Path, task: str) -> list[Path]:
    if task == "math":
        patterns = [
            "lm_eval/samples_minerva_math_*.jsonl",
            "lm_eval/*/samples_minerva_math_*.jsonl",
        ]
    elif task == "mbpp":
        patterns = [
            "lm_eval/samples_mbpp_*.jsonl",
            "lm_eval/*/samples_mbpp_*.jsonl",
        ]
    else:
        patterns = [
            f"lm_eval/samples_{task}_*.jsonl",
            f"lm_eval/*/samples_{task}_*.jsonl",
        ]
    files = []
    for pattern in patterns:
        files.extend(bs_dir.glob(pattern))
    return sorted(files)


def _load_task_samples(bs_dir: Path, task: str) -> list[dict]:
    samples = []
    for path in _sample_files(bs_dir, task):
        samples.extend(_read_jsonl(path))
    if task == "gsm8k":
        samples = [s for s in samples if s.get("filter") == "flexible-extract"]
    return samples


def _sample_key(sample: dict, fallback_idx: int):
    """Stable key across block sizes when lm_eval metadata is available."""
    for fields in (
        ("doc_hash", "target_hash"),
        ("prompt_hash", "target_hash"),
    ):
        vals = tuple(sample.get(field) for field in fields)
        if all(v is not None for v in vals):
            return vals
    if sample.get("doc_id") is not None:
        return ("doc_id", sample["doc_id"])
    return ("idx", fallback_idx)


def _sample_correct(sample: dict, task: str) -> bool:
    if task == "math":
        if sample.get("math_verify") is None:
            raise KeyError("MATH sample is missing required math_verify metric")
        return bool(sample["math_verify"])
    if task == "gsm8k":
        if sample.get("filter") != "flexible-extract":
            raise ValueError("GSM8K oracle must use flexible-extract sample rows")
        if sample.get("exact_match") is None:
            raise KeyError("GSM8K sample is missing exact_match")
        return bool(sample["exact_match"])
    if task == "mbpp":
        for key in ("pass_at_1", "pass@1", "exact_match", "acc", "correct"):
            if sample.get(key) is not None:
                return bool(sample[key])
        return False
    for key in ("pass_at_1", "pass@1", "exact_match", "acc", "correct"):
        if sample.get(key) is not None:
            return bool(sample[key])
    return False


def _load_per_sample_confidence_with_cached_correctness(results_dir, model, task):
    """Load per-sample NFE and cached lm_eval correctness for each block size."""
    per_sample = defaultdict(dict)
    for bs in BLOCK_SIZES:
        bs_dir = results_dir / model / task / "confidence" / f"bs{bs}"
        if not bs_dir.exists():
            raise FileNotFoundError(f"Missing confidence directory: {bs_dir}")

        rank_records = []
        for path in sorted(bs_dir.glob("rank_*.jsonl")):
            rank_records.extend(_read_jsonl(path))

        samples = _load_task_samples(bs_dir, task)
        if not samples or len(samples) != len(rank_records):
            raise RuntimeError(
                f"{model}/{task}/bs{bs}: sample/rank mismatch "
                f"({len(samples)} samples vs {len(rank_records)} rank records)"
            )

        for idx, (rank_rec, sample) in enumerate(zip(rank_records, samples)):
            key = _sample_key(sample, idx)
            per_sample[key][bs] = {
                "nfe": float(rank_rec.get("nfe", math.nan)),
                "correct": _sample_correct(sample, task),
            }
    return per_sample


def _load_model_sanitize(model: str):
    import importlib.util

    sanitize_path = PROJECT_ROOT / model / "sanitize.py"
    spec = importlib.util.spec_from_file_location(f"{model}_sanitize_for_oracle", sanitize_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load sanitize.py for {model}: {sanitize_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sanitize


def _mbpp_entry_point(sample: dict):
    test_list = sample.get("doc", {}).get("test_list", [])
    if not test_list:
        return None
    import re

    first_test = test_list[0] if isinstance(test_list, list) else test_list
    match = re.search(r"assert\s+(\w+)\s*\(", first_test)
    return match.group(1) if match else None


def _build_code_eval_prediction(sample: dict, task: str, sanitize):
    raw = sample["resps"][0][0]
    if task == "humaneval":
        completion = raw.split("```python\n", 1)[-1].split("```")[0]
        pred = sanitize(
            sample["doc"]["prompt"] + "\n" + completion,
            sample["doc"]["entry_point"],
        )
    elif task == "mbpp":
        completion = raw.split("```python\n", 1)[-1].split("```")[0]
        completion = completion.split("[DONE]", 1)[0]
        pred = sanitize(completion, _mbpp_entry_point(sample))
    else:
        raise ValueError(f"Unsupported code task for oracle: {task}")
    return sample["target"], pred


def _load_code_confidence_with_correctness(results_dir, model, task):
    """Load HumanEval/MBPP correctness with the repository code_eval postprocess."""
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    try:
        import evaluate as hf_evaluate

        sanitize = _load_model_sanitize(model)
        code_eval = hf_evaluate.load("code_eval")
    except Exception as exc:
        raise RuntimeError(f"code_eval setup failed for {model}/{task}: {exc}") from exc

    per_sample = defaultdict(dict)
    refs, preds, keys = [], [], []

    for bs in BLOCK_SIZES:
        bs_dir = results_dir / model / task / "confidence" / f"bs{bs}"
        if not bs_dir.exists():
            raise FileNotFoundError(f"Missing confidence directory: {bs_dir}")

        rank_records = []
        for path in sorted(bs_dir.glob("rank_*.jsonl")):
            rank_records.extend(_read_jsonl(path))

        samples = _load_task_samples(bs_dir, task)

        if not samples or len(samples) != len(rank_records):
            raise RuntimeError(
                f"{model}/{task}/bs{bs}: sample/rank mismatch "
                f"({len(samples)} samples vs {len(rank_records)} rank records)"
            )

        for idx, (rank_rec, sample) in enumerate(zip(rank_records, samples)):
            key = _sample_key(sample, idx)
            per_sample[key][bs] = {
                "nfe": float(rank_rec.get("nfe", math.nan)),
                "correct": False,
            }
            try:
                ref, pred = _build_code_eval_prediction(sample, task, sanitize)
            except Exception:
                continue
            refs.append(ref)
            preds.append([pred])
            keys.append((key, bs))

    if refs:
        try:
            result = code_eval.compute(references=refs, predictions=preds, k=[1])
            details = result[1]
            for i, (key, bs) in enumerate(keys):
                passed = details[i][0][1].get("passed", False)
                per_sample[key][bs]["correct"] = bool(passed)
        except Exception as exc:
            raise RuntimeError(f"code_eval compute failed for {model}/{task}: {exc}") from exc

    return per_sample


def _load_per_sample_confidence_with_correctness(results_dir, model, task):
    if task in {"humaneval", "mbpp"}:
        return _load_code_confidence_with_correctness(results_dir, model, task)
    return _load_per_sample_confidence_with_cached_correctness(results_dir, model, task)


def _math_correct(rec: dict) -> bool:
    """Return True if the math record is marked correct by lm_eval."""
    # lm_eval stores exact_match in samples_*.jsonl or per-record
    for key in ("exact_match", "acc", "correct"):
        v = rec.get(key)
        if v is not None:
            return bool(v)
    return False


def _mbpp_correct(rec: dict, model_name: str) -> bool:
    """Evaluate MBPP pass@1 for a single record."""
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    try:
        from sanitize import sanitize
        import evaluate as hf_evaluate
        code_eval = hf_evaluate.load("code_eval")
        doc = rec.get("doc", {})
        entry_point = None
        test_list = doc.get("test_list", [])
        if test_list:
            import re
            m = re.search(r"assert\s+(\w+)\s*\(", test_list[0] if isinstance(test_list, list) else test_list)
            if m:
                entry_point = m.group(1)
        raw = rec.get("resps", [[""]])[0][0] if rec.get("resps") else rec.get("answer", "")
        clean = raw.split("```python\n", 1)[-1].split("```")[0].split("[DONE]")[0]
        pred = sanitize(clean, entry_point)
        ref = rec.get("target", "")
        if not ref:
            return False
        result = code_eval.compute(references=[ref], predictions=[[pred]], k=[1])
        return result[0]["pass@1"] > 0.5
    except Exception:
        return False


def compute_oracle(results_dir, model, task):
    """Compute oracle NFE and accuracy for a given model/task.

    Oracle: for each sample, inspect the six confidence block sizes.
      - opt_nfe: lowest NFE among attempted block sizes.
      - opt_acc: correct if any attempted block size is correct.
    """
    per_sample = _load_per_sample_confidence_with_correctness(results_dir, model, task)
    if not per_sample:
        print(f"  [oracle] no per-sample data for {model}/{task}")
        return math.nan, math.nan

    oracle_nfes = []
    oracle_correct = []

    for idx in sorted(per_sample):
        bl_recs = per_sample[idx]
        valid = [
            (float(rec.get("nfe", math.nan)), bs, bool(rec.get("correct")))
            for bs, rec in bl_recs.items()
            if not math.isnan(float(rec.get("nfe", math.nan)))
        ]
        if not valid:
            continue

        chosen_nfe, _chosen_bs, _chosen_correct = min(valid, key=lambda item: (item[0], item[1]))
        any_correct = any(bool(rec.get("correct")) for rec in bl_recs.values())
        oracle_nfes.append(chosen_nfe)
        oracle_correct.append(1.0 if any_correct else 0.0)

    n = len(oracle_nfes)
    if n == 0:
        return math.nan, math.nan
    return (sum(oracle_nfes) / n,
            sum(oracle_correct) / n)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _valid_number(value) -> bool:
    if value is None:
        return False
    return not math.isnan(float(value))


def draw_panel(ax, model: str, oracles: dict):
    """Draw one model panel with four dataset groups."""
    group_labels = [str(bs) for bs in BLOCK_SIZES] + ["BB", "Oracle"]
    group_width = len(group_labels)
    group_gap = 1.8
    bar_width = 0.62

    ax_acc = ax.twinx()
    all_xs = []
    all_tick_labels = []
    all_nfes = []
    all_accs = []

    for group_idx, task in enumerate(TASKS):
        data = COARSE[model][task]
        opt_nfe, opt_acc = oracles[(model, task)]
        start = group_idx * (group_width + group_gap)
        xs = [start + i for i in range(group_width)]

        nfe_vals = [data[bs]["nfe"] for bs in BLOCK_SIZES]
        confidence_acc_vals = [data[bs]["acc"] for bs in BLOCK_SIZES]
        bb_nfe = data["bb"]["nfe"]
        bb_acc = data["bb"]["acc"]
        nfe_vals += [bb_nfe, opt_nfe if _valid_number(opt_nfe) else math.nan]

        bar_colors = ([CONF_BAR_COLORS.get(bs, BB["slate"]) for bs in BLOCK_SIZES]
                      + [BB["green"], BB["red"]])

        bars = ax.bar(xs, nfe_vals, color=bar_colors, width=bar_width,
                      edgecolor="white", linewidth=0.6, alpha=0.92, zorder=2)
        for bar in bars[-2:]:
            bar.set_edgecolor(BB["ink"])
            bar.set_linewidth(1.2)

        valid_xy = [
            (x, a) for x, a in zip(xs[:len(BLOCK_SIZES)], confidence_acc_vals)
            if _valid_number(a)
        ]
        if valid_xy:
            lx, ly = zip(*valid_xy)
            ax_acc.plot(lx, ly, color=BB["red"], marker="o", markersize=4.8,
                        linewidth=1.8, zorder=4)
            all_accs.extend(float(v) for v in ly)

        if _valid_number(bb_acc):
            ax_acc.scatter([xs[-2]], [bb_acc], color=BB["green"], marker="D",
                           s=42, edgecolors=BB["ink"], linewidths=1.0, zorder=5)
            all_accs.append(float(bb_acc))
        if _valid_number(opt_acc):
            ax_acc.scatter([xs[-1]], [opt_acc], color=BB["red"], marker="s",
                           s=46, edgecolors=BB["ink"], linewidths=1.0, zorder=5)
            all_accs.append(float(opt_acc))

        bs32_acc = data[32]["acc"]
        if _valid_number(bs32_acc):
            ax_acc.hlines(bs32_acc, xs[0] - 0.35, xs[-1] + 0.35,
                          color=BB["vanilla"], linewidth=1.5,
                          linestyle="--", alpha=0.9, zorder=3)

        ax.axvline(xs[BLOCK_SIZES.index(32)], color=BB["grid"],
                   linewidth=0.9, linestyle=":", alpha=0.65, zorder=1)

        center = (xs[0] + xs[-1]) / 2
        ax.text(center, -0.15, TASK_LABELS[task], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=LABEL_SIZE, fontweight="bold",
                color=BB["ink"], clip_on=False)

        all_xs.extend(xs)
        all_tick_labels.extend(group_labels)
        all_nfes.extend(float(v) for v in nfe_vals if _valid_number(v))

    _style_paper_axes(ax, ax_acc)

    ax.set_xticks(all_xs)
    ax.set_xticklabels(
        all_tick_labels,
        fontsize=max(8, TICK_SIZE - 2),
        color=BB["ink"],
        fontweight="bold",
        rotation=45,
        ha="right",
    )
    ax.set_xlim(min(all_xs) - 0.8, max(all_xs) + 0.8)
    if all_nfes:
        ax.set_ylim(0.0, max(all_nfes) * 1.18)
    if all_accs:
        lo = max(0.0, min(all_accs) - 0.06)
        hi = min(1.0, max(all_accs) + 0.06)
        if hi - lo < 0.18:
            mid = (hi + lo) / 2
            lo = max(0.0, mid - 0.09)
            hi = min(1.0, mid + 0.09)
        ax_acc.set_ylim(lo, hi)

    ax.tick_params(axis="both", labelsize=TICK_SIZE, colors=BB["ink"], width=SPINE_WIDTH, length=4)
    ax_acc.tick_params(axis="y", labelsize=TICK_SIZE, colors=BB["red"], width=SPINE_WIDTH, length=4)
    ax.set_xlabel("")
    ax.set_ylabel("Avg NFE", fontsize=LABEL_SIZE, fontweight="bold", color=BB["ink"])
    ax_acc.set_ylabel("Accuracy", fontsize=LABEL_SIZE, fontweight="bold", color=BB["red"])
    ax.grid(axis="y", color=BB["grid"], linewidth=0.55, alpha=0.45, zorder=0)
    ax.set_title(MODEL_LABELS[model], fontsize=TITLE_SIZE + 2,
                 fontweight="bold", color=BB["ink"], pad=6)
    _bold_ticklabels(ax, ax_acc)

    return ax_acc


def build_plot_records(oracles: dict) -> list[dict]:
    """Return the exact source data used by optimal_blocksize_ablation.png.

    The figure plots confidence block-size points, Block Batching, and Oracle
    NFE. Oracle accuracy is correct when any confidence block size is correct
    for that sample.
    """
    records = []
    for model in MODELS:
        for task in TASKS:
            data = COARSE[model][task]
            for bs in BLOCK_SIZES:
                records.append({
                    "model": model,
                    "task": task,
                    "method": "confidence",
                    "block_size": bs,
                    "x_label": str(bs),
                    "avg_nfe": data[bs]["nfe"],
                    "accuracy": data[bs]["acc"],
                    "accuracy_plotted": data[bs]["acc"],
                    "accuracy_line": data[bs]["acc"],
                    "accuracy_marker": None,
                    "marker_role": None,
                    "is_bs32_accuracy_reference": bs == 32,
                })
            records.append({
                "model": model,
                "task": task,
                "method": "block_batching",
                "block_size": None,
                "x_label": "BB",
                "avg_nfe": data["bb"]["nfe"],
                "accuracy": data["bb"]["acc"],
                "accuracy_plotted": None,
                "accuracy_line": None,
                "accuracy_marker": data["bb"]["acc"],
                "marker_role": "block_batching_accuracy",
                "is_bs32_accuracy_reference": False,
            })
            opt_nfe, opt_acc = oracles[(model, task)]
            records.append({
                "model": model,
                "task": task,
                "method": "oracle_min_nfe_any_correct",
                "block_size": None,
                "x_label": "Oracle",
                "avg_nfe": opt_nfe,
                "accuracy": opt_acc,
                "accuracy_plotted": None,
                "accuracy_line": None,
                "accuracy_marker": opt_acc,
                "marker_role": "oracle_accuracy",
                "is_bs32_accuracy_reference": False,
            })
    return records


def save_plot_data(records: list[dict], out_prefix: Path) -> None:
    """Save plot source data as JSON and CSV next to the figure."""
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")

    payload = {
        "figure": "assets/paper/optimal_blocksize_ablation.png",
        "description": (
            "Source data for Optimal Block Size Ablation over GSM8K, HumanEval, MBPP, and MATH. "
            "Bars use avg_nfe. The deep-red accuracy line uses accuracy_plotted; "
            "Block Batching and Oracle accuracy are plotted as separate markers, "
            "not connected to the confidence accuracy line. Oracle NFE is each sample's "
            "minimum confidence block-size NFE; oracle accuracy is correct if any of the "
            "six confidence block-size outputs is correct for that sample."
        ),
        "block_sizes": BLOCK_SIZES,
        "records": records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fieldnames = [
        "model",
        "task",
        "method",
        "block_size",
        "x_label",
        "avg_nfe",
        "accuracy",
        "accuracy_plotted",
        "accuracy_line",
        "accuracy_marker",
        "marker_role",
        "is_bs32_accuracy_reference",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved plot data JSON: {json_path}")
    print(f"Saved plot data CSV:  {csv_path}")


def main():
    # ── Compute oracles ────────────────────────────────────────────────────────
    oracles = {}
    for model in MODELS:
        for task in TASKS:
            print(f"Computing oracle for {model}/{task}...")
            oracles[(model, task)] = compute_oracle(RESULTS_BASE, model, task)
            o = oracles[(model, task)]
            print(f"  opt_nfe={o[0]:.1f}  opt_acc={o[1]:.4f}")

    out = PROJECT_ROOT / "assets" / "paper" / "optimal_blocksize_ablation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    data_prefix = out.with_name(out.stem + "_data")
    save_plot_data(build_plot_records(oracles), data_prefix)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches
    except ModuleNotFoundError as exc:
        print(f"[skip plot] {exc}; data export completed.")
        return

    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": TICK_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": BB["ink"],
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    # Two model panels; each panel groups all four datasets.
    fig, axes = plt.subplots(1, len(MODELS), figsize=(16.8, 6.2),
                             facecolor=BB["paper"])
    fig.patch.set_facecolor(BB["paper"])

    for col, model in enumerate(MODELS):
        ax = axes[col]
        ax_acc = draw_panel(ax, model, oracles)
        if col != 0:
            ax.set_ylabel("")
        if col != len(MODELS) - 1:
            ax_acc.set_ylabel("")

    # Shared legend
    handles = [
        mpatches.Patch(color=CONF_BAR_COLORS[32], label="Confidence (bs=N)"),
        mpatches.Patch(color=BB["green"],          label="Block Batching NFE"),
        mpatches.Patch(color=BB["red"],            label="Oracle NFE"),
        mlines.Line2D([], [], color=BB["red"], linewidth=2.0, marker="o",
                      markersize=5.5,          label="Confidence accuracy"),
        mlines.Line2D([], [], color=BB["green"], linewidth=0, marker="D",
                      markeredgecolor=BB["ink"], markersize=6.0,
                      label="Block Batching accuracy"),
        mlines.Line2D([], [], color=BB["red"], linewidth=0, marker="s",
                      markeredgecolor=BB["ink"], markersize=6.0,
                      label="Oracle accuracy"),
        mlines.Line2D([], [], color=BB["vanilla"], linewidth=1.5, linestyle="--",
                      label="bs=32 accuracy"),
    ]
    fig.text(0.5, 0.115, "Block Size / Method", ha="center", va="center",
             fontsize=LABEL_SIZE, fontweight="bold", color=BB["ink"])

    legend = fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                        fontsize=max(8, LEGEND_SIZE - 1),
                        framealpha=1.0, fancybox=False, facecolor=BB["panel"],
                        edgecolor=BB["ink"], bbox_to_anchor=(0.5, 0.015),
                        borderpad=0.25, handletextpad=0.55, columnspacing=1.25)
    legend.get_frame().set_linewidth(LEGEND_FRAME_WIDTH)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.27,
                        wspace=0.32)

    fig.savefig(out, dpi=300, facecolor=BB["paper"], bbox_inches="tight", pad_inches=0.03)
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
