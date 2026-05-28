#!/usr/bin/env python3
"""Build source data and plots for block-size-combination ablations."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "blockBatching_ablation"))
import analyze as analyze_mod  # noqa: E402
from analysis_utils import accuracy_rows, json_count, json_number  # noqa: E402
from block_combo_ablation import ALL_COMBOS, combo_label  # noqa: E402

BB = {
    "ink":     "#2B2F33",
    "grid":    "#C9C7BD",
    "paper":   "#FFFFFF",
    "panel":   "#FFFFFF",
    "vanilla": "#8A8C7A",
    "fast":    "#3F7FA6",
    "green":   "#2F8F6B",
    "green_dk":"#1F5F49",
    "green_lt":"#DDEFE6",
    "frontier":"#5AA6D6",
    "frontier_dk":"#2F6F9F",
    "slate":   "#6F7F96",
}

COMBO_SIZE_COLORS = {2: "#BBDCA8", 3: "#91C782", 4: "#64AD63", 5: "#398B4D", 6: "#1F6336"}

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR    = EXPERIMENT_DIR / "results"
BASELINES: dict = {}

PANELS = [
    ("llada", "humaneval"), ("llada", "mbpp"),
    ("dream", "humaneval"), ("dream", "mbpp"),
]
PANEL_TITLES = {
    ("llada", "humaneval"): "LLaDA — HumanEval",
    ("llada", "mbpp"):      "LLaDA — MBPP",
    ("dream", "humaneval"): "Dream — HumanEval",
    ("dream", "mbpp"):      "Dream — MBPP",
}


def _variant_label(combo_name: str) -> str:
    return combo_name.replace("block_combo_bs", "")


def _parse_combo_label(label: str) -> list[int]:
    return [int(part) for part in str(label).split("-") if part]


def _combo_sort_key(label: str) -> tuple[int, list[int]]:
    block_sizes = _parse_combo_label(label)
    return (len(block_sizes), block_sizes)


def _pareto_frontier(points):
    sorted_pts = sorted(points, key=lambda t: t[0])
    frontier = []
    best_acc = -math.inf
    for nfe, acc, label in sorted_pts:
        if acc > best_acc:
            frontier.append((nfe, acc, label))
            best_acc = acc
    return frontier


def load_combo_records(results_dir: Path, model: str, task: str):
    """Return one JSON-ready row per expected combo for a model/task panel."""
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    variant_glob = "block_combo_*"

    records = analyze_mod.load_records(
        str(results_dir), model, task, variant=variant_glob)

    stats_by_label = {}
    accu_by_label = {}
    stats_rows = analyze_mod.compute_stats(records)
    for bl, n_samples, record_acc, avg_nfe, avg_tok, avg_lat, avg_tps in stats_rows:
        label = str(bl)
        stats_by_label[label] = {
            "n_samples": json_count(n_samples),
            "record_accuracy": json_number(record_acc),
            "avg_nfe": json_number(avg_nfe),
            "avg_tokens": json_number(avg_tok),
            "avg_latency_s": json_number(avg_lat),
            "avg_tps": json_number(avg_tps),
        }

    if records:
        for bl, n_accuracy_samples, accuracy in accuracy_rows(
            results_dir, model, task, variant_glob, label_fn=_variant_label
        ):
            accu_by_label[str(bl)] = {
                "n_accuracy_samples": json_count(n_accuracy_samples),
                "accuracy": json_number(accuracy),
            }

    expected_labels = [combo_label(combo) for combo in ALL_COMBOS]
    observed_labels = set(stats_by_label) | set(accu_by_label)
    labels = sorted(set(expected_labels) | observed_labels, key=_combo_sort_key)

    rows = []
    for label in labels:
        block_sizes = _parse_combo_label(label)
        stats = stats_by_label.get(label, {})
        accu = accu_by_label.get(label, {})
        avg_nfe = stats.get("avg_nfe")
        accuracy = accu.get("accuracy", stats.get("record_accuracy"))
        n_samples = stats.get("n_samples")
        n_accuracy_samples = accu.get("n_accuracy_samples")
        rows.append({
            "model": model,
            "task": task,
            "method": "block_combo",
            "variant": f"block_combo_bs{label}",
            "label": label,
            "block_sizes": block_sizes,
            "num_block_sizes": len(block_sizes),
            "n_samples": n_samples,
            "n_accuracy_samples": n_accuracy_samples,
            "avg_nfe": avg_nfe,
            "accuracy": accuracy,
            "record_accuracy": stats.get("record_accuracy"),
            "avg_tokens": stats.get("avg_tokens"),
            "avg_latency_s": stats.get("avg_latency_s"),
            "avg_tps": stats.get("avg_tps"),
            "has_records": n_samples is not None,
            "has_accuracy": accuracy is not None,
            "is_expected_combo": label in expected_labels,
            "is_plotted": avg_nfe is not None and accuracy is not None,
            "is_pareto": False,
        })

    frontier_labels = {
        label
        for _nfe, _acc, label in _pareto_frontier([
            (row["avg_nfe"], row["accuracy"], row["label"])
            for row in rows
            if row["is_plotted"]
        ])
    }
    for row in rows:
        row["is_pareto"] = row["label"] in frontier_labels

    return rows


def build_panel_records():
    return {
        (model, task): load_combo_records(RESULTS_DIR, model, task)
        for model, task in PANELS
    }


def save_plot_data(panel_records: dict, baselines: dict, out_path: Path):
    json_path = out_path.with_name(out_path.stem + "_data.json")
    records = []
    panels = []
    for model, task in PANELS:
        panel_rows = panel_records[(model, task)]
        plotted = [row for row in panel_rows if row["is_plotted"]]
        panels.append({
            "model": model,
            "task": task,
            "title": PANEL_TITLES[(model, task)],
            "expected_configurations": len(ALL_COMBOS),
            "recorded_configurations": sum(1 for row in panel_rows if row["has_records"]),
            "plotted_configurations": len(plotted),
            "pareto_configurations": [row["label"] for row in panel_rows if row["is_pareto"]],
        })
        records.extend(panel_rows)

    table = {
        model: {
            task: panel_records[(model, task)]
            for _model, task in PANELS
            if _model == model
        }
        for model in sorted({model for model, _task in PANELS})
    }

    payload = {
        "figure": out_path.name,
        "description": (
            "Source data for block_combo_grid.png. Each block_combo record is "
            "one block-size configuration for a model/task panel. avg_nfe is "
            "computed from rank JSONL records; accuracy is computed with the "
            "same task-specific post-processing used by blockBatching_ablation/analyze.py."
        ),
        "results_dir": str(RESULTS_DIR),
        "base_block_sizes": sorted({size for combo in ALL_COMBOS for size in combo}),
        "total_expected_configurations_per_panel": len(ALL_COMBOS),
        "panels": panels,
        "table": table,
        "records": records,
        "baselines": baselines,
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved plot data JSON: {json_path}")


def draw_panel(ax, model: str, task: str, baselines: dict, rows: list[dict]):
    points = [
        (row["avg_nfe"], row["accuracy"], row["label"], row["num_block_sizes"])
        for row in rows
        if row["is_plotted"]
    ]

    ax.set_facecolor(BB["paper"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BB["grid"])
    ax.spines["bottom"].set_color(BB["grid"])
    ax.tick_params(colors=BB["ink"], labelsize=7)
    ax.grid(axis="both", color=BB["grid"], linewidth=0.3, alpha=0.6)
    ax.set_title(PANEL_TITLES[(model, task)], fontsize=8.5,
                 color=BB["ink"], pad=4)
    ax.set_xlabel("Avg NFE", fontsize=8, color=BB["ink"])
    ax.set_ylabel("Accuracy", fontsize=8, color=BB["ink"])

    if not points:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=BB["slate"], fontsize=8)
        return

    for nfe, acc, label, n_sizes in points:
        color = COMBO_SIZE_COLORS.get(n_sizes, BB["slate"])
        ax.scatter(nfe, acc, s=22, color=color, alpha=0.75,
                   edgecolors="none", zorder=2)

    frontier = [
        (row["avg_nfe"], row["accuracy"], row["label"])
        for row in rows
        if row["is_pareto"]
    ]
    frontier.sort(key=lambda item: item[0])
    if frontier:
        fx = [t[0] for t in frontier]
        fy = [t[1] for t in frontier]
        ax.plot(fx, fy, color=BB["frontier_dk"], linewidth=1.4, alpha=0.9, zorder=4)
        ax.scatter(fx, fy, s=40, color=BB["frontier"], edgecolors=BB["ink"],
                   linewidths=0.4, zorder=5)
        for i, (nfe, acc, label) in enumerate(frontier):
            yoff = 5 if i % 2 == 0 else -10
            xoff = 3
            ax.annotate(label, (nfe, acc),
                        textcoords="offset points", xytext=(xoff, yoff),
                        fontsize=5.0, color=BB["frontier_dk"], fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", fc=BB["paper"],
                                  ec=BB["frontier_dk"], linewidth=0.3, alpha=0.88),
                        arrowprops=dict(arrowstyle="-", color=BB["frontier_dk"],
                                        lw=0.4, alpha=0.6))

    bl = baselines.get(model, {}).get(task, {})
    v  = bl.get("vanilla", {})
    fd = bl.get("fast_dllm", {})
    ymin, ymax = ax.get_ylim()
    if v.get("acc") is not None:
        ax.axhline(v["acc"], color=BB["vanilla"], linewidth=0.9,
                   linestyle="--", alpha=0.8, zorder=1,
                   label=f"Vanilla acc={v['acc']:.3f}")
        ax.axvline(v["nfe"], color=BB["vanilla"], linewidth=0.6,
                   linestyle=":", alpha=0.5, zorder=1)
    if fd.get("acc") is not None:
        ax.axhline(fd["acc"], color=BB["fast"], linewidth=0.9,
                   linestyle="--", alpha=0.8, zorder=1,
                   label=f"Fast-dLLM acc={fd['acc']:.3f}")
        ax.axvline(fd["nfe"], color=BB["fast"], linewidth=0.6,
                   linestyle=":", alpha=0.5, zorder=1)


def main():
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": BB["ink"], "axes.labelcolor": BB["ink"],
        "xtick.color": BB["ink"], "ytick.color": BB["ink"],
    })

    baselines = BASELINES
    panel_records = build_panel_records()
    out = PROJECT_ROOT / "assets" / "ablations" / "block_combo" / "block_combo_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_plot_data(panel_records, baselines, out)

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.5),
                             facecolor=BB["paper"])
    fig.patch.set_facecolor(BB["paper"])

    for ax, (model, task) in zip(axes.flat, PANELS):
        draw_panel(ax, model, task, baselines, panel_records[(model, task)])

    legend_handles = [
        mpatches.Patch(color=COMBO_SIZE_COLORS[2], label="2-size combo"),
        mpatches.Patch(color=COMBO_SIZE_COLORS[3], label="3-size combo"),
        mpatches.Patch(color=COMBO_SIZE_COLORS[4], label="4-size combo"),
        mpatches.Patch(color=COMBO_SIZE_COLORS[5], label="5-size combo"),
        mpatches.Patch(color=COMBO_SIZE_COLORS[6], label="6-size combo"),
        mlines.Line2D([], [], color=BB["frontier_dk"], linewidth=1.4, label="Pareto frontier"),
        mlines.Line2D([], [], color=BB["vanilla"], linewidth=0.9, linestyle="--", label="Vanilla"),
        mlines.Line2D([], [], color=BB["fast"],    linewidth=0.9, linestyle="--", label="Fast-dLLM"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=7, framealpha=0.95,
               facecolor=BB["panel"], edgecolor=BB["grid"],
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Block-Size Combo Ablation — NFE vs Accuracy",
                 fontsize=9.5, color=BB["ink"], y=1.01)
    fig.tight_layout(rect=[0, 0.08, 1, 1], h_pad=2.5, w_pad=2.0)

    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BB["paper"])
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
