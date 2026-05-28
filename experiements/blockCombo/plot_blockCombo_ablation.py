#!/usr/bin/env python3
"""Render the block-size-combination ablation figure."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines  as mlines
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = Path(__file__).parent / "results" / "block_combo_grid_data.json"
OUT_FILE = PROJECT_ROOT / "assets" / "ablations" / "block_combo" / "block_combo_ablation.pdf"
RESULTS_DIR = Path(__file__).parent / "results"

INK      = "#2B2F33"
GRID_COL = "#C9C7BD"
PAPER    = "#FFFFFF"
VANILLA  = "#8A8C7A"
FAST     = "#3F7FA6"
PARETO   = "#B64B4A"
PARETO_DK= "#7A2F2F"

COMBO_COLORS = {
    2: "#6BAE8C",
    3: "#4A9470",
    4: "#317A57",
    5: "#1F6342",
    6: "#1F5F49",
}

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          12,
    "axes.labelsize":     14,
    "axes.titlesize":     14,
    "axes.labelweight":   "bold",
    "axes.linewidth":     1.8,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "xtick.major.width":  1.4,
    "ytick.major.width":  1.4,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "legend.fontsize":    11,
    "legend.edgecolor":   INK,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "text.color":         INK,
    "axes.labelcolor":    INK,
    "xtick.color":        INK,
    "ytick.color":        INK,
})


def _pareto(points):
    """Non-dominated front: min NFE, max acc."""
    pts = sorted(points, key=lambda t: t[0])
    front, best = [], -math.inf
    for nfe, acc, lbl in pts:
        if acc > best:
            front.append((nfe, acc, lbl))
            best = acc
    return front


def _best_combo(pareto_pts):
    """Return the single best combo: highest accuracy; tie-break by lowest NFE."""
    return max(pareto_pts, key=lambda t: (t[1], -t[0]))


def draw_panel(ax, records, baselines, panel_label):
    """Draw one scatter panel."""
    plotted = [r for r in records if r["is_plotted"]]
    if not plotted:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=GRID_COL, fontsize=12)
        return

    nfes = [r["avg_nfe"] for r in plotted]
    accs = [r["accuracy"] for r in plotted]
    nfe_pad = (max(nfes) - min(nfes)) * 0.08 or 5
    acc_pad = (max(accs) - min(accs)) * 0.08 or 0.01
    ax.set_xlim(min(nfes) - nfe_pad, max(nfes) + nfe_pad)
    ax.set_ylim(min(accs) - acc_pad, max(accs) + acc_pad * 3)

    for r in plotted:
        color = COMBO_COLORS.get(r["num_block_sizes"], "#4A9470")
        ax.scatter(r["avg_nfe"], r["accuracy"],
                   s=20, color=color, alpha=0.60,
                   edgecolors="none", zorder=2)

    pareto_recs = [r for r in plotted if r["is_pareto"]]
    if pareto_recs:
        pts = sorted([(r["avg_nfe"], r["accuracy"], r["label"])
                      for r in pareto_recs], key=lambda t: t[0])
        px = [t[0] for t in pts]
        py = [t[1] for t in pts]
        ax.plot(px, py, color=PARETO, linewidth=1.8, zorder=4, alpha=0.9)
        ax.scatter(px, py, s=40, color=PARETO, zorder=5,
                   edgecolors=PARETO_DK, linewidths=0.8)

        bx, by, blbl = _best_combo(pts)
        ax.annotate(
            blbl, (bx, by),
            textcoords="offset points", xytext=(6, 6),
            fontsize=9, color=PARETO_DK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc=PAPER,
                      ec=PARETO, lw=0.8, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=PARETO, lw=0.8),
        )

    v  = baselines.get("vanilla",  {})
    fd = baselines.get("fast_dllm",{})
    if v.get("acc") is not None:
        ax.axhline(v["acc"], color=VANILLA, lw=1.3, ls="--", alpha=0.85, zorder=1)
    if fd.get("acc") is not None:
        ax.axhline(fd["acc"], color=FAST, lw=1.3, ls="--", alpha=0.85, zorder=1)

    ax.set_title(panel_label, fontsize=13, fontweight="bold",
                 color=INK, pad=6, loc="left")

    ax.set_facecolor(PAPER)
    ax.set_xlabel("Average NFE",  fontsize=14, fontweight="bold", color=INK)
    ax.set_ylabel("Accuracy",     fontsize=14, fontweight="bold", color=INK)
    ax.tick_params(labelsize=12, colors=INK, width=1.4)
    ax.grid(axis="both", color=GRID_COL, lw=0.4, alpha=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color(INK)


def main():
    data = json.loads(DATA_FILE.read_text())
    records   = data["records"]
    baselines = data["baselines"]

    PANELS = [
        ("llada", "humaneval", "LLaDA — HumanEval"),
        ("llada", "mbpp",      "LLaDA — MBPP"),
        ("dream", "humaneval", "Dream — HumanEval"),
        ("dream", "mbpp",      "Dream — MBPP"),
    ]
    DREAM_ACC_UNRELIABLE = True

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.8), facecolor=PAPER)
    fig.patch.set_facecolor(PAPER)

    for ax, (model, task, label) in zip(axes.flat, PANELS):
        recs = [r for r in records if r["model"]==model and r["task"]==task]
        if DREAM_ACC_UNRELIABLE and model == "dream":
            import glob
            recs_fixed = []
            for r in recs:
                res_files = glob.glob(
                    f"{RESULTS_DIR}/{model}/{task}/{r['variant']}/lm_eval/*/results_*.json")
                if res_files:
                    d = json.load(open(res_files[0])).get("results", {})
                    task_res = d.get(task, d.get("humaneval", d.get("mbpp", {})))
                    acc = None
                    for k, v in task_res.items():
                        if "pass" in k.lower() and "stderr" not in k and isinstance(v, float):
                            acc = v; break
                    if acc is not None and acc > 0.01:
                        r = dict(r); r["accuracy"] = acc; r["is_plotted"] = True
                        recs_fixed.append(r)
                else:
                    if r.get("accuracy") and r["accuracy"] > 0.01:
                        recs_fixed.append(r)
            recs = recs_fixed
        bl = baselines.get(model, {}).get(task, {})
        draw_panel(ax, recs, bl, label)

    legend_handles = [
        mpatches.Patch(color=COMBO_COLORS[s],
                       label=f"{s} block sizes")
        for s in [2, 3, 4, 5, 6]
    ] + [
        mlines.Line2D([], [], color=PARETO,  lw=2.0, marker="o", ms=5,
                      label="Pareto frontier"),
        mlines.Line2D([], [], color=VANILLA, lw=1.4, ls="--",
                      label="Vanilla"),
        mlines.Line2D([], [], color=FAST,    lw=1.4, ls="--",
                      label="Fast-dLLM"),
    ]
    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        fontsize=11,
        framealpha=1.0,
        facecolor=PAPER,
        edgecolor=INK,
        bbox_to_anchor=(0.5, 0.0),
    )
    leg.get_frame().set_linewidth(2.0)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.12, 1, 1], h_pad=3.5, w_pad=2.5)
    fig.savefig(OUT_FILE,  dpi=300, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(OUT_FILE.with_suffix(".png"), dpi=180,
                bbox_inches="tight", facecolor=PAPER)
    print(f"Saved: {OUT_FILE}")
    print(f"Saved: {OUT_FILE.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
