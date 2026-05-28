#!/usr/bin/env python3
"""Generate the MBPP overview figure from the compact benchmark table.

This plot uses only the table values for MBPP. It does not read block-combo
outputs and does not include Dream-1.5B.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_CACHE_ROOT = Path("/tmp") / f"blockbatch-matplotlib-{os.environ.get('USER', os.getuid())}"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_FONTCONFIG_CACHE = _CACHE_ROOT / "fontconfig"
_FONTCONFIG_CACHE.mkdir(parents=True, exist_ok=True)
_FONTCONFIG_FILE = _CACHE_ROOT / "fonts.conf"
_FONTCONFIG_FILE.write_text(
    '<?xml version="1.0"?>\n'
    '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
    '<fontconfig>\n'
    f'  <cachedir>{_FONTCONFIG_CACHE}</cachedir>\n'
    '</fontconfig>\n',
    encoding="utf-8",
)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("FONTCONFIG_FILE", str(_FONTCONFIG_FILE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets" / "paper"
OUT_PNG = ASSETS_DIR / "overview.png"
OUT_JSON = OUT_DIR / "overview_source_data.json"

METHODS = ["Vanilla", "Fast-dLLM", "BlockBatch"]
PLOTTED_METHODS = ["Vanilla", "Fast-dLLM", "BlockBatch"]
PLOT_CONFIG = {
    "task": "MBPP",
    "xlabel": r"NFE Reduced vs. Vanilla Baseline $\uparrow$",
    "ylabel": "MBPP Accuracy (%)",
    "figsize": (7.0, 5.45),
    "legend_anchor": (0.5, 0.018),
}
PANEL_ORDER = ["Dream-Base-7B", "LLaDA-1.5", "LLaDA-Instruct-8B"]
PLOT_LEFT = 0.145
PLOT_RIGHT = 0.99
PLOT_BOTTOM = 0.305
PLOT_TOP = 0.985
X_LABEL_Y = 0.19
Y_LABEL_Y = PLOT_BOTTOM + (PLOT_TOP - PLOT_BOTTOM) / 2

# Values are latency seconds, NFE, and accuracy percentage from the compact table.
TABLE = {
    "LLaDA-1.5": {
        "display": "LLaDA-1.5",
        "color": "#1F5B3A",
        "rows": {
            "Vanilla": (29.2, 256.0, 43.40),
            "Fast-dLLM": (4.2, 76.1, 38.00),
            "BlockBatch": (2.6, 54.5, 40.00),
        },
    },
    "LLaDA-Instruct-8B": {
        "display": "LLaDA-8B",
        "color": "#3E5C76",
        "rows": {
            "Vanilla": (9.9, 256.0, 41.40),
            "Fast-dLLM": (2.2, 68.2, 38.40),
            "BlockBatch": (1.7, 45.3, 39.60),
        },
    },
    "Dream-Base-7B": {
        "display": "Dream-7B",
        "color": "#8A4B3A",
        "rows": {
            "Vanilla": (8.0, 256.0, 55.80),
            "Fast-dLLM": (2.6, 111.9, 53.20),
            "BlockBatch": (2.3, 81.1, 52.00),
        },
    },
}

METHOD_STYLE = {
    "Vanilla": {"marker": "o", "size": 72, "label": "Vanilla"},
    "Fast-dLLM": {"marker": "s", "size": 64, "label": "Fast-dLLM"},
    "BlockBatch": {"marker": "D", "size": 104, "label": "BlockBatch"},
}

INK = "#262B2F"
GRID = "#D2D5D1"
PAPER = "#FFFFFF"
BB_EDGE = "#0B2E1D"
VANILLA_ARROW = "#C9CDD1"
FAST_ARROW = "#B23A3A"

def build_records(task: str, table: dict, methods: list[str]) -> list[dict]:
    records = []
    for model, model_data in table.items():
        vanilla = model_data["rows"]["Vanilla"]
        assert vanilla is not None
        vanilla_latency, vanilla_nfe, vanilla_acc = vanilla
        for method in methods:
            value = model_data["rows"].get(method)
            if value is None:
                records.append(
                    {
                        "task": task,
                        "model": model,
                        "model_label": model_data["display"],
                        "method": method,
                        "available": False,
                        "plotted": False,
                        "reason": "not available in compact table",
                    }
                )
                continue
            latency, nfe, acc = value
            records.append(
                {
                    "task": task,
                    "model": model,
                    "model_label": model_data["display"],
                    "method": method,
                    "available": True,
                    "plotted": method in PLOTTED_METHODS,
                    "latency_s": latency,
                    "nfe": nfe,
                    "accuracy_pct": acc,
                    "vanilla_latency_s": vanilla_latency,
                    "vanilla_nfe": vanilla_nfe,
                    "vanilla_accuracy_pct": vanilla_acc,
                    "latency_speedup": vanilla_latency / latency,
                    "nfe_reduced": vanilla_nfe - nfe,
                    "nfe_reduction_pct": 100.0 * (1.0 - nfe / vanilla_nfe),
                    "accuracy_delta_pp": acc - vanilla_acc,
                }
            )
    return records


def write_source(records: list[dict], config: dict) -> None:
    payload = {
        "source": "Compact benchmark table in paper draft",
        "task": config["task"],
        "excluded": ["block-combo runs", "Dream-1.5B"],
        "x_axis": "nfe_reduced = vanilla_nfe - method_nfe",
        "figure_layout": "Short 7.0 x 5.45 inch double-column layout; y-axis label centered on the plotted panel region.",
        "speedup_basis": "NFE ratio; arrow label = source NFE / BlockBatch NFE",
        "notes": [
            "The plot uses raw NFE and accuracy values.",
            "The overview is rendered as three vertically stacked y-axis segments, one per model.",
            "Vanilla, Fast-dLLM, and BlockBatch are plotted for MBPP.",
            "Gray arrows show Vanilla-to-BlockBatch NFE speedup.",
            "Red arrows show Fast-dLLM-to-BlockBatch NFE speedup.",
        ],
        "records": records,
        "speedup_annotations": build_speedup_annotations(records),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def setup_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.linewidth": 2.4,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "legend.fontsize": 10.8,
            "legend.edgecolor": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def plotted_records(records: list[dict]) -> list[dict]:
    return [row for row in records if row.get("available") and row.get("plotted")]


def model_records(records: list[dict], model: str) -> list[dict]:
    return [row for row in records if row.get("model") == model]


def x_value(row: dict, config: dict) -> float:
    return row["nfe_reduced"]


def scatter_method_points(ax, records: list[dict], table: dict) -> None:
    for row in plotted_records(records):
        model_data = table[row["model"]]
        style = METHOD_STYLE[row["method"]]
        is_blockbatch = row["method"] == "BlockBatch"
        ax.scatter(
            x_value(row, PLOT_CONFIG),
            row["accuracy_pct"],
            s=style["size"],
            marker=style["marker"],
            facecolors=model_data["color"],
            edgecolors=BB_EDGE if is_blockbatch else INK,
            linewidths=1.9 if is_blockbatch else 1.25,
            alpha=0.96 if is_blockbatch else 0.74,
            zorder=5 if is_blockbatch else 3,
        )


def method_record(records: list[dict], model: str, method: str) -> dict | None:
    for row in records:
        if row["model"] == model and row["method"] == method and row.get("available"):
            return row
    return None


def nfe_speedup(source: dict, blockbatch: dict) -> float:
    return source["nfe"] / blockbatch["nfe"]


def speedup_label(value: float) -> str:
    return f"{value:.1f}x"


def arrow_midpoint(source: dict, target: dict, fraction: float = 0.5) -> tuple[float, float]:
    source_x = x_value(source, PLOT_CONFIG)
    target_x = x_value(target, PLOT_CONFIG)
    source_y = source["accuracy_pct"]
    target_y = target["accuracy_pct"]
    return (
        source_x + (target_x - source_x) * fraction,
        source_y + (target_y - source_y) * fraction,
    )


def build_speedup_annotations(records: list[dict]) -> list[dict]:
    annotations = []
    models = list(dict.fromkeys(row["model"] for row in records if row.get("available")))
    for model in models:
        vanilla = method_record(records, model, "Vanilla")
        fast = method_record(records, model, "Fast-dLLM")
        blockbatch = method_record(records, model, "BlockBatch")
        if vanilla is None or fast is None or blockbatch is None:
            continue
        vanilla_to_bb = nfe_speedup(vanilla, blockbatch)
        fast_to_bb = nfe_speedup(fast, blockbatch)
        annotations.append(
            {
                "model": model,
                "model_label": blockbatch["model_label"],
                "basis": "NFE",
                "vanilla_to_blockbatch": {
                    "source_nfe": vanilla["nfe"],
                    "target_nfe": blockbatch["nfe"],
                    "speedup": vanilla_to_bb,
                    "label": speedup_label(vanilla_to_bb),
                },
                "fastdllm_to_blockbatch": {
                    "source_nfe": fast["nfe"],
                    "target_nfe": blockbatch["nfe"],
                    "speedup": fast_to_bb,
                    "label": speedup_label(fast_to_bb),
                },
            }
        )
    return annotations


def draw_blockbatch_arrows(ax, records: list[dict], table: dict) -> None:
    for model, model_data in table.items():
        vanilla = method_record(records, model, "Vanilla")
        fast = method_record(records, model, "Fast-dLLM")
        blockbatch = method_record(records, model, "BlockBatch")
        if vanilla is None or fast is None or blockbatch is None:
            continue

        ax.annotate(
            "",
            xy=(x_value(blockbatch, PLOT_CONFIG), blockbatch["accuracy_pct"]),
            xytext=(x_value(vanilla, PLOT_CONFIG), vanilla["accuracy_pct"]),
            arrowprops={
                "arrowstyle": "-|>",
                "linestyle": "-",
                "linewidth": 1.9,
                "color": VANILLA_ARROW,
                "mutation_scale": 11,
                "shrinkA": 7,
                "shrinkB": 10,
            },
            zorder=1,
        )
        ax.annotate(
            "",
            xy=(x_value(blockbatch, PLOT_CONFIG), blockbatch["accuracy_pct"]),
            xytext=(x_value(fast, PLOT_CONFIG), fast["accuracy_pct"]),
            arrowprops={
                "arrowstyle": "-|>",
                "linestyle": "-",
                "linewidth": 2.0,
                "color": FAST_ARROW,
                "mutation_scale": 11,
                "shrinkA": 8,
                "shrinkB": 9,
            },
            zorder=2,
        )

        vanilla_to_bb = nfe_speedup(vanilla, blockbatch)
        fast_to_bb = nfe_speedup(fast, blockbatch)
        gray_x, gray_y = arrow_midpoint(vanilla, blockbatch, 0.52)
        ax.text(
            gray_x,
            gray_y,
            speedup_label(vanilla_to_bb),
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=FAST_ARROW,
            bbox={
                "boxstyle": "square,pad=0.2",
                "facecolor": PAPER,
                "edgecolor": FAST_ARROW,
                "linewidth": 1.1,
                "alpha": 0.96,
            },
            zorder=6,
        )
        red_x, red_y = arrow_midpoint(fast, blockbatch, 0.50)
        ax.text(
            red_x,
            red_y,
            speedup_label(fast_to_bb),
            ha="center",
            va="center",
            fontsize=9.3,
            fontweight="bold",
            color=FAST_ARROW,
            bbox={
                "boxstyle": "square,pad=0.18",
                "facecolor": PAPER,
                "edgecolor": FAST_ARROW,
                "linewidth": 1.1,
                "alpha": 0.96,
            },
            zorder=7,
        )


def panel_limits(rows: list[dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = [row for row in plotted_records(rows)]
    xs = [x_value(row, PLOT_CONFIG) for row in points]
    ys = [row["accuracy_pct"] for row in points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    x_pad = max(8.0, 0.08 * x_span)
    y_pad = max(0.42, 0.09 * y_span)
    return (
        (max(0.0, min(xs) - x_pad), max(xs) + x_pad),
        (min(ys) - y_pad, max(ys) + y_pad),
    )


def common_xlim(records: list[dict]) -> tuple[float, float]:
    points = [row for row in plotted_records(records)]
    xs = [x_value(row, PLOT_CONFIG) for row in points]
    x_span = max(xs) - min(xs)
    x_pad = max(6.0, 0.045 * x_span)
    return (min(xs) - x_pad, max(xs) + x_pad)


def style_axes(ax, rows: list[dict], model_data: dict, config: dict, xlim: tuple[float, float]) -> None:
    _, ylim = panel_limits(rows)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(axis="both", color=GRID, lw=0.5, alpha=0.72, zorder=0)
    ax.text(
        0.075,
        0.92,
        model_data["display"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=model_data["color"],
        bbox={
            "boxstyle": "square,pad=0.16",
            "facecolor": PAPER,
            "edgecolor": model_data["color"],
            "linewidth": 1.0,
            "alpha": 0.94,
        },
        zorder=8,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(2.4)
        spine.set_color(INK)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def method_legend_handles() -> list:
    return [
        mlines.Line2D(
            [],
            [],
            linestyle="None",
            marker=METHOD_STYLE[method]["marker"],
            markerfacecolor="#D7D9D4",
            markeredgecolor=INK,
            markeredgewidth=1.45,
            markersize=8.8 if method != "BlockBatch" else 9.6,
            label=METHOD_STYLE[method]["label"],
        )
        for method in PLOTTED_METHODS
    ]


def model_legend_handles(table: dict) -> list:
    return [
        mlines.Line2D([], [], color=data["color"], lw=2.4, ls="-", label=data["display"])
        for data in table.values()
    ]


def arrow_legend_handles() -> list:
    return [
        mlines.Line2D([], [], color=VANILLA_ARROW, lw=2.0, label="Vanilla to BlockBatch"),
        mlines.Line2D([], [], color=FAST_ARROW, lw=2.0, label="Fast-dLLM to BlockBatch"),
    ]


def add_legend(fig, config: dict) -> None:
    method_handles = [
        *method_legend_handles(),
        *arrow_legend_handles(),
    ]
    legend = fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=config["legend_anchor"],
        frameon=True,
        facecolor=PAPER,
        edgecolor=INK,
        framealpha=1.0,
        ncol=2,
        borderpad=0.58,
        handletextpad=0.50,
        columnspacing=1.05,
    )
    legend.get_frame().set_linewidth(2.2)
    for text in legend.get_texts():
        text.set_fontweight("bold")
        if text.get_text() == "BlockBatch":
            text.set_color(BB_EDGE)


def plot(records: list[dict], table: dict, config: dict) -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()
    fig, axes = plt.subplots(
        len(PANEL_ORDER),
        1,
        figsize=config["figsize"],
        facecolor=PAPER,
        sharex=True,
        gridspec_kw={"hspace": 0.0},
    )
    fig.patch.set_facecolor(PAPER)
    xlim = common_xlim(records)

    for idx, (ax, model) in enumerate(zip(axes, PANEL_ORDER)):
        model_data = table[model]
        rows = model_records(records, model)
        ax.set_facecolor(PAPER)
        scatter_method_points(ax, rows, table)
        draw_blockbatch_arrows(ax, rows, {model: model_data})
        style_axes(ax, rows, model_data, config, xlim)
        if idx < len(PANEL_ORDER) - 1:
            ax.tick_params(axis="x", labelbottom=False)

    fig.supylabel(config["ylabel"], fontsize=14, fontweight="bold", x=0.018, y=Y_LABEL_Y)
    fig.supxlabel(config["xlabel"], fontsize=14, fontweight="bold", y=X_LABEL_Y)
    add_legend(fig, config)
    fig.subplots_adjust(left=PLOT_LEFT, right=PLOT_RIGHT, bottom=PLOT_BOTTOM, top=PLOT_TOP, hspace=0.0)
    fig.savefig(OUT_PNG, dpi=300, facecolor=PAPER)
    plt.close(fig)


def main() -> None:
    records = build_records(PLOT_CONFIG["task"], TABLE, METHODS)
    write_source(records, PLOT_CONFIG)
    plot(records, TABLE, PLOT_CONFIG)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
