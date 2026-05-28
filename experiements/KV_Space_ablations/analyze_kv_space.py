"""Offline analysis for KV-space ablation logs."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

FULL_EVENTS = {"full_sequence", "full_refresh"}
BLOCK_EVENTS = {"block_denoise"}
POLICY_EVENTS = {"policy_pre", "policy_post"}

# ── Block Batching paper palette (Nature-style muted) ─────────────────────────
BB = {
    "ink":    "#2B2F33",
    "grid":   "#C9C7BD",
    "paper":  "#FFFFFF",
    "panel":  "#FFFFFF",
    "sync":   "#7A5C9E",   # BBPurple — merge/sync arrows
    "pareto": "#B64B4A",   # BBRed   — Pareto frontier
    "teal":   "#2C9C9C",   # BBTeal  — secondary trajectory
    "slate":  "#6F7F96",   # BBSlate — misc
}

# Per-branch paired colors: denoise is light, refresh/final is the darker
# shade of the same hue. Colors are muted and colorblind-conscious.
BLOCK_SIZE_COLOR_PAIRS = {
    4:   {"denoise": "#A6CEE3", "refresh": "#1F78B4"},  # blue
    8:   {"denoise": "#B2DF8A", "refresh": "#33A02C"},  # green
    16:  {"denoise": "#FDBF6F", "refresh": "#FF7F00"},  # orange
    32:  {"denoise": "#CAB2D6", "refresh": "#6A3D9A"},  # purple
    64:  {"denoise": "#FB9A99", "refresh": "#E31A1C"},  # red
    128: {"denoise": "#E6C8A8", "refresh": "#8C510A"},  # brown
}
_BS_DEFAULT_COLOR = "#8A8C7A"


def _bs_color(block_size: int, tone: str = "denoise") -> str:
    """Return the branch color for a block size and tone."""
    return BLOCK_SIZE_COLOR_PAIRS.get(int(block_size), {}).get(tone, _BS_DEFAULT_COLOR)


def _event_color(event_type: str, block_size: int) -> str:
    """
    Full refresh/final events use the dark branch color; block denoise uses
    the light branch color for the same block size.
    """
    if event_type in FULL_EVENTS or event_type == "final":
        return _bs_color(block_size, "refresh")
    if event_type in BLOCK_EVENTS:
        return _bs_color(block_size, "denoise")
    return BB["slate"]


def _load_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _is_origin_plot_point(point: dict) -> bool:
    if point["event_type"] != "full_sequence":
        return False
    extra = point.get("extra") or {}
    return int(extra.get("num_block", 0)) == 0


def _is_initial_point(point: dict) -> bool:
    if point["event_type"] != "full_sequence":
        return False
    extra = point.get("extra") or {}
    return int(extra.get("num_block", 0)) == 0


def _vector_from_payload(event_payload: dict, raw_payload: Optional[dict], layer_set: str,
                         branch_offset: int, vector_source: str) -> Optional[torch.Tensor]:
    if vector_source == "raw":
        if raw_payload is None or layer_set not in raw_payload:
            return None
        return raw_payload[layer_set][branch_offset].to(torch.float32)
    layer_payload = event_payload["layer_sets"].get(layer_set)
    if layer_payload is None:
        return None
    return layer_payload["sketch"][branch_offset].to(torch.float32)


def load_points(run_dir: Path, vector_source: str = "sketch",
                max_events: Optional[int] = None) -> List[dict]:
    points: List[dict] = []
    for event_count, index_record in enumerate(_load_jsonl(run_dir / "events.jsonl")):
        if max_events is not None and event_count >= max_events:
            break
        event_payload = torch.load(run_dir / index_record["event_path"], map_location="cpu")
        raw_payload = None
        if vector_source == "raw" and index_record.get("raw_path"):
            raw_payload = torch.load(run_dir / index_record["raw_path"], map_location="cpu")

        for layer_set in event_payload["layer_sets"].keys():
            for branch_offset, branch in enumerate(event_payload["branches"]):
                vector = _vector_from_payload(event_payload, raw_payload, layer_set, branch_offset, vector_source)
                if vector is None:
                    continue
                points.append({
                    "sample_id": event_payload["sample_id"],
                    "method": event_payload["method"],
                    "event_index": int(event_payload["event_index"]),
                    "event_type": event_payload["event_type"],
                    "event_step": int(event_payload["event_step"]),
                    "nfe": event_payload.get("nfe"),
                    "layer_set": layer_set,
                    "branch_idx": int(branch["branch_idx"]),
                    "block_size": int(branch["block_size"]),
                    "extra": event_payload.get("extra") or {},
                    "vector": vector.reshape(-1),
                })
    return points


def load_points_tree(path: Path, vector_source: str = "sketch",
                     max_events_per_run: Optional[int] = None) -> List[dict]:
    if path is None or not path.exists():
        return []
    if (path / "events.jsonl").exists():
        return load_points(path, vector_source=vector_source, max_events=max_events_per_run)
    points: List[dict] = []
    for child_index in sorted(path.glob("*/events.jsonl")):
        points.extend(load_points(
            child_index.parent,
            vector_source=vector_source,
            max_events=max_events_per_run,
        ))
    return points


def _angle(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) <= 0.0:
        return 0.0
    cos = torch.clamp(torch.dot(a, b) / denom, -1.0, 1.0)
    return float(torch.arccos(cos).item())


def _fit_group(points: List[dict]) -> Dict:
    # Drop any points whose vector dim doesn't match the majority — handles
    # mixed-dim runs that were appended to the same events.jsonl.
    from collections import Counter
    dim_counts = Counter(p["vector"].numel() for p in points)
    dominant_dim = dim_counts.most_common(1)[0][0]
    points = [p for p in points if p["vector"].numel() == dominant_dim]

    initial = [p["vector"] for p in points if _is_initial_point(p)]
    if not initial:
        initial = [min(points, key=lambda p: p["event_index"])["vector"]]
    c0 = torch.stack(initial, dim=0).mean(dim=0)
    c0_norm = torch.linalg.vector_norm(c0).clamp_min(1e-12)
    u0 = c0 / c0_norm

    residuals = []
    enriched = []
    norms = []
    for point in points:
        c = point["vector"]
        delta = c - c0
        z = torch.dot(delta, u0)
        r = delta - z * u0
        residuals.append(r)
        norm = torch.linalg.vector_norm(c)
        r_norm = torch.linalg.vector_norm(r)
        denom = (c0_norm + z).abs().clamp_min(1e-12)
        enriched_point = dict(point)
        enriched_point.pop("vector")
        enriched_point.update({
            "kv_norm": float(norm.item()),
            "axial_z": float(z.item()),
            "tangent_norm": float(r_norm.item()),
            "cone_ratio": float((r_norm / denom).item()),
            "angular_drift": _angle(c, c0),
            "_residual": r,
            "_vector": c,
        })
        norms.append(norm)
        enriched.append(enriched_point)

    R = torch.stack(residuals, dim=0)
    centered = R - R.mean(dim=0, keepdim=True)
    if centered.shape[0] >= 2 and centered.abs().sum() > 0:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        e1 = vh[0]
        e2 = vh[1] if vh.shape[0] > 1 else torch.zeros_like(e1)
    else:
        e1 = torch.zeros_like(c0)
        e1[0] = 1.0
        e2 = torch.zeros_like(c0)
        if e2.numel() > 1:
            e2[1] = 1.0

    for point in enriched:
        r = point.pop("_residual")
        point.pop("_vector")
        point["x"] = float(torch.dot(r, e1).item())
        point["y"] = float(torch.dot(r, e2).item())
        point["z_geometry"] = point["axial_z"]
        point["z_step"] = point["event_step"]

    norm_tensor = torch.stack(norms)
    norm_concentration = float(
        (norm_tensor.std(unbiased=False) / norm_tensor.mean().clamp_min(1e-12)).item()
    )
    return {
        "points": enriched,
        "norm_concentration": norm_concentration,
        "c0_norm": float(c0_norm.item()),
    }


def _branch_dispersion(points: List[dict], z_key: str = "event_step") -> Dict[str, float]:
    grouped = defaultdict(list)
    for point in points:
        key = (
            point["sample_id"],
            point["method"],
            point["layer_set"],
            point["event_type"],
            point[z_key],
        )
        grouped[key].append(point)
    out = {}
    for key, vals in grouped.items():
        if len(vals) < 2:
            continue
        xy = torch.tensor([[p["x"], p["y"]] for p in vals], dtype=torch.float32)
        center = xy.mean(dim=0, keepdim=True)
        disp = ((xy - center) ** 2).sum(dim=1).mean()
        out["|".join(map(str, key))] = float(disp.item())
    return out


def _sync_contractions(points: List[dict]) -> List[dict]:
    by_key = defaultdict(list)
    for point in points:
        if point["event_type"] in POLICY_EVENTS:
            key = (point["sample_id"], point["method"], point["layer_set"], point["event_step"])
            by_key[key].append(point)
    contractions = []
    for key, vals in by_key.items():
        pre = [p for p in vals if p["event_type"] == "policy_pre"]
        post = [p for p in vals if p["event_type"] == "policy_post"]
        if len(pre) < 2 or len(post) < 2:
            continue
        def dispersion(items):
            xy = torch.tensor([[p["x"], p["y"]] for p in items], dtype=torch.float32)
            center = xy.mean(dim=0, keepdim=True)
            return ((xy - center) ** 2).sum(dim=1).mean().item()
        d_pre = dispersion(pre)
        d_post = dispersion(post)
        contractions.append({
            "sample_id": key[0],
            "method": key[1],
            "layer_set": key[2],
            "event_step": key[3],
            "D_before": d_pre,
            "D_after": d_post,
            "Gamma": d_post / d_pre if d_pre > 0 else None,
        })
    return contractions


def _padded_limits(values: List[float], pad_frac: float = 0.08) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        center = (lo + hi) / 2.0
        pad = max(abs(center) * pad_frac, 1.0)
        return center - pad, center + pad
    pad = span * pad_frac
    return lo - pad, hi + pad


def _set_padded_3d_limits(ax, points: List[dict], z_key: str) -> None:
    """Pad x/y/z limits so markers and axis labels are not visually clipped."""
    ax.set_xlim(*_padded_limits([p["x"] for p in points]))
    ax.set_ylim(*_padded_limits([p["y"] for p in points]))
    ax.set_zlim(*_padded_limits([p[z_key] for p in points]))


def _set_padded_2d_limits(ax, points: List[dict]) -> None:
    ax.set_xlim(*_padded_limits([p["x"] for p in points]))
    ax.set_ylim(*_padded_limits([p["y"] for p in points]))


def _plot_group(points: List[dict], out_dir: Path, sample_id: str, layer_set: str) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    matplotlib.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   6.5,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "axes.linewidth":    0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "text.color":        BB["ink"],
        "axes.labelcolor":   BB["ink"],
        "xtick.color":       BB["ink"],
        "ytick.color":       BB["ink"],
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    block_sizes_present = sorted({int(p["block_size"]) for p in points})

    for z_name, z_key, zlabel in [
        ("geometry",     "z_geometry", "Axial displacement"),
        ("step_aligned", "z_step",     "Generation step"),
    ]:
        fig = plt.figure(figsize=(7.0, 5.4), facecolor=BB["paper"])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(BB["paper"])

        line_groups = defaultdict(list)
        for point in points:
            line_key = (point["method"], point["block_size"], point["branch_idx"])
            line_groups[line_key].append(point)
            is_refresh = point["event_type"] in FULL_EVENTS or point["event_type"] == "final"
            color = _event_color(point["event_type"], point["block_size"])
            is_origin = _is_origin_plot_point(point)
            ax.scatter(
                point["x"], point["y"], point[z_key],
                color=color,
                marker="*" if is_origin else "o",
                s=90 if is_origin else 14,
                alpha=0.95 if (is_origin or is_refresh) else 0.80,
                edgecolors=BB["ink"] if is_origin else "none",
                linewidths=0.5,
                zorder=3 if is_origin else 2,
            )

        # Trajectory lines per branch, colored by the darker block-size tone.
        for (method, bs, _), vals in line_groups.items():
            vals = sorted(vals, key=lambda p: (p["event_step"], p["event_index"]))
            ax.plot(
                [p["x"] for p in vals],
                [p["y"] for p in vals],
                [p[z_key] for p in vals],
                color=_bs_color(bs, "refresh"), linewidth=0.5, alpha=0.35,
            )

        legend_handles = []
        for bs in block_sizes_present:
            legend_handles.append(
                mpatches.Patch(facecolor=_bs_color(bs, "denoise"), label=f"bs={bs} denoise", linewidth=0)
            )
            legend_handles.append(
                mpatches.Patch(facecolor=_bs_color(bs, "refresh"), label=f"bs={bs} refresh", linewidth=0)
            )

        fig.legend(
            handles=legend_handles,
            loc="center right",
            bbox_to_anchor=(0.985, 0.52),
            ncol=1,
            framealpha=0.90,
            facecolor=BB["panel"],
            edgecolor=BB["grid"],
            fontsize=6,
        )

        ax.set_title(f"KV space — {sample_id} / {layer_set}",
                     pad=4, color=BB["ink"])
        ax.set_xlabel("$\\langle r, e_1 \\rangle$", labelpad=8)
        ax.set_ylabel("$\\langle r, e_2 \\rangle$", labelpad=8)
        ax.set_zlabel(zlabel, labelpad=8)
        ax.tick_params(labelsize=6, colors=BB["ink"])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, linewidth=0.25, color=BB["grid"], alpha=0.5)
        _set_padded_3d_limits(ax, points, z_key)
        ax.set_position([0.06, 0.12, 0.70, 0.80])

        fig.savefig(out_dir / f"{sample_id}_{layer_set}_{z_name}.png",
                    dpi=220, facecolor=BB["paper"], pad_inches=0.25)
        plt.close(fig)


def _plot_merge_sync(points: List[dict], out_dir: Path, selected_step: Optional[int] = None) -> None:
    import matplotlib.pyplot as plt

    policy_points = [p for p in points if p["event_type"] in POLICY_EVENTS]
    if selected_step is not None:
        policy_points = [p for p in policy_points if p["event_step"] == selected_step]
    if not policy_points:
        return

    grouped = defaultdict(list)
    for point in policy_points:
        grouped[(point["sample_id"], point["layer_set"], point["event_step"])].append(point)

    out_dir.mkdir(parents=True, exist_ok=True)
    for (sample_id, layer_set, event_step), vals in grouped.items():
        pre = [p for p in vals if p["event_type"] == "policy_pre"]
        post = [p for p in vals if p["event_type"] == "policy_post"]
        if not pre or not post:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 5.8), facecolor=BB["paper"])
        ax.set_facecolor(BB["paper"])
        for point in pre:
            ax.scatter(point["x"], point["y"], marker="o", s=58,
                       color=_bs_color(point["block_size"], "denoise"), alpha=0.85)
        for point in post:
            ax.scatter(point["x"], point["y"], marker="x", s=72,
                       color=_bs_color(point["block_size"], "refresh"), alpha=0.95)
        post_by_bs = {p["block_size"]: p for p in post}
        for point in pre:
            after = post_by_bs.get(point["block_size"])
            if after is None:
                continue
            ax.annotate(
                "",
                xy=(after["x"], after["y"]),
                xytext=(point["x"], point["y"]),
                arrowprops={"arrowstyle": "->", "color": "#30343b", "lw": 0.8, "alpha": 0.55},
            )
            ax.text(after["x"], after["y"], str(point["block_size"]), fontsize=8)
        ax.set_title(f"Merge/Sync Before vs After | {sample_id} | {layer_set} | step {event_step}")
        ax.set_xlabel("<r,e1>")
        ax.set_ylabel("<r,e2>")
        _set_padded_2d_limits(ax, pre + post)
        ax.grid(color=BB["grid"], alpha=0.35, linewidth=0.4)
        fig.subplots_adjust(left=0.13, right=0.96, bottom=0.14, top=0.90)
        fig.savefig(out_dir / f"{sample_id}_{layer_set}_step{event_step}_merge_sync.png",
                    dpi=220, facecolor=BB["paper"])
        plt.close(fig)


def _partial_summary(points: List[dict]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    grouped = defaultdict(list)
    for point in points:
        grouped[(point["sample_id"], point["method"], point["block_size"])].append(point)
    for (sid, method, block_size), vals in grouped.items():
        key = f"{sid}|{method}|bs{block_size}"
        steps = [int(p["event_step"]) for p in vals]
        events = sorted({p["event_type"] for p in vals})
        summary[key] = {
            "num_points": len(vals),
            "num_events": len({p["event_index"] for p in vals}),
            "min_step": min(steps) if steps else None,
            "max_step": max(steps) if steps else None,
            "event_types": events,
        }
    return summary


def analyze(official_dir: Optional[Path], bulk_dir: Optional[Path], output_dir: Path,
            vector_source: str = "sketch", sample_id: Optional[str] = None,
            merge_sync_step: Optional[int] = None,
            max_events_per_run: Optional[int] = None) -> None:
    all_points = []
    if official_dir is not None:
        all_points.extend(load_points_tree(
            official_dir,
            vector_source=vector_source,
            max_events_per_run=max_events_per_run,
        ))
    if bulk_dir is not None:
        all_points.extend(load_points_tree(
            bulk_dir,
            vector_source=vector_source,
            max_events_per_run=max_events_per_run,
        ))
    if not all_points:
        raise RuntimeError("No KV points found")
    if sample_id is not None:
        all_points = [p for p in all_points if p["sample_id"] == sample_id]
        if not all_points:
            raise RuntimeError(f"No KV points found for sample_id={sample_id!r}")

    grouped = defaultdict(list)
    for point in all_points:
        grouped[(point["sample_id"], point["layer_set"])].append(point)

    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = output_dir / "coordinates.jsonl"
    metrics = {
        "vector_source": vector_source,
        "sample_id_filter": sample_id,
        "max_events_per_run": max_events_per_run,
        "partial_summary": _partial_summary(all_points),
        "groups": {},
        "branch_dispersion": {},
        "sync_contractions": [],
    }

    with coordinates_path.open("w", encoding="utf-8") as f:
        for (sample_id, layer_set), group_points in grouped.items():
            fit = _fit_group(group_points)
            coords = fit["points"]
            metrics["groups"][f"{sample_id}|{layer_set}"] = {
                "norm_concentration": fit["norm_concentration"],
                "c0_norm": fit["c0_norm"],
                "num_points": len(coords),
            }
            metrics["branch_dispersion"].update(_branch_dispersion(coords))
            metrics["sync_contractions"].extend(_sync_contractions(coords))
            _plot_group(coords, output_dir / "plots", sample_id, layer_set)
            _plot_merge_sync(coords, output_dir / "merge_sync_dynamics", selected_step=merge_sync_step)
            for point in coords:
                f.write(json.dumps(point) + "\n")

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def analyze_coordinates(coordinates_jsonl: Path, output_dir: Path,
                        sample_id: Optional[str] = None,
                        merge_sync_step: Optional[int] = None) -> None:
    """Replot existing coordinates.jsonl without reloading KV event tensors."""
    coords = list(_load_jsonl(coordinates_jsonl))
    if sample_id is not None:
        coords = [p for p in coords if p["sample_id"] == sample_id]
    if not coords:
        raise RuntimeError(f"No coordinate points found in {coordinates_jsonl}")

    grouped = defaultdict(list)
    for point in coords:
        grouped[(point["sample_id"], point["layer_set"])].append(point)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "source_coordinates": str(coordinates_jsonl),
        "sample_id_filter": sample_id,
        "partial_summary": _partial_summary(coords),
        "groups": {},
        "branch_dispersion": {},
        "sync_contractions": [],
    }

    coordinates_path = output_dir / "coordinates.jsonl"
    with coordinates_path.open("w", encoding="utf-8") as f:
        for (sample_id, layer_set), group_points in grouped.items():
            metrics["groups"][f"{sample_id}|{layer_set}"] = {
                "num_points": len(group_points),
                "replotted_from_coordinates": True,
            }
            metrics["branch_dispersion"].update(_branch_dispersion(group_points))
            metrics["sync_contractions"].extend(_sync_contractions(group_points))
            _plot_group(group_points, output_dir / "plots", sample_id, layer_set)
            _plot_merge_sync(group_points, output_dir / "merge_sync_dynamics", selected_step=merge_sync_step)
            for point in group_points:
                f.write(json.dumps(point) + "\n")

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, default=None)
    parser.add_argument("--bulk-dir", type=Path, default=None)
    parser.add_argument("--coordinates-jsonl", type=Path, default=None,
                        help="Replot an existing coordinates.jsonl without loading KV event tensors.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vector-source", choices=["sketch", "raw"], default="sketch")
    parser.add_argument("--sample-id", default=None,
                        help="Optional sample id filter for generating one sample's graphs.")
    parser.add_argument("--merge-sync-step", type=int, default=None,
                        help="Optional event step for merge/sync before-after plots.")
    parser.add_argument("--max-events-per-run", type=int, default=None,
                        help="Use only the first N logged events from each run folder for partial plotting.")
    args = parser.parse_args()
    if args.coordinates_jsonl is not None:
        analyze_coordinates(
            args.coordinates_jsonl,
            args.output_dir,
            args.sample_id,
            args.merge_sync_step,
        )
        return
    analyze(
        args.official_dir,
        args.bulk_dir,
        args.output_dir,
        args.vector_source,
        args.sample_id,
        args.merge_sync_step,
        args.max_events_per_run,
    )


if __name__ == "__main__":
    main()
