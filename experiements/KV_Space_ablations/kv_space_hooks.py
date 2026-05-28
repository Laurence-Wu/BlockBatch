"""Observational KV-space logging utilities.

The helpers in this file are intentionally read-only with respect to generation
state. They detach cache tensors, move them to CPU, and write experiment
artifacts under the KV_Space_ablations folder.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch


EVENT_FULL = "full_sequence"
EVENT_REFRESH = "full_refresh"
EVENT_BLOCK = "block_denoise"
EVENT_POLICY_PRE = "policy_pre"
EVENT_POLICY_POST = "policy_post"
EVENT_FINAL = "final"


@dataclass
class KVHookConfig:
    enabled: bool = True
    run_id: str = "kv_space_run"
    output_dir: str = "experiements/KV_Space_ablations"
    method: str = "unknown"
    sample_id: str = "sample_0"
    sketch_dim: int = 256
    sketch_seed: int = 1729
    sketch_chunk_size: int = 8192
    raw_snapshot_limit: int = 0
    raw_snapshot_events: Sequence[str] = field(
        default_factory=lambda: (EVENT_FULL, EVENT_REFRESH, EVENT_BLOCK, EVENT_FINAL)
    )
    layer_sets: Optional[Mapping[str, Sequence[int]]] = None
    dtype: str = "float32"


def default_layer_sets(num_layers: int) -> Dict[str, List[int]]:
    if num_layers <= 0:
        return {"early": [], "middle": [], "late": [], "all": []}
    return {
        "early": [0],
        "middle": [num_layers // 2],
        "late": [num_layers - 1],
        "all": list(range(num_layers)),
    }


def _safe_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Iterable):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _cache_layer_count(cache: Sequence) -> int:
    return len(cache) if cache is not None else 0


def _cache_batch_size(cache: Sequence) -> int:
    if not cache:
        return 0
    return int(cache[0][0].shape[0])


def _normalize_layer_sets(cache: Sequence, layer_sets: Optional[Mapping[str, Sequence[int]]]) -> Dict[str, List[int]]:
    num_layers = _cache_layer_count(cache)
    raw_sets = default_layer_sets(num_layers) if layer_sets is None else {
        str(name): list(indices) for name, indices in layer_sets.items()
    }
    normalized: Dict[str, List[int]] = {}
    for name, indices in raw_sets.items():
        valid = sorted({int(i) for i in indices if 0 <= int(i) < num_layers})
        if valid:
            normalized[name] = valid
    return normalized


def flatten_cache_vector(cache: Sequence, branch_row: int, layers: Sequence[int]) -> torch.Tensor:
    """Return one raw concatenated K,V vector for a branch and layer set."""
    parts = []
    for layer_idx in layers:
        key, value = cache[layer_idx]
        parts.append(key[branch_row].detach().reshape(-1).to(device="cpu", dtype=torch.float32))
        parts.append(value[branch_row].detach().reshape(-1).to(device="cpu", dtype=torch.float32))
    if not parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(parts, dim=0)


def _random_projection_chunk(chunk: torch.Tensor, sketch_dim: int, seed: int, chunk_idx: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(chunk_idx) * 1000003)
    proj = torch.randn(
        chunk.numel(),
        sketch_dim,
        generator=generator,
        dtype=torch.float32,
    )
    return chunk.to(torch.float32).matmul(proj)


def deterministic_sketch(vector: torch.Tensor, sketch_dim: int, seed: int, chunk_size: int) -> torch.Tensor:
    """Gaussian random projection with deterministic per-chunk matrices."""
    if vector.numel() == 0:
        return torch.zeros(sketch_dim, dtype=torch.float32)
    out = torch.zeros(sketch_dim, dtype=torch.float32)
    flat = vector.to(dtype=torch.float32, device="cpu")
    for chunk_idx, start in enumerate(range(0, flat.numel(), chunk_size)):
        chunk = flat[start:start + chunk_size]
        out += _random_projection_chunk(chunk, sketch_dim, seed, chunk_idx)
    return out / math.sqrt(float(sketch_dim))


class KVSpaceLogger:
    """Event logger for raw/sketched KV geometry analysis."""

    def __init__(self, config: KVHookConfig):
        self.config = config
        self.enabled = bool(config.enabled)
        self.root = Path(config.output_dir).resolve()
        self.run_dir = self.root / config.method / config.run_id
        self.event_dir = self.run_dir / "events"
        self.raw_dir = self.run_dir / "raw"
        self.index_path = self.run_dir / "events.jsonl"
        self._event_index = 0
        self._raw_snapshots_written = 0
        if self.enabled:
            self.event_dir.mkdir(parents=True, exist_ok=True)
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "config.json").write_text(
                json.dumps(_to_jsonable(asdict(config)), indent=2) + "\n"
            )

    def record_cache(
        self,
        event_type: str,
        cache: Optional[Sequence],
        *,
        event_step: int,
        branch_indices: Sequence[int],
        block_sizes: Sequence[int],
        sample_id: Optional[str] = None,
        nfe: Optional[int] = None,
        extra: Optional[Mapping] = None,
    ) -> None:
        if not self.enabled or cache is None:
            return
        layer_sets = _normalize_layer_sets(cache, self.config.layer_sets)
        if not layer_sets:
            return

        batch_size = _cache_batch_size(cache)
        branch_indices = [int(i) for i in branch_indices]
        block_sizes = [int(bs) for bs in block_sizes]
        if len(branch_indices) != len(block_sizes):
            raise ValueError("branch_indices and block_sizes must have the same length")

        row_map = []
        for local_row, branch_idx in enumerate(branch_indices):
            row = branch_idx if branch_idx < batch_size else local_row
            if row >= batch_size:
                raise IndexError(f"cache row {row} out of range for batch size {batch_size}")
            row_map.append(row)

        should_store_raw = (
            event_type in set(self.config.raw_snapshot_events)
            and self._raw_snapshots_written < int(self.config.raw_snapshot_limit)
        )

        payload = {
            "event_index": self._event_index,
            "event_type": event_type,
            "event_step": int(event_step),
            "method": self.config.method,
            "sample_id": str(sample_id if sample_id is not None else self.config.sample_id),
            "nfe": None if nfe is None else int(nfe),
            "branches": [
                {"branch_idx": int(branch_idx), "block_size": int(block_size), "cache_row": int(row)}
                for branch_idx, block_size, row in zip(branch_indices, block_sizes, row_map)
            ],
            "layer_sets": {},
            "extra": _to_jsonable(extra or {}),
        }

        raw_payload = {}
        for set_name, layers in layer_sets.items():
            sketches = []
            norms = []
            raw_vectors = []
            dims = []
            for row in row_map:
                vector = flatten_cache_vector(cache, row, layers)
                dims.append(int(vector.numel()))
                norms.append(_safe_float(torch.linalg.vector_norm(vector)))
                sketches.append(deterministic_sketch(
                    vector,
                    int(self.config.sketch_dim),
                    int(self.config.sketch_seed),
                    int(self.config.sketch_chunk_size),
                ))
                if should_store_raw:
                    raw_vectors.append(vector)

            payload["layer_sets"][set_name] = {
                "layers": list(layers),
                "dims": dims,
                "norms": norms,
                "sketch": torch.stack(sketches, dim=0) if sketches else torch.empty(0),
            }
            if should_store_raw:
                raw_payload[set_name] = torch.stack(raw_vectors, dim=0) if raw_vectors else torch.empty(0)

        event_path = self.event_dir / f"event_{self._event_index:06d}.pt"
        torch.save(payload, event_path)

        raw_path = None
        if should_store_raw:
            raw_path = self.raw_dir / f"raw_{self._event_index:06d}.pt"
            torch.save(raw_payload, raw_path)
            self._raw_snapshots_written += 1

        index_record = {
            "event_index": self._event_index,
            "event_type": event_type,
            "event_step": int(event_step),
            "method": self.config.method,
            "sample_id": str(sample_id if sample_id is not None else self.config.sample_id),
            "nfe": None if nfe is None else int(nfe),
            "branch_count": len(branch_indices),
            "block_sizes": block_sizes,
            "event_path": str(event_path.relative_to(self.run_dir)),
            "raw_path": None if raw_path is None else str(raw_path.relative_to(self.run_dir)),
            "extra": _to_jsonable(extra or {}),
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(index_record) + "\n")

        self._event_index += 1


def state_branch_indices(states: Sequence) -> List[int]:
    return [int(s.branch_idx) for s in states]


def state_block_sizes(states: Sequence) -> List[int]:
    return [int(s.block_size) for s in states]
