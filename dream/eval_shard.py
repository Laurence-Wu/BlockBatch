"""Sharded lm_eval wrapper.

Monkey-patches lm_eval to evaluate only the docs assigned to this shard
(stride-based: shard_index, shard_index+shard_count, shard_index+2*shard_count, ...).

Usage (via accelerate):
    accelerate launch eval_shard.py \
        --shard-index 0 --shard-count 4 \
        --model dream \
        --model_args "pretrained=...,save_dir=/path/to/shard0" \
        --tasks minerva_math --num_fewshot 4 --batch_size 1 \
        --output_path /path/to/shard0/lm_eval --log_samples
"""
import argparse
import sys

# ── Parse shard args before lm_eval sees sys.argv ────────────────────────────
_shard_parser = argparse.ArgumentParser(add_help=False)
_shard_parser.add_argument("--shard-index", type=int, required=True)
_shard_parser.add_argument("--shard-count", type=int, default=4)
_shard_args, _remaining = _shard_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining   # hide shard flags from lm_eval


# ── Monkey-patch Task.build_all_requests to slice docs for this shard ────────
def _patch_lm_eval_for_shard(shard_index: int, shard_count: int) -> None:
    from lm_eval.api import task as _task_mod

    _orig = _task_mod.Task.build_all_requests

    def _patched(self, *args, **kwargs):
        _orig(self, *args, **kwargs)
        # lm_eval 0.4.x: instances is a read-only property backed by _instances
        self._instances = self._instances[shard_index::shard_count]

    _task_mod.Task.build_all_requests = _patched


_patch_lm_eval_for_shard(_shard_args.shard_index, _shard_args.shard_count)

# ── Import eval.py to register the Dream model class ─────────────────────────
import eval  # noqa: F401  (registers @register_model("dream"))

# ── Hand off to lm_eval CLI ───────────────────────────────────────────────────
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
