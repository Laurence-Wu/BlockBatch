"""Small runner for KV-space ablation logging.

This script is intentionally narrow: it runs one prompt through the local
ablation copies, writes KV logs, and leaves plotting to analyze_kv_space.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
LLADA_DIR = REPO_ROOT / "llada"
for path in (str(THIS_DIR), str(LLADA_DIR), str(REPO_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
for path in reversed((str(THIS_DIR), str(LLADA_DIR), str(REPO_ROOT))):
    sys.path.insert(0, path)

from model.modeling_llada import LLaDAModelLM  # noqa: E402
from generate import generate_with_dual_cache_hooked  # noqa: E402
from generate_blockBatching_hooked_bulk import generate_block_batching  # noqa: E402
from kv_space_hooks import KVHookConfig, KVSpaceLogger  # noqa: E402


def _parse_block_sizes(raw: str) -> List[int]:
    return [int(x) for x in raw.replace(",", "-").split("-") if x]


def _load_model(model_path: str, cache_dir: str | None, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model = LLaDAModelLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    ).to(device).eval()
    return model, tokenizer


def _make_logger(args, method: str, run_suffix: str) -> KVSpaceLogger:
    return KVSpaceLogger(KVHookConfig(
        enabled=True,
        run_id=f"{args.run_id}_{run_suffix}",
        output_dir=str(args.output_dir),
        method=method,
        sample_id=args.sample_id,
        sketch_dim=args.sketch_dim,
        raw_snapshot_limit=args.raw_snapshot_limit,
    ))


@torch.no_grad()
def run(args) -> None:
    block_sizes = _parse_block_sizes(args.block_sizes)
    model, tokenizer = _load_model(args.model_path, args.cache_dir, args.device)
    prompt_text = args.prompt
    if args.apply_chat_template:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
    prompt = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

    if args.mode in {"official", "both"}:
        for block_size in block_sizes:
            logger = _make_logger(args, "official_llada", f"official_bs{block_size}")
            generate_with_dual_cache_hooked(
                model,
                prompt,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=block_size,
                temperature=args.temperature,
                remasking=args.remasking,
                mask_id=args.mask_id,
                threshold=args.threshold,
                kv_hook=logger,
                sample_id=args.sample_id,
            )

    if args.mode in {"bulk", "both"}:
        logger = _make_logger(args, "original_bulk", "bulk")
        generate_block_batching(
            model,
            prompt,
            gen_length=args.gen_length,
            block_sizes=block_sizes,
            temperature=args.temperature,
            remasking=args.remasking,
            mask_id=args.mask_id,
            threshold=args.threshold,
            kv_hook=logger,
            sample_id=args.sample_id,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["official", "bulk", "both"], default="both")
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "huggingface"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default="Write a Python function that returns True if a number is prime.")
    parser.add_argument("--apply-chat-template", action="store_true",
                        help="Wrap the prompt with the LLaDA instruct chat template before tokenization.")
    parser.add_argument("--sample-id", default="sample_0")
    parser.add_argument("--run-id", default="kv_space")
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR)
    parser.add_argument("--block-sizes", default="4-8-16-32-64-128")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence")
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--sketch-dim", type=int, default=256)
    parser.add_argument("--raw-snapshot-limit", type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
