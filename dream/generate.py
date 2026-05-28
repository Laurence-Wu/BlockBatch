# Copyright 2026 BlockBatch Authors
# SPDX-License-Identifier: Apache-2.0

"""Dream generation wrapper with LLaDA-like API.

This module intentionally delegates generation to Dream's official model-side
implementation in model/generation_utils_block.py (DreamGenerationMixin).

Behavioral guarantees for this wrapper:
- Always uses confidence-threshold decoding (alg='confidence_threshold')
- Always uses dual cache (dual_cache=True)
- Returns (full_sequence_ids, nfe) like llada/generate.py
"""

from __future__ import annotations

import types
from typing import Optional, Tuple

import torch

from model.generation_utils_block import DreamGenerationMixin


def _bind_official_generation(model) -> None:
    """Ensure official Dream generation methods are bound on the model instance."""
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)


def _validate_args(prompt: torch.Tensor, steps: int, gen_length: int, block_length: int) -> None:
    if prompt.dim() != 2:
        raise ValueError(f"prompt must be rank-2 [B, L], got shape {tuple(prompt.shape)}")
    if gen_length <= 0:
        raise ValueError(f"gen_length must be > 0, got {gen_length}")
    if block_length <= 0:
        raise ValueError(f"block_length must be > 0, got {block_length}")
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(
            f"steps ({steps}) must be divisible by num_blocks ({num_blocks})")


def _run_official_dream_generate(
    model,
    prompt: torch.Tensor,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    threshold: Optional[float],
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Run official Dream diffusion_generate and count forward evaluations (NFE)."""
    _bind_official_generation(model)

    pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        attention_mask = torch.ones_like(prompt, dtype=torch.long, device=prompt.device)
    else:
        attention_mask = prompt.ne(pad_token_id).long()

    nfe_counter = {"nfe": 0}

    def _count_pre_hook(_module, _args):
        nfe_counter["nfe"] += 1

    handle = model.register_forward_pre_hook(_count_pre_hook)
    try:
        sequences = model.diffusion_generate(
            prompt,
            attention_mask=attention_mask,
            max_new_tokens=gen_length,
            output_history=False,
            return_dict_in_generate=False,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alg="confidence_threshold",
            threshold=threshold,
            dual_cache=True,
            block_length=block_length,
        )
    finally:
        handle.remove()

    return sequences, nfe_counter["nfe"]


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: Optional[int] = None,
    threshold: Optional[float] = 0.9,
    factor: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
):
    """Dream generation entrypoint with llada/generate.py-compatible signature.

    Notes:
    - remasking/factor are accepted for API compatibility but ignored.
    - mask_id is accepted for API compatibility but generation relies on model config.
    - This wrapper always forces confidence-threshold decoding and dual-cache mode.
    """
    del remasking, factor, mask_id
    if threshold is None:
        threshold = 0.9
    _validate_args(prompt, steps, gen_length, block_length)
    return _run_official_dream_generate(
        model,
        prompt,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        threshold=threshold,
        top_p=top_p,
        top_k=top_k,
    )


@torch.no_grad()
def generate_with_dual_cache(
    model,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: Optional[int] = None,
    threshold: Optional[float] = 0.9,
    factor: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
):
    """Alias kept for familiarity with llada/generate.py naming."""
    return generate(
        model=model,
        prompt=prompt,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        remasking=remasking,
        mask_id=mask_id,
        threshold=threshold,
        factor=factor,
        top_p=top_p,
        top_k=top_k,
    )
