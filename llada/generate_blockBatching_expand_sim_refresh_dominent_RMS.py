"""
generate_blockBatching.py  —  Fused Batched Block Batching (LLaDA)

Port of generate_blockBatching_fusedBest_v2 — public entry point is generate_block_batching().

Extends V1 with batched block-denoise + unified KV cache:
  - Initial full-seq forward: ONE forward (batch=1), KV expanded to all N branches
  - Main loop: batched block-denoise every step (O(sum(block_sizes) × L) FLOP)
  - Two monkey-patches: RoPE for per-batch Q positions + SDPA for varlen packing
  - RMS chord shell-triggered KV refresh (each active branch refreshes its own row)

Optimizations vs naive V2:
  - Initial forward: N× → 1× model call (KV broadcast)
  - Block-denoise Q tokens: N×max_q_len → sum(block_sizes) via varlen packing
  - For BLOCK_SIZES=[4,8,16,32,64]: 320 → 124 Q tokens (<128 FA tile threshold)

Entry point: run_fused_block_batching_v2()
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import math
from generate import get_transfer_index

# ── Logging ──────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def _sync_for_timing(device: Optional[torch.device] = None) -> None:
    """Synchronize CUDA work before taking a wall-clock timestamp."""
    if device is not None and device.type != "cuda":
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize(device)


def _timed_now(device: Optional[torch.device] = None) -> float:
    _sync_for_timing(device)
    return time.time()


def _timed_elapsed(start: float, device: Optional[torch.device] = None) -> float:
    _sync_for_timing(device)
    return time.time() - start


# ── Constants ────────────────────────────────────────────────────────────────

_TOKEN_PROB_THRESHOLD = 0.1
SYNC_THRESHOLD = 8
CONF_THRESHOLD = 0.5
REFRESH_BLOCK_SIZE = 32  # KV refresh granularity (tokens); independent of BLOCK_SIZES


# ── RoPE monkey-patch for per-batch Q positions ──────────────────────────────
#
# LLaDA's RotaryEmbedding.forward uses a scalar block_end_index to compute
# Q RoPE positions (lines 761-764 in modeling_llada.py). For batched
# block-denoise where each branch has a different block_end, we need per-batch
# Q positions. This patch intercepts forward() when _FUSED_POSITION_IDS is set.
#
# K always gets full-sequence RoPE (positions 0..L-1); only Q needs per-batch.

_FUSED_POSITION_IDS: Optional[torch.Tensor] = None  # (N, max_q_len), set during block-denoise
_original_rope_forward = None   # saved original at patch time


def _patched_rope_forward(self, q: torch.Tensor, k: torch.Tensor,
                           block_end_index=None):
    """Drop-in replacement for RotaryEmbedding.forward.

    When _FUSED_POSITION_IDS is set (during batched block-denoise):
      - Q gets per-batch RoPE via gathered sin/cos[position_ids]
      - K gets standard full-sequence RoPE (positions 0..key_len-1)
    Otherwise falls back to the original forward.
    """
    global _FUSED_POSITION_IDS
    if _FUSED_POSITION_IDS is not None:
        position_ids = _FUSED_POSITION_IDS  # (B, q_len)
        q_ = q.float() if self.config.rope_full_precision else q
        k_ = k.float() if self.config.rope_full_precision else k
        with torch.autocast(q.device.type, enabled=False):
            key_len = k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            # Q: gather sin/cos for per-batch absolute positions
            # pos_sin/pos_cos shape: (1, 1, key_len, hd) → squeeze to (key_len, hd)
            sin_table = pos_sin.squeeze(0).squeeze(0)  # (key_len, hd)
            cos_table = pos_cos.squeeze(0).squeeze(0)  # (key_len, hd)
            q_sin = sin_table[position_ids]             # (B, q_len, hd)
            q_cos = cos_table[position_ids]             # (B, q_len, hd)
            q_sin = q_sin.unsqueeze(1)                  # (B, 1, q_len, hd)
            q_cos = q_cos.unsqueeze(1)                  # (B, 1, q_len, hd)
            q_ = self.apply_rotary_pos_emb(q_sin, q_cos, q_)
            # K: standard full-sequence RoPE
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)
    return _original_rope_forward(self, q, k, block_end_index)


def _apply_rope_patch() -> None:
    """Monkey-patch RotaryEmbedding.forward once. Safe to call multiple times."""
    global _original_rope_forward
    if _original_rope_forward is not None:
        return  # already patched
    from model.modeling_llada import RotaryEmbedding
    _original_rope_forward = RotaryEmbedding.forward
    RotaryEmbedding.forward = _patched_rope_forward
    logger.debug("V2 RoPE monkey-patch applied")


# ── Varlen SDPA monkey-patch ──────────────────────────────────────────────────
#
# LLaDA's _scaled_dot_product_attention pads all N branches to max_q_len.
# For BLOCK_SIZES=[4,8,16,32,64]: 5×64=320 Q tokens, but only 124 real tokens.
# When _VARLEN_Q_LENS is set, this patch uses flash_attn_varlen_func to pack
# only the real Q tokens per branch, avoiding padding waste.
#
# At SDPA call time (after replace_position in-place update):
#   q: (B, nh, max_q_len, hs)   — padded
#   k: (B, n_kv_h, L, hs)       — full KV cache (L = Lp + gen_length)
#   v: (B, n_kv_h, L, hs)
#
# Packed:
#   q_packed: (total_q, nh, hs)  — real tokens only, no padding
#   k_packed: (B*L, n_kv_h, hs) — all branches, each of length L

_VARLEN_Q_LENS: Optional[List[int]] = None  # real Q len per batch element, set during block-denoise
_original_sdpa = None


def _patched_sdpa(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    """Drop-in for LLaDABlock._scaled_dot_product_attention.

    When _VARLEN_Q_LENS is set AND flash_attn is available: uses varlen packing.
    Otherwise falls back to the original SDPA.
    """
    global _VARLEN_Q_LENS
    if _VARLEN_Q_LENS is not None and self.flash_attn_func is not None and attn_mask is None:
        from flash_attn import flash_attn_varlen_func

        B, nh, max_q, hs = q.shape
        n_kv_h, L = k.shape[1], k.shape[2]
        q_lens = _VARLEN_Q_LENS  # list[int], len == B

        # Pack Q: (B, nh, max_q, hs) → (total_q, nh, hs)
        q_t = q.permute(0, 2, 1, 3)  # (B, max_q, nh, hs)
        q_packed = torch.cat([q_t[i, :q_lens[i]] for i in range(B)], dim=0)

        # cu_seqlens_q: cumulative sum [0, q0, q0+q1, ...]
        cu_q = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        for i, ql in enumerate(q_lens):
            cu_q[i + 1] = cu_q[i] + ql

        # Pack K/V: (B, n_kv_h, L, hs) → (B*L, n_kv_h, hs)
        k_packed = k.permute(0, 2, 1, 3).contiguous().view(B * L, n_kv_h, hs)
        v_packed = v.permute(0, 2, 1, 3).contiguous().view(B * L, n_kv_h, hs)
        cu_k = torch.arange(0, (B + 1) * L, L, dtype=torch.int32, device=q.device)

        out_packed = flash_attn_varlen_func(
            q_packed, k_packed, v_packed, cu_q, cu_k,
            max_seqlen_q=max(q_lens), max_seqlen_k=L,
            dropout_p=dropout_p, causal=False,
        )  # (total_q, nh, hs)

        # Unpack: fill real tokens, leave padding as zeros
        out = torch.zeros(B, max_q, nh, hs, device=q.device, dtype=q.dtype)
        offset = 0
        for i, q_len in enumerate(q_lens):
            out[i, :q_len] = out_packed[offset:offset + q_len]
            offset += q_len
        return out.permute(0, 2, 1, 3)  # (B, nh, max_q, hs)

    return _original_sdpa(self, q, k, v, attn_mask, dropout_p, is_causal)


def _apply_sdpa_patch() -> None:
    """Monkey-patch LLaDABlock._scaled_dot_product_attention once."""
    global _original_sdpa
    if _original_sdpa is not None:
        return  # already patched
    from model.modeling_llada import LLaDABlock
    _original_sdpa = LLaDABlock._scaled_dot_product_attention
    LLaDABlock._scaled_dot_product_attention = _patched_sdpa
    logger.debug("V2 varlen SDPA monkey-patch applied")


# ── Full varlen block monkey-patch ──────────────────────────────────────────
#
# When _VARLEN_Q_LENS is set (block denoise), pack real tokens once at the top
# of the block, run attn_norm + QKV + MLP on the compact (total_q, d) tensor,
# and scatter back only for self.attention (which already handles KV scatter /
# RoPE / varlen SDPA via the existing patches) and the final return.
#
# For BLOCK_SIZES=[4,8,16,32,64,128] this shrinks the GEMMs from
# max_q=128 × N=6 = 768 rows down to total_q=252 rows (~67% fewer FLOPs in
# attn_norm, q/k/v_proj, and the MLP).

_original_llama_block_forward = None

# ── Kernel-level profiling for the varlen block forward ────────────────
# Uses CUDA events so stage boundaries don't stall the GPU; the synchronized
# block-denoise elapsed timer doubles as the flush barrier (see
# _flush_layer_timing).

_PROFILE_LAYERS: bool = True  # toggle to False to disable with zero overhead
_LAYER_TIMING: dict = {}  # stage → (accumulated_ms, count)
_PENDING_EVENTS: list = []  # [(stage, start_evt, end_evt), ...]

_LAYER_STAGES = (
    'layer_wall',
    'pack_x',
    'attn_norm',
    'qkv_proj',
    'qkv_reshape',
    'kv_scatter',
    'rope_q',
    'rope_k',
    'fa_prep',
    'flash_attn_core',
    'attn_out_proj',
    'attn_residual',
    'mlp_norm',
    'mlp_gate',
    'mlp_down',
    'mlp_residual',
    'unpack_out',
)


def _layer_event_pair() -> Tuple[torch.cuda.Event, torch.cuda.Event]:
    """Create two timing events; caller records start/end around the stage."""
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    return s, e


def _reset_layer_timing() -> None:
    _LAYER_TIMING.clear()
    _PENDING_EVENTS.clear()


def _flush_layer_timing(stats_top: dict) -> None:
    """Convert recorded events → accumulated milliseconds in stats_top.
    Must be called after a synchronized elapsed timer so all events are ready."""
    if not _PENDING_EVENTS:
        return
    for stage, s, e in _PENDING_EVENTS:
        dt_ms = s.elapsed_time(e)
        prev_ms, prev_n = _LAYER_TIMING.get(stage, (0.0, 0))
        _LAYER_TIMING[stage] = (prev_ms + dt_ms, prev_n + 1)
    _PENDING_EVENTS.clear()
    for stage, (ms, n) in _LAYER_TIMING.items():
        key = f'layer_{stage}_time'
        stats_top[key] = stats_top.get(key, 0.0) + ms / 1000.0
        ck = f'layer_{stage}_count'
        stats_top[ck] = stats_top.get(ck, 0) + n
    _LAYER_TIMING.clear()


def _pack_varlen(x_padded: torch.Tensor, q_lens: List[int]) -> torch.Tensor:
    """(N, max_q, *F) padded → (total_q, *F) packed, real tokens only."""
    return torch.cat([x_padded[i, :q_lens[i]] for i in range(len(q_lens))], dim=0)


def _unpack_varlen(x_pack: torch.Tensor, q_lens: List[int],
                   N: int, max_q: int) -> torch.Tensor:
    """(total_q, *F) packed → (N, max_q, *F) padded; padding slots = 0."""
    out = x_pack.new_zeros((N, max_q, *x_pack.shape[1:]))
    offset = 0
    for i, ql in enumerate(q_lens):
        out[i, :ql] = x_pack[offset:offset + ql]
        offset += ql
    return out


def _patched_llama_block_forward(self, x, attention_bias=None, layer_past=None,
                                  use_cache=False, replace_position=None):
    """LLaDALlamaBlock.forward with fully-packed varlen path (bypasses self.attention).

    When _VARLEN_Q_LENS is None: identical to the original.
    When set: packs real tokens once, runs attn_norm / QKV / KV-scatter /
    RoPE / flash_attn_varlen / attn_out / residual / MLP all on the compact
    (total_q, d) tensor — one _pack_varlen at the top, one _unpack_varlen at
    the bottom, zero padding-position compute in between. KV cache at
    non-block positions is preserved (replenished from prior forward passes).
    """
    if _VARLEN_Q_LENS is None:
        return _original_llama_block_forward(
            self, x, attention_bias=attention_bias, layer_past=layer_past,
            use_cache=use_cache, replace_position=replace_position)

    q_lens = _VARLEN_Q_LENS
    N, max_q = x.shape[0], x.shape[1]
    profile = _PROFILE_LAYERS

    def _stage(name):
        s, e = _layer_event_pair()
        s.record()
        _PENDING_EVENTS.append((name, s, e))
        return e  # caller records e after the stage

    if profile: e_wall = _stage('layer_wall')

    if profile: e = _stage('pack_x')
    x_pack = _pack_varlen(x, q_lens)
    if profile: e.record()

    if profile: e = _stage('attn_norm')
    h_pack = self.attn_norm(x_pack)
    if profile: e.record()

    if profile: e = _stage('qkv_proj')
    q_pack, k_pack, v_pack = self.q_proj(h_pack), self.k_proj(h_pack), self.v_proj(h_pack)
    if profile: e.record()

    if profile: e = _stage('qkv_reshape')
    n_heads    = self.config.n_heads
    n_kv_heads = self.config.effective_n_kv_heads
    head_dim   = self.config.d_model // n_heads
    total_q    = x_pack.shape[0]
    q_pack_hd = q_pack.view(total_q, n_heads,    head_dim)
    k_pack_hd = k_pack.view(total_q, n_kv_heads, head_dim)
    v_pack_hd = v_pack.view(total_q, n_kv_heads, head_dim)
    if profile: e.record()

    if profile: e = _stage('kv_scatter')
    k_cache, v_cache = layer_past
    L_cache = k_cache.shape[2]
    offset = 0
    for bi, ql in enumerate(q_lens):
        rp = replace_position[bi].nonzero(as_tuple=True)[0]
        if len(rp) > 0:
            k_cache[bi, :, rp] = k_pack_hd[offset:offset + ql].permute(1, 0, 2)
            v_cache[bi, :, rp] = v_pack_hd[offset:offset + ql].permute(1, 0, 2)
        offset += ql
    if profile: e.record()

    if profile: e = _stage('rope_q')
    rotary_emb = self.rotary_emb
    rope_fp32  = rotary_emb.config.rope_full_precision
    pos_sin_full, pos_cos_full = rotary_emb.get_rotary_embedding(L_cache, x.device)
    sin_table = pos_sin_full.squeeze(0).squeeze(0)
    cos_table = pos_cos_full.squeeze(0).squeeze(0)
    pos_flat = torch.cat(
        [_FUSED_POSITION_IDS[bi, :q_lens[bi]] for bi in range(N)], dim=0
    )
    q_sin = sin_table[pos_flat].unsqueeze(0).unsqueeze(0)
    q_cos = cos_table[pos_flat].unsqueeze(0).unsqueeze(0)
    q_4d = q_pack_hd.permute(1, 0, 2).unsqueeze(0)
    if rope_fp32:
        q_4d, q_sin, q_cos = q_4d.float(), q_sin.float(), q_cos.float()
    else:
        q_sin = q_sin.to(q_4d.dtype)
        q_cos = q_cos.to(q_4d.dtype)
    q_roped = rotary_emb.apply_rotary_pos_emb(q_sin, q_cos, q_4d).to(x.dtype)
    q_fa = q_roped.squeeze(0).permute(1, 0, 2).contiguous()
    if profile: e.record()

    if profile: e = _stage('rope_k')
    k_4d = k_cache.to(torch.float32) if rope_fp32 else k_cache
    ps_k = pos_sin_full.to(k_4d.dtype)
    pc_k = pos_cos_full.to(k_4d.dtype)
    k_roped = rotary_emb.apply_rotary_pos_emb(ps_k, pc_k, k_4d).to(x.dtype)
    if profile: e.record()

    if profile: e = _stage('fa_prep')
    from flash_attn import flash_attn_varlen_func
    k_fa = k_roped.permute(0, 2, 1, 3).contiguous().view(N * L_cache, n_kv_heads, head_dim)
    v_fa = v_cache.permute(0, 2, 1, 3).contiguous().view(N * L_cache, n_kv_heads, head_dim)
    cu_q = torch.zeros(N + 1, dtype=torch.int32, device=x.device)
    for i, ql in enumerate(q_lens):
        cu_q[i + 1] = cu_q[i] + ql
    cu_k = torch.arange(0, (N + 1) * L_cache, L_cache, dtype=torch.int32, device=x.device)
    if profile: e.record()

    if profile: e = _stage('flash_attn_core')
    att_pack = flash_attn_varlen_func(
        q_fa.to(torch.bfloat16),
        k_fa.to(torch.bfloat16),
        v_fa.to(torch.bfloat16),
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lens), max_seqlen_k=L_cache,
        dropout_p=0.0, causal=False,
    )
    if profile: e.record()

    if profile: e = _stage('attn_out_proj')
    att_pack = self.attn_out(att_pack.view(total_q, n_heads * head_dim).to(x.dtype))
    cache = (k_cache, v_cache) if use_cache else None
    if profile: e.record()

    if profile: e = _stage('attn_residual')
    x_pack = x_pack + self.dropout(att_pack)
    if profile: e.record()

    if profile: e = _stage('mlp_norm')
    res = x_pack
    x_pack = self.ff_norm(x_pack)
    if profile: e.record()

    if profile: e = _stage('mlp_gate')
    x_pack = self.act(self.ff_proj(x_pack)) * self.up_proj(x_pack)
    if profile: e.record()

    if profile: e = _stage('mlp_down')
    x_pack = self.ff_out(x_pack)
    if profile: e.record()

    if profile: e = _stage('mlp_residual')
    x_pack = res + self.dropout(x_pack)
    if profile: e.record()

    if profile: e = _stage('unpack_out')
    out = _unpack_varlen(x_pack, q_lens, N, max_q)
    if profile: e.record()

    if profile: e_wall.record()

    return out, cache


def _apply_llama_block_patch() -> None:
    """Monkey-patch LLaDALlamaBlock.forward once. Safe to call multiple times."""
    global _original_llama_block_forward
    if _original_llama_block_forward is not None:
        return  # already patched
    from model.modeling_llada import LLaDALlamaBlock
    _original_llama_block_forward = LLaDALlamaBlock.forward
    LLaDALlamaBlock.forward = _patched_llama_block_forward
    logger.debug("V2 LLaDA block varlen monkey-patch applied")


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class FusedState:
    """Per-branch state for fused batched generation (V2).

    x is a VIEW into x_batch[branch_idx:branch_idx+1].
    Never rebind state.x — writes must go through x_batch[branch_idx, ...].
    """
    branch_idx: int
    block_size: int

    x: torch.Tensor          # (1, Lp+gen_length) — VIEW into x_batch row
    progress: int

    block_start: int
    block_end: int

    done: bool
    hit_eos: bool
    eos_position: Optional[int]
    eos_prob: float

    nfe_full: int
    total_forward_time: float
    tokens_decoded: int = 0
    nfe_block: int = 0

    # Policy data — lazy-filled on first forward
    token_prob_map: list = field(default_factory=list)  # [{tok: prob, ...} per position]
    tokens_merged: int = 0
    tokens_copied_from: int = 0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _check_eos_in_block(state: FusedState, x_batch: torch.Tensor,
                        block_logits: torch.Tensor, block_start: int, block_end: int,
                        mask_id: int, eos_token_id: int) -> None:
    """Set state.hit_eos/eos_position/eos_prob if [block_start, block_end) is fully decoded
    and contains an EOS token."""
    bi = state.branch_idx
    tokens = x_batch[bi, block_start:block_end]
    if (tokens == mask_id).any():
        return
    if eos_token_id not in tokens:
        return
    state.hit_eos = True
    eos_idx = (tokens == eos_token_id).nonzero(as_tuple=True)[0][0].item()
    state.eos_position = block_start + eos_idx
    p = F.softmax(block_logits.to(torch.bfloat16), dim=-1)
    state.eos_prob = p[0, eos_idx, eos_token_id].item()


def _update_token_prob_map(state: FusedState, probs_gpu: torch.Tensor,
                           offset: int) -> None:
    """Sparse-update state.token_prob_map from a GPU prob tensor."""
    if not state.token_prob_map:
        state.token_prob_map = [{} for _ in range(state.x.shape[1])]
    tpm = state.token_prob_map
    pos_idx, tok_idx = (probs_gpu > _TOKEN_PROB_THRESHOLD).nonzero(as_tuple=True)
    pos_list = pos_idx.cpu().tolist()
    tok_list = tok_idx.cpu().tolist()
    val_list = probs_gpu[pos_idx, tok_idx].cpu().tolist()
    for i in range(probs_gpu.shape[0]):
        tpm[offset + i] = {}
    for p, t, v in zip(pos_list, tok_list, val_list):
        tpm[offset + p][t] = v


def update_block_window(state: FusedState, Lp: int, gen_length: int,
                        mask_id: int) -> None:
    """Snap block window to the first MASK token and update progress."""
    mask_positions = (state.x[:, Lp:] == mask_id).nonzero(as_tuple=True)[1]
    if len(mask_positions) == 0:
        state.done = True
        state.block_start = Lp + gen_length
        state.block_end = Lp + gen_length
        state.progress = gen_length
    else:
        first_mask = mask_positions[0].item()
        block_idx = first_mask // state.block_size
        state.block_start = Lp + block_idx * state.block_size
        state.block_end = min(state.block_start + state.block_size, Lp + gen_length)
        state.progress = block_idx * state.block_size


def _unmask_inplace(state: FusedState, x_batch: torch.Tensor,
                    logits: torch.Tensor, block_start: int, block_end: int,
                    mask_id: int, temperature: float, remasking: str,
                    threshold: float) -> None:
    """Unmask tokens in [block_start, block_end) in-place on x_batch."""
    bi = state.branch_idx
    block_slice = x_batch[bi:bi + 1, block_start:block_end]
    mask = (block_slice == mask_id)
    x0, transfer = get_transfer_index(logits, temperature, remasking, mask,
                                      block_slice, None, threshold)
    state.tokens_decoded += transfer.sum().item()
    x_batch[bi, block_start:block_end] = torch.where(
        transfer.squeeze(0), x0.squeeze(0), x_batch[bi, block_start:block_end])


# ── V2 Forward functions ─────────────────────────────────────────────────────

def _broadcast_kv(past_key_values, N: int) -> list:
    """Expand batch=1 KV cache to N independent clones."""
    return [
        (k.expand(N, -1, -1, -1).clone(), v.expand(N, -1, -1, -1).clone())
        for k, v in past_key_values
    ]


def _copy_refreshed_cache_rows(refreshed_cache: list, unified_cache: list,
                               branch_indices: List[int]) -> None:
    cache_len = unified_cache[0][0].shape[2]
    for layer_idx, (k_ref, v_ref) in enumerate(refreshed_cache):
        assert k_ref.shape[2] == cache_len, (
            f"refresh cache length {k_ref.shape[2]} != expected {cache_len}")
        k_cache, v_cache = unified_cache[layer_idx]
        for local_idx, branch_idx in enumerate(branch_indices):
            k_cache[branch_idx].copy_(k_ref[local_idx])
            v_cache[branch_idx].copy_(v_ref[local_idx])


def batched_full_seq_forward(
    states: List[FusedState],
    x_batch: torch.Tensor,
    model,
    temperature: float,
    remasking: str,
    mask_id: int,
    threshold: float,
    eos_token_id: int,
    stats_top: dict,
    Lp: int,
    gen_length: int,
    is_refresh: bool = False,
    unified_cache: Optional[list] = None,
) -> Optional[list]:
    """Full-sequence forward with broadcast init and row-owned refresh.

    is_refresh=False (init):
      - Full forward on branch 0's full sequence; builds KV from scratch.
      - Broadcasts the identical initial KV to all branch rows.
    is_refresh=True:
      - Full forward on every active branch sequence.
      - Copies each refreshed KV row back only to its original branch.
      - Applies each branch's own logits to its current block.

    Returns the new/updated unified_cache, or None if all states are done.
    """
    active = [s for s in states if not s.done]
    if not active:
        return None

    N = len(states)
    timing_device = x_batch.device

    if is_refresh:
        assert unified_cache is not None
        branch_indices = [s.branch_idx for s in active]
        branch_index_tensor = torch.tensor(
            branch_indices, dtype=torch.long, device=x_batch.device)
        key_fwd, key_kv, key_post = 'refresh_model_time', 'refresh_kv_broadcast_time', 'refresh_post_time'
        log_label = 'refresh'
    else:
        branch_indices = [states[0].branch_idx]
        key_fwd, key_kv, key_post = 'init_fwd_model_time', 'kv_broadcast_time', 'init_post_time'
        log_label = 'full_seq'

    logger.debug("batched_%s_fwd: branches=%s active=%s",
                 log_label, branch_indices, [s.branch_idx for s in active])

    # ── model forward ────────────────────────────────────────────────────────
    t_fwd = _timed_now(timing_device)
    if is_refresh:
        x_full = x_batch.index_select(0, branch_index_tensor)
        out = model(x_full, use_cache=True, output_hidden_states=True)
    else:
        out = model(x_batch[0:1], use_cache=True, output_hidden_states=True)
    dt = _timed_elapsed(t_fwd, timing_device)
    stats_top[key_fwd] += dt

    # ── KV broadcast / refresh write-back ───────────────────────────────────
    t_kv = _timed_now(timing_device)
    if is_refresh:
        _copy_refreshed_cache_rows(out.past_key_values, unified_cache, branch_indices)
        new_cache = unified_cache
    else:
        new_cache = _broadcast_kv(out.past_key_values, N)
    stats_top[key_kv] += _timed_elapsed(t_kv, timing_device)

    # ── post: update metadata and commit current block per branch ────────────
    t_post = _timed_now(timing_device)
    per_branch_dt = dt / len(active)
    for local_idx, state in enumerate(active):
        state.nfe_full += 1
        state.total_forward_time += per_branch_dt

        logits_full = out.logits[local_idx:local_idx + 1] if is_refresh else out.logits
        probs_full = F.softmax(logits_full.float(), dim=-1).squeeze(0)
        _update_token_prob_map(state, probs_full, offset=0)

        before_decoded = state.tokens_decoded
        logits_blk = logits_full[:, state.block_start:state.block_end]
        _unmask_inplace(state, x_batch, logits_blk, state.block_start, state.block_end,
                        mask_id, temperature, remasking, threshold)
        _check_eos_in_block(state, x_batch, logits_blk, state.block_start, state.block_end,
                            mask_id, eos_token_id)
        decoded_now = int(state.tokens_decoded - before_decoded)

        logger.debug("  %s branch %d (bs=%d): nfe_full=%d decoded=%d "
                     "decoded_now=%d block=[%d,%d)",
                     log_label, state.branch_idx, state.block_size, state.nfe_full,
                     state.tokens_decoded, decoded_now,
                     state.block_start, state.block_end)

    stats_top[key_post] += _timed_elapsed(t_post, timing_device)
    if is_refresh:
        stats_top['refresh_count'] += 1
    return new_cache


def batched_block_denoise(
    states: List[FusedState],
    x_batch: torch.Tensor,
    unified_cache: list,
    model,
    temperature: float,
    remasking: str,
    mask_id: int,
    threshold: float,
    eos_token_id: int,
    stats_top: dict,
) -> None:
    """Batched block-denoise forward with KV cache.

    Sends ALL N branches in one model call (done branches are no-ops via
    replace_position=False). Blocks are padded to max_q_len. The unified_cache
    is modified IN-PLACE by the model's replace_position loop.
    """
    active = [s for s in states if not s.done]
    if not active:
        return

    # ── pre: padding / position_ids setup ───────────────────────────────
    device = x_batch.device
    t_pre = _timed_now(device)
    N = len(states)
    L = x_batch.shape[1]
    max_q_len = max(s.block_end - s.block_start for s in active)

    x_padded = torch.full((N, max_q_len), mask_id, dtype=torch.long, device=device)
    replace_position = torch.zeros((N, L), dtype=torch.bool, device=device)
    position_ids = torch.zeros((N, max_q_len), dtype=torch.long, device=device)

    for state in states:
        bi = state.branch_idx
        if state.done:
            continue  # x_padded[bi]=mask_id, replace_position[bi]=False — no-op
        length = state.block_end - state.block_start
        x_padded[bi, :length] = x_batch[bi, state.block_start:state.block_end]
        replace_position[bi, state.block_start:state.block_end] = True
        position_ids[bi, :length] = torch.arange(
            state.block_start, state.block_end, device=device)
        # Padding positions: never written to cache (replace_position=False), but
        # must be a valid index for the RoPE sin/cos table (0..L-1).
        position_ids[bi, length:] = state.block_end - 1

    global _FUSED_POSITION_IDS, _VARLEN_Q_LENS
    # Position ids are always required: RoPE needs the absolute slot indices
    # whether or not we're packing.
    _FUSED_POSITION_IDS = position_ids

    # Varlen packing only pays off when there is real padding to eliminate.
    # With N=1 (or all q_lens equal, e.g. BLOCK_SIZES=[128]) the padded tensor
    # is already compact — the block pack/unpack and the SDPA patch's K/V
    # rematerialization become pure overhead. Leave _VARLEN_Q_LENS=None in
    # that case so the block forward falls through to the original path and
    # SDPA uses flash_attn_func on the padded batch directly.
    q_lens = [
        (s.block_end - s.block_start) if not s.done else 1
        for s in states
    ]
    varlen_beneficial = len(q_lens) > 1 and max(q_lens) > min(q_lens)
    _VARLEN_Q_LENS = q_lens if varlen_beneficial else None

    logger.debug("batched_block_denoise: active=%s max_q_len=%d total_q=%d varlen=%s",
                 [s.branch_idx for s in active], max_q_len, sum(q_lens),
                 varlen_beneficial)
    stats_top['block_denoise_pre_time'] += _timed_elapsed(t_pre, device)

    # ── model fwd ────────────────────────────────────────────────────────
    t_fwd = _timed_now(device)
    if _PROFILE_LAYERS:
        _reset_layer_timing()
    try:
        out = model(x_padded, past_key_values=unified_cache, use_cache=True,
                    replace_position=replace_position, output_hidden_states=True)
    finally:
        _FUSED_POSITION_IDS = None  # always clear, even on exception
        _VARLEN_Q_LENS = None
    dt = _timed_elapsed(t_fwd, device)
    if _PROFILE_LAYERS:
        _flush_layer_timing(stats_top)   # reuses elapsed timer sync above

    # unified_cache modified IN-PLACE by replace_position path — no scatter needed
    stats_top['batched_block_denoise_count'] += 1
    stats_top['batched_block_denoise_wall_time'] += dt
    stats_top['block_denoise_model_time'] += dt

    # ── post: logit / unmask per branch ──────────────────────────────────
    t_post = _timed_now(device)
    per_branch_dt = dt / len(active)

    for state in active:
        bi = state.branch_idx
        length = state.block_end - state.block_start
        logits_j = out.logits[bi:bi + 1, :length]  # (1, block_size, vocab)

        state.nfe_block += 1
        state.total_forward_time += per_branch_dt

        probs_blk = F.softmax(logits_j.float(), dim=-1).squeeze(0)  # (block_size, vocab)
        _update_token_prob_map(state, probs_blk, offset=state.block_start)

        _unmask_inplace(state, x_batch, logits_j, state.block_start, state.block_end,
                        mask_id, temperature, remasking, threshold)
        _check_eos_in_block(state, x_batch, logits_j, state.block_start, state.block_end,
                            mask_id, eos_token_id)

        logger.debug("  block_denoise branch %d (bs=%d): nfe_block=%d decoded=%d eos=%s",
                     bi, state.block_size, state.nfe_block, state.tokens_decoded, state.hit_eos)

    stats_top['block_denoise_post_time'] += _timed_elapsed(t_post, device)


# ── Shell refresh ────────────────────────────────────────────────────────────

def _measure_kv_shell_divergence(
    states: List[FusedState],
    unified_cache: list,
    stats_top: dict,
) -> Optional[Tuple[FusedState, float, float, float]]:
    """Return RMS chord shell divergence for active KV rows.

    For each layer's K and V cache tensor, each active branch row is flattened
    and normalized to a unit vector z_i. We then compute the Euclidean mean mu,
    per-branch squared radius ||z_i - mu||^2, and rho^2 = 1 - ||mu||^2.
    Values are averaged across all K/V tensors.

    Returns (outlier_state, rmax2, rho2, bound), where bound = 2 * rho2.
    Returns None when fewer than two active branches or no cache tensors are
    available.
    """
    timing_device = unified_cache[0][0].device if unified_cache else None
    t = _timed_now(timing_device)
    stats_top['refresh_similarity_count'] = stats_top.get('refresh_similarity_count', 0) + 1
    try:
        active = [s for s in states if not s.done]
        if len(active) < 2 or unified_cache is None:
            return None

        device = unified_cache[0][0].device
        active_indices = torch.tensor(
            [s.branch_idx for s in active], dtype=torch.long, device=device)
        n_active = len(active)
        r2_accum = torch.zeros(n_active, device=device, dtype=torch.float32)
        rho2_accum = torch.zeros((), device=device, dtype=torch.float32)
        num_tensors = 0

        for k_cache, v_cache in unified_cache:
            for cache_tensor in (k_cache, v_cache):
                rows = cache_tensor.index_select(0, active_indices)
                rows = rows.float().reshape(n_active, -1)
                rows = F.normalize(rows, p=2, dim=1)

                mu = rows.mean(dim=0)
                centered = rows - mu
                r2_accum += centered.square().sum(dim=1)
                rho2_accum += torch.clamp(1.0 - mu.square().sum(), min=0.0)
                num_tensors += 1

        if num_tensors == 0:
            return None

        avg_r2 = r2_accum / num_tensors
        avg_rho2 = rho2_accum / num_tensors
        rmax2_tensor, outlier_idx_tensor = torch.max(avg_r2, dim=0)
        outlier_idx = int(outlier_idx_tensor.item())
        rmax2 = float(rmax2_tensor.item())
        rho2 = float(avg_rho2.item())
        bound = 2.0 * rho2
        outlier = active[outlier_idx]

        stats_top['refresh_last_shell_rmax2'] = rmax2
        stats_top['refresh_last_shell_rho2'] = rho2
        stats_top['refresh_last_shell_bound'] = bound
        stats_top['refresh_last_outlier_branch'] = outlier.branch_idx
        return outlier, rmax2, rho2, bound
    finally:
        stats_top['refresh_similarity_time'] = (
            stats_top.get('refresh_similarity_time', 0.0) +
            _timed_elapsed(t, timing_device))


def _refresh_active_branches_full_sequence(
    states: List[FusedState],
    x_batch: torch.Tensor,
    unified_cache: list,
    model,
    temperature: float,
    remasking: str,
    mask_id: int,
    threshold: float,
    eos_token_id: int,
    stats_top: dict,
    Lp: int,
    gen_length: int,
) -> None:
    """Refresh active branches with row-owned full-sequence forwards."""
    batched_full_seq_forward(
        states, x_batch, model, temperature, remasking,
        mask_id, threshold, eos_token_id, stats_top, Lp, gen_length,
        is_refresh=True, unified_cache=unified_cache)


def should_use_full_batch_forward(
    states: List[FusedState],
    unified_cache: list,
    stats_top: dict,
    Lp: int,
    gen_length: int,
) -> bool:
    """Return True when KV shell divergence should switch to full-batch forward."""
    measurement = _measure_kv_shell_divergence(states, unified_cache, stats_top)
    if measurement is None:
        stats_top['refresh_skipped_count'] += 1
        return False

    outlier, rmax2, rho2, bound = measurement
    if rmax2 <= bound:
        stats_top['refresh_skipped_count'] += 1
        logger.debug("full_batch_forward skipped: outlier=%d rmax2=%.6f rho2=%.6f bound=%.6f",
                     outlier.branch_idx, rmax2, rho2, bound)
        return False

    logger.debug("full_batch_forward enabled: outlier=%d rmax2=%.6f rho2=%.6f bound=%.6f",
                 outlier.branch_idx, rmax2, rho2, bound)
    return True


# ── Policy ───────────────────────────────────────────────────────────────────

def _sync_branch_to_leader(s: FusedState, leader: FusedState,
                            x_batch: torch.Tensor,
                            unified_cache: Optional[list],
                            Lp: int, gen_length: int, mask_id: int) -> None:
    """Copy leader's sequence + KV into branch s, then realign s's block window."""
    old_decoded = s.tokens_decoded

    x_batch[s.branch_idx] = x_batch[leader.branch_idx]

    if unified_cache is not None:
        for layer_kv in unified_cache:
            layer_kv[0][s.branch_idx] = layer_kv[0][leader.branch_idx].clone()
            layer_kv[1][s.branch_idx] = layer_kv[1][leader.branch_idx].clone()

    s.done = leader.done
    s.hit_eos = leader.hit_eos
    s.eos_position = leader.eos_position
    s.eos_prob = leader.eos_prob
    mask_pos = (s.x[:, Lp:] == mask_id).nonzero(as_tuple=True)[1]
    if len(mask_pos) == 0:
        s.done = True
        s.progress = gen_length
        s.block_start = Lp + gen_length
        s.block_end = Lp + gen_length
    else:
        first_mask = mask_pos[0].item()
        block_idx = first_mask // s.block_size
        s.block_start = Lp + block_idx * s.block_size
        s.block_end = min(s.block_start + s.block_size, Lp + gen_length)
        s.progress = block_idx * s.block_size

    s.tokens_copied_from += max(leader.tokens_decoded - old_decoded, 0)
    s.tokens_decoded = leader.tokens_decoded

def _are_compatible(dst: FusedState, src_list: List[FusedState],
                    mask_id: int, Lp: int) -> List[FusedState]:
    """Return sources compatible with dst (matching decoded tokens in overlap)."""
    gen_dst = dst.x[0, Lp:]
    compatible = []
    for src in src_list:
        gen_src = src.x[0, Lp:]
        both = (gen_dst != mask_id) & (gen_src != mask_id)
        if not both.any() or (gen_dst[both] == gen_src[both]).all().item():
            compatible.append(src)
    return compatible


def _merge_states(dst: FusedState, src_list: List[FusedState],
                  Lp: int, gen_length: int, mask_id: int,
                  conf_threshold: float = CONF_THRESHOLD) -> int:
    """Merge best-confidence tokens from compatible sources into dst.

    Sources may have already advanced past useful decoded tokens, so scan the
    whole generated prefix covered by compatible sources, not just their current
    block windows.
    """
    tpm = dst.token_prob_map or None
    if tpm is None:
        return 0

    active_sources = [src for src in src_list if not src.done]
    if not active_sources:
        return 0

    changed = False
    tokens_filled = 0
    scan_end = min(Lp + gen_length, len(tpm),
                   max(src.block_end for src in active_sources))

    for pos in range(Lp, scan_end):
        if dst.x[0, pos].item() != mask_id:
            continue
        pos_probs = tpm[pos]
        if not pos_probs:
            continue

        best_tok, best_prob = mask_id, 0.0
        for s in active_sources:
            tok = s.x[0, pos].item()
            if tok == mask_id:
                continue
            prob = pos_probs.get(tok, 0.0)
            if prob > best_prob:
                best_prob, best_tok = prob, tok

        if best_prob > conf_threshold and best_tok != mask_id:
            dst.x[0, pos] = best_tok
            changed = True
            tokens_filled += 1

    if changed and not dst.done:
        if (dst.x[0, dst.block_start:dst.block_end] == mask_id).sum() == 0:
            update_block_window(dst, Lp, gen_length, mask_id)

    return tokens_filled


class FusedCrossStateBestPolicy:
    """Cross-state best-token merge + hard-sync policy on x_batch + KV cache."""

    def __init__(self, Lp: int, gen_length: int, mask_id: int,
                 sync_threshold: int = SYNC_THRESHOLD,
                 conf_threshold: float = CONF_THRESHOLD):
        self._Lp = Lp
        self._gen_length = gen_length
        self._mask_id = mask_id
        self._sync_threshold = sync_threshold
        self._conf_threshold = conf_threshold

    def compare_and_copy(self, states: List[FusedState],
                         x_batch: torch.Tensor,
                         unified_cache: Optional[list] = None) -> int:
        """Run merge + sync.

        unified_cache: if provided, also copies leader's KV rows into lagging
        branches. Each element is a (key, value) tuple per layer,
        shape (N, n_kv_heads, L, head_dim).

        Returns the number of branches hard-synced to the leader.
        """
        leader_idx = max(range(len(states)), key=lambda k: states[k].tokens_decoded)
        leader = states[leader_idx]
        if leader.tokens_decoded == 0:
            return 0

        eligible = [i for i in range(len(states)) if not states[i].done]
        order = sorted(eligible, key=lambda k: states[k].block_end, reverse=True)

        # Merge phase
        if not all(states[i].block_end <= self._Lp for i in order):
            for i_rank in range(len(order)):
                dst = states[order[i_rank]]
                src_list = [states[order[j]] for j in range(i_rank + 1, len(order))]
                compatible = _are_compatible(dst, src_list, self._mask_id, self._Lp)
                if compatible:
                    filled = _merge_states(dst, compatible, self._Lp, self._gen_length,
                                           self._mask_id, self._conf_threshold)
                    dst.tokens_merged += filled
                    dst.tokens_decoded += filled
                    if filled:
                        logger.debug("merge: dst=%d filled=%d", dst.branch_idx, filled)

        leader_idx = max(range(len(states)), key=lambda k: states[k].tokens_decoded)
        leader = states[leader_idx]

        # Sync phase — hard-copy leader into any state too far behind
        synced = 0
        for i, s in enumerate(states):
            if i == leader_idx or s.done:
                continue
            if leader.tokens_decoded - s.tokens_decoded > self._sync_threshold:
                gained = leader.tokens_decoded - s.tokens_decoded
                _sync_branch_to_leader(s, leader, x_batch, unified_cache,
                                       self._Lp, self._gen_length, self._mask_id)
                synced += 1
                logger.debug("sync: leader=%d -> lag=%d gained=%d (cache=%s)",
                             leader.branch_idx, s.branch_idx, gained,
                             unified_cache is not None)
        return synced

    def hard_sync_all(self, states: List[FusedState],
                      x_batch: torch.Tensor,
                      unified_cache: Optional[list] = None,
                      leader: Optional[FusedState] = None) -> int:
        """Unconditionally sync every non-leader non-done branch to the leader.

        Unlike compare_and_copy's sync phase (gated by sync_threshold), this
        forces every lagging branch onto the leader's sequence + KV rows. Used
        before a periodic refresh so a single batch=1 forward suffices.

        Returns the number of branches actually synced.
        """
        if leader is None:
            leader_idx = max(range(len(states)), key=lambda k: states[k].tokens_decoded)
            leader = states[leader_idx]
        else:
            leader_idx = leader.branch_idx
        if leader.tokens_decoded == 0:
            return 0

        synced = 0
        for i, s in enumerate(states):
            if i == leader_idx or s is leader or s.done:
                continue
            gained = leader.tokens_decoded - s.tokens_decoded
            _sync_branch_to_leader(s, leader, x_batch, unified_cache,
                                   self._Lp, self._gen_length, self._mask_id)
            synced += 1
            logger.debug("hard_sync_all: leader=%d -> lag=%d gained=%d",
                         leader.branch_idx, s.branch_idx, gained)
        return synced


# ── Result ───────────────────────────────────────────────────────────────────

def build_result_v2(winner: FusedState, all_states: List[FusedState],
                    x_batch: torch.Tensor, start_time: float, stats_top: dict):
    """Build (final_x, total_nfe, stats_dict) for V2."""
    total_wall = _timed_elapsed(start_time, x_batch.device)
    init_full_nfe = 1
    block_denoise_nfe = stats_top.get('batched_block_denoise_count', 0)
    refresh_nfe = stats_top.get('refresh_count', 0)
    total_nfe = init_full_nfe + block_denoise_nfe + refresh_nfe
    winner_branch_nfe = winner.nfe_block + winner.nfe_full
    stats = {
        'final_block_size': winner.block_size,
        'final_eos_position': winner.eos_position,
        'total_nfe': total_nfe,
        'init_full_nfe': init_full_nfe,
        'block_denoise_nfe': block_denoise_nfe,
        'refresh_nfe': refresh_nfe,
        'winner_branch_nfe': winner_branch_nfe,
        'total_wall_time': total_wall,
        'total_tokens_generated': winner.progress,
        'tokens_per_second': winner.progress / total_wall if total_wall > 0 else 0.0,
        'nfe_per_block_size': {
            s.block_size: {
                'nfe_block': s.nfe_block,
                'nfe_full': s.nfe_full,
                'equivalent_nfe': s.nfe_block + s.nfe_full * 1.5,
                'tok_per_second': (s.tokens_decoded / s.total_forward_time
                                   if s.total_forward_time > 0 else 0.0),
            } for s in all_states
        },
        'progress_per_block_size': {s.block_size: s.progress for s in all_states},
        'tokens_decoded_per_block_size': {s.block_size: s.tokens_decoded for s in all_states},
        'tokens_merged_per_block_size': {s.block_size: s.tokens_merged for s in all_states},
        'tokens_copied_from_per_block_size': {s.block_size: s.tokens_copied_from for s in all_states},
    }
    stats.update(stats_top)
    return x_batch[winner.branch_idx:winner.branch_idx + 1], total_nfe, stats


# ── Loop helpers ─────────────────────────────────────────────────────────────

def _advance_all_blocks(states: List[FusedState], x_batch: torch.Tensor,
                        Lp: int, gen_length: int, mask_id: int,
                        stats_top: dict) -> None:
    """Advance block window for every non-done state whose block is fully decoded."""
    t = _timed_now(x_batch.device)
    for s in states:
        if not s.done and (x_batch[s.branch_idx, s.block_start:s.block_end] == mask_id).sum() == 0:
            update_block_window(s, Lp, gen_length, mask_id)
    stats_top['advance_block_time'] += _timed_elapsed(t, x_batch.device)


def _check_eos_ready(states: List[FusedState], x_batch: torch.Tensor,
                     step: int, start_time: float, stats_top: dict,
                     mask_id: int, label: str):
    """Return build_result_v2 tuple if any EOS-ready branch exists, else None."""
    eos_ready = [
        s for s in states
        if s.hit_eos
        and (x_batch[s.branch_idx, s.block_start:s.block_end] == mask_id).sum() == 0
    ]
    if not eos_ready:
        return None
    winner = max(eos_ready, key=lambda s: s.eos_prob)
    logger.info("EOS at %s step %d — winner branch=%d bs=%d",
                label, step, winner.branch_idx, winner.block_size)
    return build_result_v2(winner, states, x_batch, start_time, stats_top)


# ── Init ─────────────────────────────────────────────────────────────────────

def _init_states(x_batch: torch.Tensor, block_sizes: List[int], Lp: int) -> List[FusedState]:
    states = []
    for i, bs in enumerate(block_sizes):
        state = FusedState(
            branch_idx=i,
            block_size=bs,
            x=x_batch[i:i + 1],
            progress=0,
            block_start=Lp,
            block_end=Lp + bs,
            done=False,
            hit_eos=False,
            eos_position=None,
            eos_prob=0.0,
            nfe_full=0,
            total_forward_time=0.0,
        )
        states.append(state)
    return states


# ── Core loop ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_fused_block_batching_v2(
    model,
    prompt: torch.Tensor,
    gen_length: int,
    block_sizes: List[int],
    temperature: float = 0.,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
    threshold: float = 0.9,
    refresh_block_size: int = REFRESH_BLOCK_SIZE,
) -> Tuple:
    """Fused batched block generation — V2 (varlen block-denoise + KV cache).

    Applies both monkey-patches (RoPE + varlen SDPA) once, then:
      1. Initial full-seq forward — ONE forward (batch=1), KV expanded to N branches
      2. Main loop: batched varlen block-denoise every step using the cached K/V
      3. When a KV row leaves the RMS chord shell, switch to row-owned
         full-batch full-sequence forwards for that step.

    Returns:
        (final_x, total_nfe, stats_dict)

    refresh_block_size is retained for call compatibility; the full-batch
    forward switch is driven by RMS chord shell divergence in this variant.
    """
    _apply_rope_patch()          # idempotent
    _apply_sdpa_patch()          # idempotent
    _apply_llama_block_patch()   # idempotent — MLP token packing

    assert prompt.shape[0] == 1, "prompt batch size must be 1"
    block_sizes = sorted(block_sizes)
    N = len(block_sizes)
    Lp = prompt.shape[1]
    for bs in block_sizes:
        assert gen_length % bs == 0, f"gen_length {gen_length} not divisible by {bs}"

    eos_token_id = model.config.eos_token_id

    x_batch = torch.full((N, Lp + gen_length), mask_id,
                         dtype=torch.long, device=prompt.device)
    x_batch[:, :Lp] = prompt

    states = _init_states(x_batch, block_sizes, Lp)
    policy = FusedCrossStateBestPolicy(Lp=Lp, gen_length=gen_length, mask_id=mask_id)
    stats_top: dict = {
        'batched_block_denoise_count': 0,
        'batched_block_denoise_wall_time': 0.0,
        'init_fwd_model_time': 0.0,
        'kv_broadcast_time': 0.0,
        'init_post_time': 0.0,
        'block_denoise_pre_time': 0.0,
        'block_denoise_model_time': 0.0,
        'block_denoise_post_time': 0.0,
        'policy_time': 0.0,
        'advance_block_time': 0.0,
        'refresh_count': 0,
        'refresh_sync_time': 0.0,
        'refresh_model_time': 0.0,
        'refresh_kv_broadcast_time': 0.0,
        'refresh_post_time': 0.0,
        'refresh_similarity_time': 0.0,
        'refresh_similarity_count': 0,
        'refresh_skipped_count': 0,
        'refresh_last_min_similarity': 1.0,
        'refresh_last_max_distance': 0.0,
        'refresh_last_shell_rmax2': 0.0,
        'refresh_last_shell_rho2': 0.0,
        'refresh_last_shell_bound': 0.0,
        'refresh_last_outlier_branch': -1,
        'policy_sync_count': 0,
    }
    start_time = _timed_now(prompt.device)

    logger.info("run_fused_block_batching_v2: N=%d block_sizes=%s gen_length=%d Lp=%d",
                N, block_sizes, gen_length, Lp)

    # ── Initial full-seq forward (all branches, with KV cache) ──────────
    unified_cache = batched_full_seq_forward(
        states, x_batch, model, temperature, remasking,
        mask_id, threshold, eos_token_id, stats_top, Lp, gen_length)

    t_rs = _timed_now(x_batch.device)
    policy.hard_sync_all(states, x_batch, unified_cache)
    stats_top['refresh_sync_time'] += _timed_elapsed(t_rs, x_batch.device)

    _advance_all_blocks(states, x_batch, Lp, gen_length, mask_id, stats_top)

    t_pol = _timed_now(x_batch.device)
    synced = policy.compare_and_copy(states, x_batch, unified_cache)
    stats_top['policy_sync_count'] += synced
    stats_top['policy_time'] += _timed_elapsed(t_pol, x_batch.device)

    # ── Main loop ────────────────────────────────────────────────────────
    step = 0
    force_block_denoise_next = synced > 0
    while not all(s.done for s in states):
        step += 1
        logger.debug("=== v2 step %d ===", step)

        if force_block_denoise_next:
            use_full_batch_forward = False
            force_block_denoise_next = False
            logger.debug("forcing block_denoise after prior policy sync")
        else:
            use_full_batch_forward = should_use_full_batch_forward(
                states, unified_cache, stats_top, Lp, gen_length)

        if use_full_batch_forward:
            _refresh_active_branches_full_sequence(
                states, x_batch, unified_cache, model,
                temperature, remasking, mask_id, threshold, eos_token_id,
                stats_top, Lp, gen_length)

            result = _check_eos_ready(
                states, x_batch, step, start_time, stats_top,
                mask_id, 'full_batch_forward')
            if result:
                return result
            _advance_all_blocks(states, x_batch, Lp, gen_length, mask_id, stats_top)
        else:
            batched_block_denoise(states, x_batch, unified_cache, model,
                                  temperature, remasking, mask_id, threshold,
                                  eos_token_id, stats_top)

            result = _check_eos_ready(
                states, x_batch, step, start_time, stats_top,
                mask_id, 'block_denoise')
            if result:
                return result
            _advance_all_blocks(states, x_batch, Lp, gen_length, mask_id, stats_top)

        t_pol = _timed_now(x_batch.device)
        synced = policy.compare_and_copy(states, x_batch, unified_cache)
        stats_top['policy_sync_count'] += synced
        stats_top['policy_time'] += _timed_elapsed(t_pol, x_batch.device)
        force_block_denoise_next = synced > 0

    best = max(states, key=lambda s: s.progress)
    logger.info("Natural exit at step %d — best branch=%d bs=%d",
                step, best.branch_idx, best.block_size)
    return build_result_v2(best, states, x_batch, start_time, stats_top)

# Public alias
generate_block_batching = run_fused_block_batching_v2
