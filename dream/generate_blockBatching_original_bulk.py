"""
generate_blockBatching.py  —  Fused Block Batching for Dream (Qwen2 dLLM)


N branches (one per block size) run in parallel sharing a unified (N, L, …) KV cache.
Dream's AR logit shift (position i uses logit i-1) is applied once per forward, BEFORE
any token transfer/commit. Block denoise uses non-dual-cache mode (prefix KV, full
remaining input) to avoid stale masked-token K/V at future positions.


Entry point: generate_block_batching()
"""
from __future__ import annotations


import inspect
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


import torch


_TOKEN_PROB_THRESHOLD = 0.2
SYNC_THRESHOLD = 8
CONF_THRESHOLD = 0.5
REFRESH_BLOCK_SIZE = 32


_PROV_UNKNOWN = 0
_PROV_PROMPT = 1
_PROV_GENERATED = 2
_PROV_MERGED = 3
_PROV_COPIED = 4


_TRACE_STATS: Optional[dict] = None
_PROVENANCE_BATCH: Optional[torch.Tensor] = None
_TRACE_PHASE = "full"




def _new_trace_stats() -> dict:
    return {
        "generated_tokens": 0,
        "merged_tokens": 0,
        "copied_tokens": 0,
        "dropped_tokens": 0,
        "dropped_from_generated": 0,
        "dropped_from_merged": 0,
        "dropped_from_copied": 0,
        "dropped_from_prompt": 0,
        "dropped_from_other": 0,
    }




def _trace_reset() -> None:
    global _TRACE_STATS, _PROVENANCE_BATCH, _TRACE_PHASE
    _TRACE_STATS = None
    _PROVENANCE_BATCH = None
    _TRACE_PHASE = "full"




def _trace_init(N: int, L: int, Lp: int, device: torch.device) -> None:
    global _TRACE_STATS, _PROVENANCE_BATCH, _TRACE_PHASE
    _TRACE_STATS = _new_trace_stats()
    _PROVENANCE_BATCH = torch.full((N, L), _PROV_UNKNOWN, dtype=torch.int16, device=device)
    _PROVENANCE_BATCH[:, :Lp] = _PROV_PROMPT
    _TRACE_PHASE = "full"




def _trace_set_phase(phase: str) -> None:
    global _TRACE_PHASE
    _TRACE_PHASE = phase




def _trace_add(key: str, delta: int) -> None:
    if _TRACE_STATS is None or delta <= 0:
        return
    _TRACE_STATS[key] = int(_TRACE_STATS.get(key, 0)) + int(delta)




def _drop_bucket_from_prov(prov_id: int) -> str:
    if prov_id == _PROV_GENERATED:
        return "generated"
    if prov_id == _PROV_MERGED:
        return "merged"
    if prov_id == _PROV_COPIED:
        return "copied"
    if prov_id == _PROV_PROMPT:
        return "prompt"
    return "other"




def _summarize_drop_origins(prov_values: torch.Tensor) -> Dict[str, int]:
    counts = {
        "generated": 0,
        "merged": 0,
        "copied": 0,
        "prompt": 0,
        "other": 0,
    }
    if prov_values.numel() == 0:
        return counts
    for pid in prov_values.detach().cpu().tolist():
        counts[_drop_bucket_from_prov(int(pid))] += 1
    return counts




def _trace_add_drop_counts(origin_counts: Dict[str, int]) -> None:
    if _TRACE_STATS is None:
        return
    dropped_total = int(sum(origin_counts.values()))
    if dropped_total <= 0:
        return
    _trace_add("dropped_tokens", dropped_total)
    _trace_add("dropped_from_generated", int(origin_counts.get("generated", 0)))
    _trace_add("dropped_from_merged", int(origin_counts.get("merged", 0)))
    _trace_add("dropped_from_copied", int(origin_counts.get("copied", 0)))
    _trace_add("dropped_from_prompt", int(origin_counts.get("prompt", 0)))
    _trace_add("dropped_from_other", int(origin_counts.get("other", 0)))




@dataclass
class FusedState:
    branch_idx: int
    block_size: int
    block_start: int
    block_end: int
    block_idx: int = 0
    cache_row: int = 0
    progress: int = 0
    tokens_decoded: int = 0
    decoded_since_refresh: int = 0
    done: bool = False
    hit_eos: bool = False
    eos_position: int = -1
    eos_prob: float = 0.0
    token_prob_map: Optional[List[Dict]] = field(default=None, repr=False)
    cached_prob: Optional[Dict[int, float]] = field(default=None, repr=False)

# ── KV helpers ────────────────────────────────────────────────────────────────


def _broadcast_kv(past_key_values_1, N: int) -> list:
    """Broadcast single-batch KV (1, L, …) to N-batch (N, L, …) via expand+clone."""
    return [(k.expand(N, *k.shape[1:]).clone(),
             v.expand(N, *v.shape[1:]).clone())
            for k, v in past_key_values_1]




def _trim_kv(past_key_values, end: int) -> list:
    """Trim each (K, V) to the first `end` sequence positions."""
    return [(k[:, :end, :], v[:, :end, :]) for k, v in past_key_values]




def _sync_kv_row(unified_kv: list, dst_idx: int, src_idx: int) -> None:
    """In-place copy row src_idx → row dst_idx within the unified (N, L, …) cache."""
    for k, v in unified_kv:
        k[dst_idx].copy_(k[src_idx])
        v[dst_idx].copy_(v[src_idx])




def _shift_logits(logits: torch.Tensor) -> torch.Tensor:
    """Dream's AR shift: position i uses logit i-1 (predicts position i from predecessor)."""
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)




# ── Monkey-patches for batched dual_cache ─────────────────────────────────────


_PATCHES_APPLIED = False
_BATCHED_POSITION_IDS: Optional[torch.Tensor] = None  # (N, max_q) set during batched forward




def _apply_batched_dual_cache_patches(model) -> None:
    """Patch DreamSdpaAttention and apply_rotary_pos_emb to support batched
    dual_cache with different block sizes per row.  Applied once per process."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return


    attn   = model.model.layers[0].self_attn
    mod    = inspect.getmodule(type(attn))
    AttnCls = type(attn)


    # ── Patch 1: apply_rotary_pos_emb — per-row position_ids via global ─────────
    # We do NOT pass position_ids to the model (which would give wrong cos/sin from
    # rotary_emb). Instead we store them in _BATCHED_POSITION_IDS and gather from
    # the full L-length cos table that rotary_emb computes from the default positions.
    def _patched_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, **kwargs):
        cos = cos.unsqueeze(unsqueeze_dim)    # (1, 1, L, head_dim) — full table
        sin = sin.unsqueeze(unsqueeze_dim)


        if _BATCHED_POSITION_IDS is not None:
            # Per-row gather for Q: absolute positions into the full L-length table
            # cos[0, 0] shape: (L, head_dim) — not indexed yet, full range available
            cos_2d, sin_2d = cos[0, 0], sin[0, 0]    # (L, head_dim) views, no copy
            q_embed = torch.empty_like(q)
            for b in range(q.shape[0]):
                pos_b      = _BATCHED_POSITION_IDS[b]   # (max_q,) ABSOLUTE positions
                cos_b      = cos_2d[pos_b]               # (max_q, head_dim)
                sin_b      = sin_2d[pos_b]
                q_embed[b] = q[b] * cos_b + mod.rotate_half(q[b]) * sin_b
        else:
            # Standard fast path
            query_len, key_len = q.shape[-2], k.shape[-2]
            q_embed = (q * cos[:, :, key_len - query_len:key_len, :]) + \
                      (mod.rotate_half(q) * sin[:, :, key_len - query_len:key_len, :])


        k_embed = (k * cos) + (mod.rotate_half(k) * sin)    # K uses full L positions
        return q_embed, k_embed


    mod.apply_rotary_pos_emb = _patched_rope


    # ── Patch 2: DreamSdpaAttention.forward — per-row dual_cache scatter ───────
    _orig_fwd = AttnCls.forward


    def _patched_fwd(self, hidden_states, attention_mask=None, position_ids=None,
                     past_key_value=None, output_attentions=False, use_cache=False,
                     cache_position=None, position_embeddings=None,
                     replace_position=None, dual_cache=False, **kwargs):
        if output_attentions:
            return super(AttnCls, self).forward(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache)


        import torch.nn.functional as F
        bsz, q_len, _ = hidden_states.size()


        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)


        if past_key_value is not None:
            if dual_cache:
                past_key, past_value = past_key_value
                B = past_key.shape[0]
                if B == 1:
                    # Original single-batch fast path
                    ri = replace_position.nonzero(as_tuple=True)[1]
                    past_key[:, ri]   = key_states
                    past_value[:, ri] = value_states
                else:
                    # Per-row scatter for different block sizes across branches
                    for b in range(B):
                        idx_b = replace_position[b].nonzero(as_tuple=True)[0]
                        n     = idx_b.shape[0]
                        past_key[b, idx_b]   = key_states[b, :n]
                        past_value[b, idx_b] = value_states[b, :n]
                key_states   = past_key
                value_states = past_value
            else:
                past_key, past_value = past_key_value
                key_states   = torch.cat([past_key,   key_states],   dim=-2)
                value_states = torch.cat([past_value, value_states], dim=-2)


        past_key_value = (key_states, value_states) if use_cache else None


        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings


        # position_ids (if 2D) handled per-row by patched RoPE
        query_states, key_states = mod.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids=position_ids)


        key_states   = mod.repeat_kv(key_states,   self.num_key_value_groups)
        value_states = mod.repeat_kv(value_states, self.num_key_value_groups)


        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states   = key_states.contiguous()
            value_states = value_states.contiguous()


        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )


        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


    AttnCls.forward = _patched_fwd
    _PATCHES_APPLIED = True




# ── Block window advancement ──────────────────────────────────────────────────


def _first_eos_rel(gen: torch.Tensor, eos_token_id: Optional[int]) -> Optional[int]:
    if eos_token_id is None or eos_token_id < 0:
        return None
    eos_pos = (gen == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) == 0:
        return None
    return int(eos_pos[0].item())


def _counted_end_rel(gen: torch.Tensor, eos_token_id: Optional[int]) -> int:
    first_eos = _first_eos_rel(gen, eos_token_id)
    return gen.shape[0] if first_eos is None else first_eos + 1


def _decode_end_rel(gen: torch.Tensor, eos_token_id: Optional[int]) -> int:
    first_eos = _first_eos_rel(gen, eos_token_id)
    return gen.shape[0] if first_eos is None else first_eos


def _first_eos_abs_for_row(
    x: torch.Tensor,
    Lp: int,
    gen_length: int,
    eos_token_id: Optional[int],
) -> Optional[int]:
    first_eos = _first_eos_rel(x[0, Lp:Lp + gen_length], eos_token_id)
    return None if first_eos is None else Lp + first_eos


def _refresh_branch_accounting(
    state: FusedState,
    x: torch.Tensor,
    Lp: int,
    gen_length: int,
    mask_id: int,
    eos_token_id: Optional[int] = None,
) -> Tuple[int, Optional[int], torch.Tensor]:
    gen = x[0, Lp:Lp + gen_length]
    counted_end = _counted_end_rel(gen, eos_token_id)
    decode_end = _decode_end_rel(gen, eos_token_id)
    state.tokens_decoded = int((gen[:counted_end] != mask_id).sum().item())
    mask_pos = (gen[:decode_end] == mask_id).nonzero(as_tuple=True)[0]
    return decode_end, _first_eos_rel(gen, eos_token_id), mask_pos


def _update_block_window_for_branch(
    state: FusedState,
    x: torch.Tensor,
    Lp: int,
    gen_length: int,
    mask_id: int,
    eos_token_id: Optional[int] = None,
) -> None:
    """Realign a branch to its own first unfinished pre-EOS block."""
    decode_end, first_eos, mask_pos = _refresh_branch_accounting(
        state,
        x,
        Lp,
        gen_length,
        mask_id,
        eos_token_id,
    )
    if len(mask_pos) == 0:
        state.done = True
        state.progress = gen_length if first_eos is None else first_eos + 1
        state.block_idx = state.progress // state.block_size
        state.block_start = Lp + state.progress
        state.block_end = Lp + state.progress
        return

    first_mask = int(mask_pos[0].item())
    block_idx = first_mask // state.block_size
    state.done = False
    state.block_idx = block_idx
    state.block_start = Lp + block_idx * state.block_size
    state.block_end = min(state.block_start + state.block_size, Lp + decode_end)
    state.progress = first_mask


def _advance_block(state: FusedState, x: torch.Tensor, Lp: int,
                   gen_length: int, mask_id: int,
                   eos_token_id: Optional[int] = None) -> int:
    """Update branch-local frontier and refresh counter. Returns frontier delta."""
    old_progress = state.progress
    _update_block_window_for_branch(state, x, Lp, gen_length, mask_id, eos_token_id)
    delta = max(int(state.progress) - int(old_progress), 0)
    if delta > 0:
        state.decoded_since_refresh += delta
    return delta


def _advance_to_active_window_or_done(
    state: FusedState,
    x: torch.Tensor,
    Lp: int,
    gen_length: int,
    mask_id: int,
    eos_token_id: Optional[int] = None,
) -> int:
    """Skip finished windows until the branch lands on a masked window or completes."""
    total_delta = 0
    while not state.done:
        block = x[0, state.block_start:state.block_end]
        if (block == mask_id).any():
            break
        prev_progress = state.progress
        total_delta += _advance_block(state, x, Lp, gen_length, mask_id, eos_token_id)
        if state.done or state.progress <= prev_progress:
            break
    return total_delta


def _copy_token_prob_map(token_prob_map: Optional[List[Dict]]) -> Optional[List[Dict]]:
    if token_prob_map is None:
        return None
    return [dict(pos_probs) for pos_probs in token_prob_map]




# ── Probability map ───────────────────────────────────────────────────────────


def _update_prob_map(state: FusedState, logits: torch.Tensor,
                     block_start: int, L: int,
                     token_prob_threshold: float = _TOKEN_PROB_THRESHOLD) -> None:
    """logits: (1, block_size, vocab) — block-relative. Stores at absolute positions."""
    if state.token_prob_map is None:
        state.token_prob_map = [{} for _ in range(L)]
    probs = torch.softmax(logits[0].float(), dim=-1)  # (block_size, vocab)
    mask = probs > token_prob_threshold
    pos_rel, tok_idx = mask.nonzero(as_tuple=True)
    for pr, ti in zip(pos_rel.tolist(), tok_idx.tolist()):
        state.token_prob_map[block_start + pr][ti] = probs[pr, ti].item()


def _sample_tokens_from_logits(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Always-greedy top-k decode with k=1."""
    del temperature, top_p, top_k
    logits = logits.float()
    _, sampled = torch.topk(logits, k=1, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    confidence = torch.gather(probs, -1, sampled).squeeze(-1)
    return confidence, sampled.squeeze(-1)


def _record_committed_tokens(
    state: FusedState,
    positions: torch.Tensor,
    tokens: torch.Tensor,
    confidence: torch.Tensor,
    eos_token_id: int,
    provenance: int = _PROV_GENERATED,
) -> None:
    if state.cached_prob is None:
        state.cached_prob = {}
    for pos, tok, prob in zip(positions.tolist(), tokens.tolist(), confidence.tolist()):
        state.cached_prob[int(pos)] = float(prob)
        if int(tok) == eos_token_id:
            state.hit_eos = True
            state.eos_position = int(pos)
            state.eos_prob = float(prob)
    if _PROVENANCE_BATCH is not None and positions.numel() > 0:
        _PROVENANCE_BATCH[state.branch_idx, positions] = provenance


def _refresh_eos_metadata_from_row(
    state: FusedState,
    x_batch: torch.Tensor,
    Lp: int,
    gen_length: int,
    eos_token_id: Optional[int],
) -> None:
    """Make EOS metadata match the sequence row."""
    if eos_token_id is None or eos_token_id < 0:
        state.hit_eos = False
        state.eos_position = -1
        state.eos_prob = 0.0
        return

    row = state.branch_idx if state.branch_idx < x_batch.shape[0] else 0
    gen = x_batch[row, Lp:Lp + gen_length]
    eos_pos = (gen == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) == 0:
        state.hit_eos = False
        state.eos_position = -1
        state.eos_prob = 0.0
        return

    abs_pos = Lp + int(eos_pos[0].item())
    old_prob = state.eos_prob if state.hit_eos and state.eos_position == abs_pos else 0.0
    cached_prob = 0.0
    if state.cached_prob is not None:
        cached_prob = float(state.cached_prob.get(abs_pos, 0.0))
    state.hit_eos = True
    state.eos_position = abs_pos
    state.eos_prob = cached_prob if cached_prob > 0.0 else old_prob


def _refresh_branch_metadata(
    state: FusedState,
    x_batch: torch.Tensor,
    Lp: int,
    gen_length: int,
    mask_id: int,
    eos_token_id: Optional[int],
) -> None:
    _update_block_window_for_branch(
        state,
        x_batch[state.branch_idx:state.branch_idx + 1],
        Lp,
        gen_length,
        mask_id,
        eos_token_id,
    )
    _refresh_eos_metadata_from_row(state, x_batch, Lp, gen_length, eos_token_id)


def _refresh_all_branch_metadata(
    states: List[FusedState],
    x_batch: torch.Tensor,
    Lp: int,
    gen_length: int,
    mask_id: int,
    eos_token_id: Optional[int],
) -> None:
    for state in states:
        _refresh_branch_metadata(state, x_batch, Lp, gen_length, mask_id, eos_token_id)




# ── Confidence decode ─────────────────────────────────────────────────────────


def _confidence_decode(state: FusedState, x: torch.Tensor, logits: torch.Tensor,
                        mask_id: int, threshold: float, eos_token_id: int) -> int:
    """Commit tokens to masked positions.


    Position 0 (block_start) must be pre-seeded from full-forward logits before
    calling this function.  The AR shift duplicates logit[0] into logit[1] for
    block-only forwards; with position 0 committed (not masked), the duplicate
    is harmless — _confidence_decode skips committed positions.


    Threshold-based transfer + rank-0 non-EOS fallback.
    """
    block_masked = (x[0, state.block_start:state.block_end] == mask_id)
    if not block_masked.any():
        return 0


    logits_float = logits[0].float()
    logits_float[:, mask_id] = -1e4          # never commit mask_id
    probs    = torch.softmax(logits_float, dim=-1)
    conf_all = probs.max(dim=-1).values
    pred_all = probs.argmax(dim=-1)


    masked_conf        = conf_all.masked_fill(~block_masked, float('-inf'))
    masked_conf_no_eos = masked_conf.clone()
    masked_conf_no_eos[pred_all == eos_token_id] = float('-inf')


    # Normal multi-commit path
    x_pred = x[0, state.block_start:state.block_end].clone()
    x_pred[block_masked] = pred_all[block_masked]
    transfer = block_masked & (conf_all >= threshold)
    if not transfer.any():
        if float(masked_conf_no_eos.max()) > float('-inf'):
            transfer[int(masked_conf_no_eos.argmax().item())] = True
        else:
            transfer[int(masked_conf.argmax().item())] = True

    x[0, state.block_start:state.block_end][transfer] = x_pred[transfer]
    newly = int(transfer.sum().item())
    if transfer.any():
        committed_pos = transfer.nonzero(as_tuple=True)[0] + state.block_start
        _record_committed_tokens(
            state,
            committed_pos,
            pred_all[transfer],
            conf_all[transfer],
            eos_token_id,
        )
    return newly




# ── Decode + advance ─────────────────────────────────────────────────────────


def _apply_logits(state: FusedState, x_br: torch.Tensor,
                  logits_shifted: torch.Tensor,
                  mask_id: int, threshold: float,
                  Lp: int, gen_length: int, L: int,
                  eos_token_id: int = -1) -> None:
    """Update prob map, commit tokens, advance block window."""
    _update_prob_map(state, logits_shifted, state.block_start, L)
    _confidence_decode(state, x_br, logits_shifted, mask_id, threshold, eos_token_id)
    _advance_block(state, x_br, Lp, gen_length, mask_id, eos_token_id)




def _check_eos_ready(states: List[FusedState], x_batch: torch.Tensor,
                     mask_id: int, eos_token_id: int, Lp: int,
                     start_time: float, total_nfe: int,
                     nfe_init: int = 0, nfe_block: int = 0,
                     nfe_refresh: int = 0, refresh_count: int = 0) -> Optional[tuple]:
    """Return on EOS-ready or on the first fully unmasked branch.


    Dream generates out-of-order — EOS may be committed before preceding positions.
    A branch is ready when EOS exists AND every position before it is decoded.
    Branches continue denoising only before EOS until the diffusion process fills
    preceding positions.  If no EOS appears, a branch that reaches gen_length is
    complete and can return immediately.
    """
    gen_length = x_batch.shape[1] - Lp
    _refresh_all_branch_metadata(states, x_batch, Lp, gen_length, mask_id, eos_token_id)
    ready = []
    for s in states:
        gen = x_batch[s.branch_idx, Lp:Lp + gen_length]
        eos_pos = (gen == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) == 0:
            continue
        first_eos = int(eos_pos[0].item())
        masks_before = int((gen[:first_eos] == mask_id).sum().item())
        if masks_before == 0:   # all positions before EOS decoded
            s.done = True
            ready.append(s)
    if ready:
        winner = max(ready, key=lambda s: (s.tokens_decoded, s.progress, s.block_size))
        return _build_result(
            winner, x_batch, start_time, total_nfe,
            nfe_init=nfe_init,
            nfe_block=nfe_block,
            nfe_refresh=nfe_refresh,
            refresh_count=refresh_count,
            exit_reason="eos_ready",
            all_states=states,
            mask_id=mask_id,
            eos_token_id=eos_token_id,
            Lp=Lp,
            gen_length=gen_length,
        )

    complete = []
    for s in states:
        gen = x_batch[s.branch_idx, Lp:Lp + gen_length]
        if not (gen == mask_id).any():
            s.done = True
            s.progress = gen_length
            s.tokens_decoded = gen_length
            complete.append(s)
    if not complete:
        return None

    winner = max(complete, key=lambda s: (s.tokens_decoded, s.block_size))
    return _build_result(
        winner,
        x_batch,
        start_time,
        total_nfe,
        nfe_init=nfe_init,
        nfe_block=nfe_block,
        nfe_refresh=nfe_refresh,
        refresh_count=refresh_count,
        exit_reason="branch_complete",
        all_states=states,
        mask_id=mask_id,
        eos_token_id=eos_token_id,
        Lp=Lp,
        gen_length=gen_length,
    )




def _decode_from_full_logits(
    states: List[FusedState], x_batch: torch.Tensor,
    shifted_logits: torch.Tensor,
    mask_id: int, threshold: float, L: int,
    Lp: int, gen_length: int, eos_token_id: int = -1,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    reset_refresh_counter: bool = False,
) -> None:
    """Seed one token per branch from full shifted Dream logits.

    shifted_logits[:, p, :] predicts absolute token position p.  Full-forward decode is
    deliberately restricted to the branch's current block window.
    """
    del threshold  # Full-forward seeding follows Dream's unconditional block-start seed.

    for local_idx, state in enumerate(states):
        if state.done:
            continue
        bi = state.branch_idx
        _advance_to_active_window_or_done(
            state,
            x_batch[bi:bi + 1],
            Lp,
            gen_length,
            mask_id,
            eos_token_id,
        )
        if state.done:
            if reset_refresh_counter:
                state.decoded_since_refresh = 0
            continue

        block_end = state.block_end
        eos_abs = _first_eos_abs_for_row(x_batch[bi:bi + 1], Lp, gen_length, eos_token_id)
        if eos_abs is not None:
            block_end = min(block_end, eos_abs)
        block_x = x_batch[bi, state.block_start:block_end]
        block_masked = (block_x == mask_id)
        if not block_masked.any():
            if reset_refresh_counter:
                state.decoded_since_refresh = 0
            continue

        first_mask_rel = int(block_masked.nonzero(as_tuple=True)[0][0].item())
        abs_pos = state.block_start + first_mask_rel
        logits_row = 0 if shifted_logits.shape[0] == 1 else local_idx
        logits = shifted_logits[logits_row, abs_pos:abs_pos + 1, :].clone()
        logits[:, mask_id] = -1e4

        confidence, pred = _sample_tokens_from_logits(logits, temperature, top_p, top_k)
        x_batch[bi, abs_pos] = pred[0]
        _record_committed_tokens(
            state,
            torch.tensor([abs_pos], device=x_batch.device, dtype=torch.long),
            pred,
            confidence,
            eos_token_id,
        )
        _trace_add("generated_tokens", 1)
        state.token_prob_map  = None    # will be rebuilt on first block-denoise
        _advance_block(
            state,
            x_batch[bi:bi + 1],
            Lp,
            gen_length,
            mask_id,
            eos_token_id,
        )
        if reset_refresh_counter:
            state.decoded_since_refresh = 0




# ── Batched block denoise ─────────────────────────────────────────────────────


def _apply_block_logits_to_target_window(
    state: FusedState, x_br: torch.Tensor, block_logits: torch.Tensor,
    target_start: int, target_end: int,
    mask_id: int, threshold: float, Lp: int, gen_length: int, L: int,
    eos_token_id: int,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> None:
    """Commit predecessor-window logits to target positions [target_start, target_end)."""
    target_lo = target_start
    target_hi = min(target_end, Lp + gen_length)
    eos_abs = _first_eos_abs_for_row(x_br, Lp, gen_length, eos_token_id)
    if eos_abs is not None:
        target_hi = min(target_hi, eos_abs)
    valid_n = target_hi - target_lo
    if valid_n <= 0:
        _advance_block(state, x_br, Lp, gen_length, mask_id, eos_token_id)
        return

    target_x    = x_br[0, target_lo:target_hi]
    target_mask = (target_x == mask_id)
    if not target_mask.any():
        _advance_block(state, x_br, Lp, gen_length, mask_id, eos_token_id)
        return

    logits_v = block_logits[0, :valid_n].float().clone()
    logits_v[:, mask_id] = -1e4

    conf_all, pred_all = _sample_tokens_from_logits(logits_v, temperature, top_p, top_k)
    masked_conf        = conf_all.masked_fill(~target_mask, float('-inf'))
    masked_conf_no_eos = masked_conf.clone()
    masked_conf_no_eos[pred_all == eos_token_id] = float('-inf')

    transfer = target_mask & (conf_all >= threshold)
    if not transfer.any():
        if float(masked_conf_no_eos.max()) > float('-inf'):
            transfer[int(masked_conf_no_eos.argmax().item())] = True
        else:
            transfer[int(masked_conf.argmax().item())] = True
    target_x_pred = target_x.clone()
    target_x_pred[target_mask] = pred_all[target_mask]
    x_br[0, target_lo:target_hi][transfer] = target_x_pred[transfer]
    newly = int(transfer.sum().item())
    committed_pos = torch.empty(0, device=x_br.device, dtype=torch.long)
    committed_tok = torch.empty(0, device=x_br.device, dtype=pred_all.dtype)
    committed_conf = torch.empty(0, device=x_br.device, dtype=conf_all.dtype)
    if transfer.any():
        committed_pos = transfer.nonzero(as_tuple=True)[0] + target_lo
        committed_tok = pred_all[transfer]
        committed_conf = conf_all[transfer]
        _record_committed_tokens(
            state,
            committed_pos,
            committed_tok,
            committed_conf,
            eos_token_id,
        )
    _trace_add("generated_tokens", newly)

    _update_prob_map(state, block_logits[:, :valid_n, :], target_lo, L)
    _advance_block(state, x_br, Lp, gen_length, mask_id, eos_token_id)




@torch.no_grad()
def batched_block_denoise(
    model, active_states: List[FusedState],
    x_batch: torch.Tensor, unified_kv: list,
    mask_id: int, threshold: float, eos_token_id: int,
    Lp: int, gen_length: int, L: int,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> bool:
    """Single batched predecessor-window dual_cache forward for active branches."""
    filtered_states = []
    for s in active_states:
        if s.done:
            continue
        row = x_batch[s.branch_idx:s.branch_idx + 1]
        _advance_to_active_window_or_done(s, row, Lp, gen_length, mask_id, eos_token_id)
        if not s.done and (x_batch[s.branch_idx, s.block_start:s.block_end] == mask_id).any():
            filtered_states.append(s)
    active_states = filtered_states
    if not active_states:
        return False
    if Lp <= 0:
        raise ValueError("Dream predecessor-window block denoise requires a non-empty prompt")


    device = x_batch.device
    N = len(active_states)


    input_lens = [s.block_end - s.block_start for s in active_states]
    max_q      = max(input_lens)


    x_packed     = torch.full((N, max_q), mask_id, dtype=torch.long, device=device)
    position_ids = torch.zeros(N, max_q, dtype=torch.long, device=device)
    replace_pos  = torch.zeros(N, L,     dtype=torch.bool,  device=device)


    for i, s in enumerate(active_states):
        ln = input_lens[i]
        pred_start = s.block_start - 1
        pred_end = s.block_end - 1
        x_packed[i, :ln] = x_batch[s.branch_idx, pred_start:pred_end]
        position_ids[i, :ln] = torch.arange(pred_start, pred_end, device=device)
        position_ids[i, ln:] = pred_end - 1          # padding: last valid RoPE index
        replace_pos[i, pred_start:pred_end] = True


    # Advanced indexing creates a COPY → must write back after forward
    active_idx = [s.branch_idx for s in active_states]
    batched_kv = [(unified_kv[l][0][active_idx], unified_kv[l][1][active_idx])
                  for l in range(len(unified_kv))]

    # Set global so patched RoPE can gather per-row positions from full L cos table.
    # Do NOT pass position_ids to model — that would make rotary_emb return a
    # (N, max_q, head_dim) table instead of the full (1, L, head_dim) we need.
    global _BATCHED_POSITION_IDS
    _BATCHED_POSITION_IDS = position_ids
    try:
        out = model(
            x_packed,
            past_key_values=batched_kv,
            use_cache=True,
            dual_cache=True,
            replace_position=replace_pos,
        )
    finally:
        _BATCHED_POSITION_IDS = None   # always clear, even on exception

    # Write-back: advanced indexing made a copy; dual_cache updated that copy in-place
    for layer_idx, (k_new, v_new) in enumerate(out.past_key_values):
        for i, s in enumerate(active_states):
            unified_kv[layer_idx][0][s.branch_idx].copy_(k_new[i])
            unified_kv[layer_idx][1][s.branch_idx].copy_(v_new[i])


    # Raw predecessor-window logits map directly to the target block.
    for i, s in enumerate(active_states):
        block_size   = s.block_end - s.block_start
        block_logits = out.logits[i:i + 1, :block_size, :]   # (1, block_size, vocab)
        x_br         = x_batch[s.branch_idx:s.branch_idx + 1]
        _apply_block_logits_to_target_window(
            s, x_br, block_logits,
            target_start=s.block_start,
            target_end=s.block_end,
            mask_id=mask_id,
            threshold=threshold,
            Lp=Lp,
            gen_length=gen_length,
            L=L,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    return True


@torch.no_grad()
def batched_full_refresh(
    model,
    active_states: List[FusedState],
    x_batch: torch.Tensor,
    unified_kv: list,
    mask_id: int,
    threshold: float,
    eos_token_id: int,
    Lp: int,
    gen_length: int,
    L: int,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> None:
    """Full-sequence refresh for all active branch rows, with per-row KV writeback."""
    active_states = [s for s in active_states if not s.done]
    if not active_states:
        return

    active_idx = [s.branch_idx for s in active_states]
    out = model(
        x_batch[active_idx],
        use_cache=True,
        dual_cache=True,
    )

    for layer_idx, (k_ref, v_ref) in enumerate(out.past_key_values):
        for local_idx, state in enumerate(active_states):
            unified_kv[layer_idx][0][state.branch_idx].copy_(k_ref[local_idx])
            unified_kv[layer_idx][1][state.branch_idx].copy_(v_ref[local_idx])

    shifted_logits = _shift_logits(out.logits)
    _decode_from_full_logits(
        active_states,
        x_batch,
        shifted_logits,
        mask_id,
        threshold,
        L,
        Lp,
        gen_length,
        eos_token_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        reset_refresh_counter=True,
    )




# ── Cross-branch policy ───────────────────────────────────────────────────────


def _leader_index_by_decoded(states: List[FusedState]) -> int:
    return max(range(len(states)), key=lambda k: states[k].tokens_decoded)


def _effective_overlap_end_rel(gen: torch.Tensor, eos_token_id: Optional[int]) -> int:
    return _counted_end_rel(gen, eos_token_id)


def _are_compatible(dst: FusedState, src_list: List[FusedState],
                    x_batch: torch.Tensor, mask_id: int, Lp: int,
                    gen_length: int,
                    eos_token_id: Optional[int]) -> List[FusedState]:
    """Return sources compatible with dst on decoded-token overlap, ignoring post-EOS."""
    gen_dst = x_batch[dst.branch_idx, Lp:Lp + gen_length]
    dst_end = _effective_overlap_end_rel(gen_dst, eos_token_id)
    compatible = []
    for src in src_list:
        gen_src = x_batch[src.branch_idx, Lp:Lp + gen_length]
        src_end = _effective_overlap_end_rel(gen_src, eos_token_id)
        compare_end = min(dst_end, src_end)
        dst_slice = gen_dst[:compare_end]
        src_slice = gen_src[:compare_end]
        both = (dst_slice != mask_id) & (src_slice != mask_id)
        if not both.any() or (dst_slice[both] == src_slice[both]).all().item():
            compatible.append(src)
    return compatible




def _merge_states(dst: FusedState, src_list: List[FusedState],
                  x_batch: torch.Tensor, mask_id: int,
                  conf_threshold: float = CONF_THRESHOLD,
                  Lp: int = 0,
                  gen_length: int = 0,
                  eos_token_id: Optional[int] = None) -> int:
    """Merge compatible-source tokens into dst using dst's own probability map."""
    tpm = dst.token_prob_map
    if tpm is None:
        return 0
    filled = 0
    merged_positions = []
    merged_tokens = []
    merged_probs = []
    if gen_length <= 0:
        return 0

    dst_eos_abs = _first_eos_abs_for_row(
        x_batch[dst.branch_idx:dst.branch_idx + 1],
        Lp,
        gen_length,
        eos_token_id,
    )
    merge_start = max(dst.block_start, Lp)
    merge_end = min(dst.block_end, Lp + gen_length)
    if dst_eos_abs is not None:
        merge_end = min(merge_end, dst_eos_abs)

    for pos in range(merge_start, merge_end):
        if x_batch[dst.branch_idx, pos].item() != mask_id:
            continue
        if pos >= len(tpm) or not tpm[pos]:
            continue
        best_tok, best_prob = mask_id, 0.0
        for src in src_list:
            if src.done:
                continue
            src_eos_abs = _first_eos_abs_for_row(
                x_batch[src.branch_idx:src.branch_idx + 1],
                Lp,
                gen_length,
                eos_token_id,
            )
            if src_eos_abs is not None and pos > src_eos_abs:
                continue
            tok = x_batch[src.branch_idx, pos].item()
            if tok == mask_id:
                continue
            prob = tpm[pos].get(tok, 0.0)
            if prob > best_prob:
                best_prob, best_tok = prob, tok
        if best_prob >= conf_threshold and best_tok != mask_id:
            x_batch[dst.branch_idx, pos] = best_tok
            merged_positions.append(pos)
            merged_tokens.append(best_tok)
            merged_probs.append(best_prob)
            filled += 1
    if filled > 0:
        device = x_batch.device
        _record_committed_tokens(
            dst,
            torch.tensor(merged_positions, device=device, dtype=torch.long),
            torch.tensor(merged_tokens, device=device, dtype=torch.long),
            torch.tensor(merged_probs, device=device, dtype=torch.float32),
            eos_token_id if eos_token_id is not None else -1,
            provenance=_PROV_MERGED,
        )
        if gen_length > 0:
            _advance_block(
                dst,
                x_batch[dst.branch_idx:dst.branch_idx + 1],
                Lp,
                gen_length,
                mask_id,
                eos_token_id,
            )
        _refresh_eos_metadata_from_row(dst, x_batch, Lp, gen_length, eos_token_id)
    _trace_add("merged_tokens", filled)
    return filled




def _sync_branch(dst: FusedState, leader: FusedState,
                 x_batch: torch.Tensor,
                 unified_kv: Optional[list],
                 Lp: int, gen_length: int, mask_id: int,
                 eos_token_id: Optional[int] = None) -> None:
    """Hard-sync a lagging branch by fully replacing row, KV, and metadata."""
    src_row = x_batch[leader.branch_idx]
    dst_before = x_batch[dst.branch_idx].clone()
    changed = (dst_before != src_row)
    copied_from_sync = changed & (src_row != mask_id)
    dropped_to_mask = changed & (dst_before != mask_id) & (src_row == mask_id)
    overwritten_non_mask = changed & (dst_before != mask_id) & (src_row != mask_id)


    dropped_origin_counts = {
        "generated": 0,
        "merged": 0,
        "copied": 0,
        "prompt": 0,
        "other": 0,
    }
    if dropped_to_mask.any() and _PROVENANCE_BATCH is not None:
        dropped_origin_counts = _summarize_drop_origins(
            _PROVENANCE_BATCH[dst.branch_idx][dropped_to_mask]
        )
    elif dropped_to_mask.any():
        dropped_origin_counts["other"] = int(dropped_to_mask.sum().item())

    x_batch[dst.branch_idx].copy_(src_row)


    if _PROVENANCE_BATCH is not None:
        _PROVENANCE_BATCH[dst.branch_idx].copy_(_PROVENANCE_BATCH[leader.branch_idx])


    _trace_add("copied_tokens", int(copied_from_sync.sum().item()))
    _trace_add_drop_counts(dropped_origin_counts)

    dst.decoded_since_refresh = leader.decoded_since_refresh
    dst.token_prob_map = _copy_token_prob_map(leader.token_prob_map)
    dst.cached_prob = dict(leader.cached_prob) if leader.cached_prob is not None else None
    dst.hit_eos = leader.hit_eos
    dst.eos_position = leader.eos_position
    dst.eos_prob = leader.eos_prob
    if unified_kv is not None:
        _sync_kv_row(unified_kv, dst.branch_idx, leader.branch_idx)


    # Realign to dst's OWN block_size boundary — copying leader.block_end would
    # collapse bs=64 into a bs=4 window and destroy block-size diversity.
    _update_block_window_for_branch(
        dst,
        x_batch[dst.branch_idx:dst.branch_idx + 1],
        Lp,
        gen_length,
        mask_id,
        eos_token_id,
    )
    _refresh_eos_metadata_from_row(dst, x_batch, Lp, gen_length, eos_token_id)




class FusedCrossStateBestPolicy:
    def __init__(self, Lp: int, gen_length: int, mask_id: int,
                 sync_threshold: int = SYNC_THRESHOLD,
                 conf_threshold: float = CONF_THRESHOLD,
                 eos_token_id: Optional[int] = None):
        self._Lp = Lp
        self._gen_length = gen_length
        self._mask_id = mask_id
        self._sync_threshold = sync_threshold
        self._conf_threshold = conf_threshold
        self._eos_token_id = eos_token_id


    def compare_and_copy(self, states: List[FusedState],
                         x_batch: torch.Tensor,
                         unified_kv: Optional[list] = None) -> None:
        """Merge from furthest block-end first, then hard-sync lagging branches."""
        _refresh_all_branch_metadata(
            states,
            x_batch,
            self._Lp,
            self._gen_length,
            self._mask_id,
            self._eos_token_id,
        )
        leader_idx = _leader_index_by_decoded(states)
        leader = states[leader_idx]
        if leader.tokens_decoded == 0:
            return

        eligible = [i for i in range(len(states)) if not states[i].done]
        order = sorted(eligible, key=lambda k: states[k].block_end, reverse=True)

        if not all(states[i].block_end <= self._Lp for i in order):
            for i_rank in range(len(order)):
                dst = states[order[i_rank]]
                src_list = [states[order[j]] for j in range(i_rank + 1, len(order))]
                compatible = _are_compatible(
                    dst,
                    src_list,
                    x_batch,
                    self._mask_id,
                    self._Lp,
                    self._gen_length,
                    self._eos_token_id,
                )
                if compatible:
                    filled = _merge_states(
                        dst,
                        compatible,
                        x_batch,
                        self._mask_id,
                        self._conf_threshold,
                        self._Lp,
                        self._gen_length,
                        self._eos_token_id,
                    )
        _refresh_all_branch_metadata(
            states,
            x_batch,
            self._Lp,
            self._gen_length,
            self._mask_id,
            self._eos_token_id,
        )
        leader_idx = _leader_index_by_decoded(states)
        leader = states[leader_idx]

        for i, dst in enumerate(states):
            if i == leader_idx or dst.done:
                continue
            if leader.tokens_decoded - dst.tokens_decoded > self._sync_threshold:
                _sync_branch(dst, leader, x_batch, unified_kv,
                             self._Lp, self._gen_length, self._mask_id,
                             self._eos_token_id)


    def hard_sync_all(self, states: List[FusedState],
                      x_batch: torch.Tensor,
                      unified_kv: Optional[list] = None) -> int:
        leader_idx = _leader_index_by_decoded(states)
        leader = states[leader_idx]
        synced = 0
        for i, s in enumerate(states):
            if i == leader_idx or s.done:
                continue
            _sync_branch(s, leader, x_batch, unified_kv,
                         self._Lp, self._gen_length, self._mask_id,
                         self._eos_token_id)
            synced += 1
        return synced




# ── Result builder ────────────────────────────────────────────────────────────


def _build_result(winner: FusedState, x_batch: torch.Tensor,
                  start_time: float, total_nfe: int,
                  nfe_init: int = 0, nfe_block: int = 0,
                  nfe_refresh: int = 0,
                  refresh_count: int = 0,
                  exit_reason: str = "unknown",
                  all_states: Optional[List[FusedState]] = None,
                  mask_id: Optional[int] = None,
                  eos_token_id: Optional[int] = None,
                  Lp: Optional[int] = None,
                  gen_length: Optional[int] = None) -> Tuple[torch.Tensor, int, dict]:
    wall = time.time() - start_time
    tokens = winner.tokens_decoded
    stats = {
        'total_wall_time': wall,
        'tokens_per_second': tokens / wall if wall > 0 else 0.0,
        'total_tokens_generated': tokens,
        'total_nfe': total_nfe,
        'nfe_init': nfe_init,
        'nfe_block': nfe_block,
        'nfe_refresh': nfe_refresh,
        'refresh_count': refresh_count,
        'final_block_size': winner.block_size,
        'winner_branch': winner.branch_idx,
        'exit_reason': exit_reason,
        'generation_policy': (
            'early_eos_block_batching'
            if exit_reason == "eos_ready"
            else 'full_budget_no_early_eos_block_batching'
        ),
        'eos_early_exit': exit_reason == "eos_ready",
    }
    states_for_stats = all_states if all_states is not None else [winner]
    block_sizes = [s.block_size for s in states_for_stats]
    stats['nfe_per_block_size'] = {bs: total_nfe for bs in block_sizes}
    stats['nfe_init_per_block_size'] = {bs: nfe_init for bs in block_sizes}
    stats['nfe_block_per_block_size'] = {bs: nfe_block for bs in block_sizes}
    stats['nfe_refresh_per_block_size'] = {bs: nfe_refresh for bs in block_sizes}
    if _TRACE_STATS is not None:
        stats['token_event_trace'] = dict(_TRACE_STATS)
    _trace_reset()
    output = x_batch[winner.branch_idx:winner.branch_idx + 1]
    if exit_reason == "eos_ready" and Lp is not None and winner.eos_position >= Lp:
        output = output[:, :winner.eos_position + 1]
    return output, total_nfe, stats




# ── Entry point ───────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_block_batching(
    model,
    prompt: torch.Tensor,
    gen_length: int = 256,
    block_sizes: List[int] = None,
    mask_id: int = None,
    threshold: float = 0.9,
    refresh_block_size: int = REFRESH_BLOCK_SIZE,
    sync_threshold: int = SYNC_THRESHOLD,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    debug: bool = False,
    stop_on_eos: bool = False,
) -> Tuple[torch.Tensor, int, dict]:
    """Generate with fused block batching for Dream.

    By default EOS is treated like the normal Dream eval path: decoded text can
    be split at EOS after generation, but EOS does not end the generation loop.
    Set stop_on_eos=True only for legacy early-exit experiments.
    """
    del debug

    if block_sizes is None:
        block_sizes = [gen_length]
    if mask_id is None:
        mask_id = model.config.mask_token_id
    if threshold is None:
        threshold = 0.9
    temperature = 0.0
    top_p = None
    top_k = 1

    block_sizes = sorted(block_sizes)
    N = len(block_sizes)
    Lp = prompt.shape[1]
    device = prompt.device
    eos_token_id = model.config.eos_token_id
    generation_eos_token_id = eos_token_id if stop_on_eos and eos_token_id is not None else -1
    L = Lp + gen_length
    if Lp <= 0:
        raise ValueError("Dream block batching requires a non-empty prompt")

    for bs in block_sizes:
        assert gen_length % bs == 0, f"gen_length {gen_length} not divisible by {bs}"


    x_batch = torch.full((N, L), mask_id, dtype=torch.long, device=device)
    x_batch[:, :Lp] = prompt
    _trace_init(N, L, Lp, device)


    states = [
        FusedState(branch_idx=k, block_size=bs,
                   block_start=Lp, block_end=min(Lp + bs, L),
                   cache_row=k)
        for k, bs in enumerate(block_sizes)
    ]
    policy = FusedCrossStateBestPolicy(
        Lp=Lp,
        gen_length=gen_length,
        mask_id=mask_id,
        sync_threshold=sync_threshold,
        eos_token_id=generation_eos_token_id,
    )

    nfe_init = 0
    nfe_block = 0
    nfe_refresh = 0
    refresh_count = 0
    total_nfe = 0
    start_time = time.time()


    # Apply monkey-patches for batched dual_cache (idempotent — runs once per process)
    _apply_batched_dual_cache_patches(model)


    with torch.no_grad():
        out0 = model(x_batch[0:1], use_cache=True, dual_cache=True)   # identical initial rows
    nfe_init += 1
    total_nfe = nfe_init + nfe_block + nfe_refresh
    unified_kv: list = _broadcast_kv(out0.past_key_values, N)
    _trace_set_phase("init")
    _decode_from_full_logits(
        states,
        x_batch,
        _shift_logits(out0.logits),
        mask_id,
        threshold,
        L,
        Lp,
        gen_length,
        generation_eos_token_id,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        reset_refresh_counter=True,
    )
    result = _check_eos_ready(states, x_batch, mask_id, generation_eos_token_id, Lp,
                              start_time, total_nfe,
                              nfe_init, nfe_block, nfe_refresh, refresh_count)
    if result:
        return result

    loop_iter = 0


    # ── Main loop ──────────────────────────────────────────────────────────
    while not all(s.done for s in states):
        loop_iter += 1
        result = _check_eos_ready(states, x_batch, mask_id, generation_eos_token_id, Lp,
                                  start_time, total_nfe,
                                  nfe_init, nfe_block, nfe_refresh, refresh_count)
        if result:
            return result

        # Batched block denoise — ONE NFE for all active branches
        active = [s for s in states if not s.done]
        if active:
            _trace_set_phase("block")
            ran_block_forward = batched_block_denoise(
                model, active, x_batch, unified_kv,
                mask_id, threshold, generation_eos_token_id,
                Lp, gen_length, L,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            if ran_block_forward:
                nfe_block += 1
                total_nfe = nfe_init + nfe_block + nfe_refresh


        result = _check_eos_ready(states, x_batch, mask_id, generation_eos_token_id, Lp,
                                  start_time, total_nfe,
                                  nfe_init, nfe_block, nfe_refresh, refresh_count)
        if result:
            return result


        _trace_set_phase("policy")
        policy.compare_and_copy(states, x_batch, unified_kv)
        result = _check_eos_ready(states, x_batch, mask_id, generation_eos_token_id, Lp,
                                  start_time, total_nfe,
                                  nfe_init, nfe_block, nfe_refresh, refresh_count)
        if result:
            return result

        active = [s for s in states if not s.done]
        if active and any(s.decoded_since_refresh >= refresh_block_size for s in active):
            _trace_set_phase("refresh")
            batched_full_refresh(
                model,
                active,
                x_batch,
                unified_kv,
                mask_id,
                threshold,
                generation_eos_token_id,
                Lp,
                gen_length,
                L,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            nfe_refresh += 1
            refresh_count += 1
            total_nfe = nfe_init + nfe_block + nfe_refresh

            result = _check_eos_ready(states, x_batch, mask_id, generation_eos_token_id, Lp,
                                      start_time, total_nfe,
                                      nfe_init, nfe_block, nfe_refresh, refresh_count)
            if result:
                return result


    winner = max(states, key=lambda s: (s.progress, s.block_size))
    total_nfe = nfe_init + nfe_block + nfe_refresh
    exit_reason = "all_branches_done"
    return _build_result(
        winner,
        x_batch,
        start_time,
        total_nfe,
        nfe_init=nfe_init,
        nfe_block=nfe_block,
        nfe_refresh=nfe_refresh,
        refresh_count=refresh_count,
        exit_reason=exit_reason,
        all_states=states,
        mask_id=mask_id,
        eos_token_id=generation_eos_token_id,
        Lp=Lp,
        gen_length=gen_length,
    )
