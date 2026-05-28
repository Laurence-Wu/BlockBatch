#!/usr/bin/env python3
"""
seeded_eval.py -- Dream generation with deterministic-island tokens pre-seeded.

Subclasses the Dream eval class (eval.py) to inject attractor tokens into the
initial masked sequence via generation_tokens_hook_func before denoising starts.

Usage (from dream/ directory):
    accelerate launch seeded_eval.py --model dream_seeded \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,
                      max_new_tokens=256,diffusion_steps=256,
                      add_bos_token=true,alg=confidence_threshold,threshold=0.9,
                      block_length=256,use_cache=false,
                      cache_dir=~/.cache/huggingface,
                      attractors_file=../blocksize_ablation/attractors_dream_gsm8k_full.json,
                      save_dir=../blocksize_ablation/results/dream/gsm8k/seeded" \
        --tasks gsm8k --num_fewshot 5 --batch_size 1 \
        --output_path ../blocksize_ablation/results/dream/gsm8k/seeded/lm_eval --log_samples
"""
import json
import os
import sys
import time
from typing import List, Optional

import torch
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate

# import the base Dream class from eval.py in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import Dream


@register_model("dream_seeded")
class DreamSeeded(Dream):
    def __init__(
        self,
        *,
        attractors_file: Optional[str] = None,
        use_cache: Optional[bool] = True,
        block_length: Optional[int] = 32,
        **kwargs,
    ) -> None:
        super().__init__(use_cache=use_cache, block_length=block_length, **kwargs)

        self._attractors: dict = {}
        if attractors_file:
            attractors_path = os.path.abspath(attractors_file)
            print(f'[DreamSeeded] loading attractors from {attractors_path}')
            with open(attractors_path, 'r', encoding='utf-8') as f:
                self._attractors = json.load(f)
            print(f'[DreamSeeded] loaded attractors for {len(self._attractors)} samples')

        self._sample_counter: int = 0
        # sync counter with already-saved results so resume doesn't misalign attractors
        if self.save_dir:
            save_path = os.path.join(self.save_dir, f'rank_{self.rank}.jsonl')
            if os.path.exists(save_path):
                with open(save_path) as f:
                    self._sample_counter = sum(1 for l in f if l.strip())
                print(f'[DreamSeeded] resuming from sample {self._sample_counter}')

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        sample_id = self._sample_counter
        self._sample_counter += 1

        attractor_list = self._attractors.get(str(sample_id), {}).get("attractors", [])

        # build {relative_pos: token_id} map using the tokenizer
        attractor_ids: dict = {}
        for entry in attractor_list:
            rel_pos = entry["pos"]
            tok_text = entry["token"]
            ids = self.tokenizer.encode(tok_text, add_special_tokens=False)
            if ids:
                attractor_ids[rel_pos] = ids[0]

        # apply chat template / bos if needed (mirrors base class)
        if self.if_apply_chat_template:
            messages = [{"role": "user", "content": prompts[0]}]
            prompts = [self.apply_chat_template(messages)]
        else:
            if self.add_bos_token:
                prompts = [self.tokenizer.bos_token + p for p in prompts]

        prompt_ids = self.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).input_ids
        prompt_len = prompt_ids.shape[1]

        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)

        def _attractor_hook(step, x, logits):
            if step is None and attractor_ids:
                for rel_pos, tid in attractor_ids.items():
                    abs_pos = prompt_len + rel_pos
                    if abs_pos < x.shape[1]:
                        x[:, abs_pos] = tid
            return x

        _nfe_counter = [0]
        _orig_fwd = self.model.forward

        def _nfe_fwd(*a, **kw):
            _nfe_counter[0] += 1
            return _orig_fwd(*a, **kw)

        self.model.forward = _nfe_fwd
        _t0 = time.time()

        generation_ids = self.model.diffusion_generate(
            prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.diffusion_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
            threshold=self.threshold,
            block_length=self.block_length,
            dual_cache=self.dual_cache,
            generation_tokens_hook_func=_attractor_hook,
        )

        self._last_wall = time.time() - _t0
        self._last_nfe = _nfe_counter[0]
        self.model.forward = _orig_fwd

        self.generated_token_num += (
            generation_ids.sequences[0][prompt_ids.shape[1]:]
            != self.tokenizer.eos_token_id
        ).sum().item()

        responses = [
            self.tokenizer.decode(g[len(p):].tolist()).split(self.tokenizer.eos_token)[0]
            for p, g in zip(prompt_ids, generation_ids.sequences)
        ]

        print('=' * 20)
        print(f'[sample {sample_id}] attractors injected: {len(attractor_ids)}, nfe: {_nfe_counter[0]}')
        print('question: ', prompts[0])
        print('answer:   ', responses[0])
        print('=' * 20, end='\n\n')
        return responses


if __name__ == '__main__':
    cli_evaluate()
