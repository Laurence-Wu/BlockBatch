# Copyright 2026 BlockBatch Authors
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation harness for Dream Block Batching generation.
Registered as: dream_blockbatching

Usage:
  accelerate launch eval_blockBatching.py --model dream_blockbatching \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,block_size_list=4-8-16-32-64,..." \
    --tasks gsm8k --num_fewshot 5 --batch_size 1 \
    --output_path <save_dir>/lm_eval --log_samples
"""
import logging
import os
import json
import time
import types
from datetime import timedelta
from typing import List, Optional, Type, TypeVar, Union

import torch
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate

from model.configuration_dream import DreamConfig
from model.modeling_dream import DreamModel
from model.generation_utils_block import DreamGenerationMixin
# Keep Dream block-batching eval pinned to the original-bulk generator.
from generate_blockBatching_original_bulk import generate_block_batching

eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


@register_model("dream_blockbatching")
class DreamBlockBatching(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 256,
        max_length: Optional[int] = 2048,
        add_bos_token: Optional[bool] = False,
        trust_remote_code: Optional[bool] = True,
        cache_dir: Optional[str] = None,
        save_dir: Optional[str] = None,
        show_speed: Optional[bool] = False,
        escape_until: Optional[bool] = False,
        stop_on_eos: Optional[bool] = False,
        # Block batching specific
        block_size_list: str = "4-8-16-32-64",
        threshold: Optional[float] = 0.9,
        sync_threshold: Optional[int] = 8,
        refresh_block_size: Optional[int] = 32,
        # loglikelihood params (kept for API compatibility)
        nll_type: Optional[str] = "mc",
        mc_num: Optional[int] = 128,
        **kwargs,
    ) -> None:
        super().__init__()

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])

        if not accelerator.num_processes > 1:
            self._device = torch.device(device if device else "cuda")
        else:
            self.accelerator = accelerator
            self._device = torch.device(f"{accelerator.device}")

        if not hasattr(self, "accelerator"):
            self._rank = 0
            self._world_size = 1
        else:
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.batch_size_per_gpu = int(batch_size) if isinstance(batch_size, str) else batch_size
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code, cache_dir)

        self.max_new_tokens = int(max_new_tokens)
        self.max_length = max_length
        self.add_bos_token = add_bos_token
        self.threshold = threshold
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.escape_until = _as_bool(escape_until)
        self.stop_on_eos = _as_bool(stop_on_eos)
        self.sync_threshold = int(sync_threshold)
        self.refresh_block_size = int(refresh_block_size)
        self.mc_num = mc_num
        self.nll_type = nll_type

        if isinstance(block_size_list, (int, float)):
            self.block_sizes = [int(block_size_list)]
        else:
            self.block_sizes = [int(x) for x in str(block_size_list).split("-")]

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code, cache_dir):
        self.model = (
            DreamModel.from_pretrained(
                pretrained,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                cache_dir=cache_dir,
            ).eval()
        ).to(self._device)
        self.model.diffusion_generate = types.MethodType(
            DreamGenerationMixin.diffusion_generate, self.model)
        self.model._sample = types.MethodType(DreamGenerationMixin._sample, self.model)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code, cache_dir=cache_dir)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    @classmethod
    def create_from_arg_string(cls: Type[T], arg_string: str,
                               additional_config: Optional[dict] = None) -> T:
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(text, return_tensors="pt",
                              add_special_tokens=add_special_tokens).input_ids

    def _generate_batch(self, prompts: List[str]):
        if self.add_bos_token:
            prompts = [self.tokenizer.bos_token + p for p in prompts]
        prompt_ids = self.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left").input_ids.to(self._device)

        mask_id = self.model.config.mask_token_id
        out_ids, total_nfe, stats = generate_block_batching(
            self.model,
            prompt_ids,
            gen_length=self.max_new_tokens,
            block_sizes=self.block_sizes,
            mask_id=mask_id,
            threshold=self.threshold,
            sync_threshold=self.sync_threshold,
            refresh_block_size=self.refresh_block_size,
            stop_on_eos=self.stop_on_eos,
        )
        nfe_init = int(stats.get('nfe_init', 0))
        nfe_block = int(stats.get('nfe_block', 0))
        nfe_refresh = int(stats.get('nfe_refresh', 0))
        nfe_from_parts = nfe_init + nfe_block + nfe_refresh
        if nfe_from_parts > 0:
            if int(total_nfe) != nfe_from_parts:
                eval_logger.warning(
                    "Dream block-batching NFE mismatch: returned=%s parts=%s "
                    "(init=%s block=%s refresh=%s). Using parts sum.",
                    total_nfe, nfe_from_parts, nfe_init, nfe_block, nfe_refresh,
                )
            total_nfe = nfe_from_parts
            stats['total_nfe'] = total_nfe
        else:
            stats.setdefault('total_nfe', total_nfe)
        stats['nfe_init'] = nfe_init
        stats['nfe_block'] = nfe_block
        stats['nfe_refresh'] = nfe_refresh

        # Count actual decoded (non-mask) tokens in generation region.
        # winner.progress only tracks tokens committed by the winner branch itself;
        # tokens donated via sync from other branches are not counted there.
        gen_ids = out_ids[0, prompt_ids.shape[1]:]
        actual_tokens = int((gen_ids != mask_id).sum().item())
        stats['total_tokens_generated'] = actual_tokens

        responses = [
            self.tokenizer.decode(
                gen_ids.tolist(), skip_special_tokens=False
            ).split(self.tokenizer.eos_token)[0]
        ]
        print('=' * 20)
        print('question:', prompts[0][:80])
        print('answer:', responses[0][:256])
        print(f'nfe={total_nfe} '
              f'(init={nfe_init} block={nfe_block} refresh={nfe_refresh})  '
              f'wall={stats["total_wall_time"]:.2f}s  '
              f'winner_bs={stats["final_block_size"]}  '
              f'tokens={actual_tokens}')
        print('=' * 20, end='\n\n')
        return responses, total_nfe, stats

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []
        processed_count = 0
        save_path = None

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f'rank_{self.rank}.jsonl')
            if os.path.exists(save_path):
                with open(save_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        res.append(data['answer'] if isinstance(data, dict) else data)
                processed_count = len(res)
                print(f"Resuming from {processed_count} samples")

        total_nfe = 0
        pbar = tqdm(total=len(requests), desc="dream_blockbatching generate_until",
                    disable=disable_tqdm)
        start_time = time.time()

        for batch_idx in range(0, len(requests), self.batch_size):
            batch = requests[batch_idx: batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch])

            if batch_idx < processed_count:
                pbar.update(len(contexts))
                continue

            responses, nfe, stats = self._generate_batch(list(contexts))

            if not self.escape_until:
                for i, r in enumerate(responses):
                    for s in gen_args[0].get('until', []):
                        r = r.split(s)[0]
                    responses[i] = r

            res.extend(responses)
            total_nfe += nfe
            pbar.update(len(contexts))

            if save_path is not None:
                with open(save_path, 'a', encoding='utf-8') as f:
                    for r in responses:
                        rec = {'answer': r, 'nfe': nfe,
                               'total_nfe': nfe,
                               'nfe_init': int(stats.get('nfe_init', 0)),
                               'nfe_block': int(stats.get('nfe_block', 0)),
                               'nfe_refresh': int(stats.get('nfe_refresh', 0)),
                               'nfe_per_block_size': stats.get('nfe_per_block_size', {}),
                               'nfe_init_per_block_size': stats.get('nfe_init_per_block_size', {}),
                               'nfe_block_per_block_size': stats.get('nfe_block_per_block_size', {}),
                               'nfe_refresh_per_block_size': stats.get('nfe_refresh_per_block_size', {}),
                               'refresh_count': int(stats.get('refresh_count', 0)),
                               'wall_s': round(stats['total_wall_time'], 3),
                               'tokens_generated': stats.get('total_tokens_generated', 0),
                               'winner_bs': stats.get('final_block_size', None),
                               'exit_reason': stats.get('exit_reason', None),
                               'generation_policy': stats.get(
                                   'generation_policy',
                                   'full_budget_no_early_eos_block_batching',
                               ),
                               'eos_early_exit': bool(stats.get('eos_early_exit', False)),
                               'escape_until': self.escape_until,
                               'stop_on_eos': self.stop_on_eos,
                               'block_sizes': self.block_sizes}
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

        pbar.close()
        if self.show_speed:
            elapsed = time.time() - start_time
            print(f"Total time: {elapsed:.1f}s | avg NFE: {total_nfe / max(len(res), 1):.1f}")
        return res

    def loglikelihood(self, requests):
        raise NotImplementedError("Use dream (base) model for loglikelihood")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def loglikelihood_without_tokenization(self, requests):
        raise NotImplementedError


if __name__ == "__main__":
    cli_evaluate()
