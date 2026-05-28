# Copyright 2026 BlockBatch Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation harness for BlockBatch generation."""
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from llada.generate_blockBatching_expand_sim_refresh_dominent_RMS import (
    generate_block_batching as generate_block_batching_rms,
)
from llada.generate_blockBatching_original_bulk import (
    generate_block_batching as generate_block_batching_original_bulk,
)
from model.modeling_llada import LLaDAModelLM
import json
import time


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_block_batching_generator(generator_variant: str):
    variant = str(generator_variant or "rms").strip().lower()
    if variant in {"rms", "refresh_rms"}:
        return generate_block_batching_rms
    if variant in {
        "original_bulk",
        "bulk",
        "original",
        "generate_blockbatching_original_bulk",
    }:
        return generate_block_batching_original_bulk
    raise ValueError(
        f"Unsupported generator_variant={generator_variant!r}. "
        "Expected one of: rms, original_bulk."
    )

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_blockbatching")
class LLaDABlockBatchingEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        use_cache=False,
        threshold=0.9,
        factor=None,
        save_dir=None,
        show_speed=False,
        block_size_list="4-8-16-32-64",
        policy_name="merge_replenish_sync",
        generator_variant="rms",
        sync_threshold=8,
        conf_threshold=None,
        refresh_block_size=32,
        cache_dir=None,
        force_instruct=False,
        rank_suffix=None,
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which
                             returns a True/False judgment used for accuracy calculation.
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function.
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality,
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.

        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        config = AutoConfig.from_pretrained(model_path, cache_dir=cache_dir)
        config.flash_attention = True
        self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, cache_dir=cache_dir, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = int(steps)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor
        self.is_instruct = _as_bool(force_instruct) or ('instruct' in model_path.lower())
        self.rank_suffix = str(rank_suffix).strip() if rank_suffix else ""
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.policy_name = policy_name
        self.generator_variant = generator_variant
        self.sync_threshold = int(sync_threshold)
        self.conf_threshold = None if conf_threshold in (None, "") else float(conf_threshold)
        self.refresh_block_size = int(refresh_block_size)
        self.generate_block_batching = _resolve_block_batching_generator(generator_variant)

        # Parse block_size_list
        if isinstance(block_size_list, (int, float)):
            self.block_size_list = [int(block_size_list)]
        else:
            self.block_size_list = [int(x) for x in str(block_size_list).split("-")]

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def _log_stats(self, stats: dict, question_id: str):
        """Log detailed block batching statistics to JSON."""
        if self.save_dir is None:
            return

        nfe_init, nfe_block, nfe_refresh = self._nfe_components(stats)
        total_nfe = nfe_init + nfe_block + nfe_refresh
        suffix = f"_{self.rank_suffix}" if self.rank_suffix else ""
        stats_file = Path(self.save_dir) / f"block_batching_stats_rank_{self.rank}_{self.gen_length}{suffix}.jsonl"
        stats_entry = {
            'question_id': question_id,
            'final_block_size': stats.get('final_block_size', None),
            'total_nfe': total_nfe,
            'nfe_init': nfe_init,
            'nfe_block': nfe_block,
            'nfe_refresh': nfe_refresh,
            'nfe_per_block_size': stats.get('nfe_per_block_size', {}),
            'conf_threshold': stats.get('conf_threshold', self.conf_threshold),
            'refresh_block_size': self.refresh_block_size,
            'total_wall_time': stats.get('total_wall_time', 0),
            'progress_per_block_size': stats.get('progress_per_block_size', {}),
        }
        with open(stats_file, 'a') as f:
            f.write(json.dumps(stats_entry) + '\n')

    @staticmethod
    def _block_size_entry(per_block_size: dict, block_size):
        if block_size is None or not isinstance(per_block_size, dict):
            return None
        candidates = [block_size, str(block_size)]
        try:
            candidates.append(int(block_size))
        except (TypeError, ValueError):
            pass
        for key in candidates:
            if key in per_block_size:
                return per_block_size[key]
        return None

    @staticmethod
    def _final_block_nfe_components(stats: dict):
        final_block_size = stats.get('final_block_size')
        values = LLaDABlockBatchingEvalHarness._block_size_entry(
            stats.get('nfe_per_block_size', {}),
            final_block_size,
        )
        if not isinstance(values, dict):
            return None

        nfe_block = int(values.get('nfe_block', 0))
        if 'nfe_init' in values or 'nfe_refresh' in values:
            nfe_init = int(values.get('nfe_init', 0))
            nfe_refresh = int(values.get('nfe_refresh', 0))
        else:
            nfe_full = int(values.get('nfe_full', 0))
            nfe_init = 1 if nfe_full > 0 else 0
            nfe_refresh = max(nfe_full - nfe_init, 0)
        return nfe_init, nfe_block, nfe_refresh

    @staticmethod
    def _nfe_components(stats: dict):
        single_block_components = LLaDABlockBatchingEvalHarness._final_block_nfe_components(stats)
        if single_block_components is not None:
            return single_block_components

        nfe_init = stats.get('nfe_init')
        nfe_block = stats.get('nfe_block')
        nfe_refresh = stats.get('nfe_refresh')
        if nfe_init is not None and nfe_block is not None and nfe_refresh is not None:
            return int(nfe_init), int(nfe_block), int(nfe_refresh)

        nfe_init = nfe_block = nfe_refresh = 0
        for values in stats.get('nfe_per_block_size', {}).values():
            nfe_block += int(values.get('nfe_block', 0))
            if 'nfe_init' in values or 'nfe_refresh' in values:
                nfe_init += int(values.get('nfe_init', 0))
                nfe_refresh += int(values.get('nfe_refresh', 0))
            else:
                # Backward compatibility for older generators where nfe_full
                # included both the initial full forward and refresh forwards.
                nfe_full = int(values.get('nfe_full', 0))
                nfe_init += 1 if nfe_full > 0 else 0
                nfe_refresh += max(nfe_full - 1, 0)
        return nfe_init, nfe_block, nfe_refresh

    def generate_until(self, requests):
        output = []
        all_block_results = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            suffix = f'_{self.rank_suffix}' if self.rank_suffix else ''
            save_path = os.path.join(self.save_dir, f'rank_{rank}_{self.gen_length}{suffix}.jsonl')
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            output.append(data['answer'])
                        else:
                            output.append(data)
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")

        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])

        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            max_len = 0
            pad_len = []
            for req in batch:
                question = req.args[0]
                if self.is_instruct:
                    m = [{"role": "user", "content": question}]
                    user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                    input_ids = self.tokenizer(user_input)['input_ids']
                else:
                    user_input = question
                    input_ids = self.tokenizer(user_input)['input_ids']
                batched_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))
                pad_len.append(max_len - len(input_ids))

            # pad batched_input_ids to the same length
            batched_input_ids = [torch.cat([torch.full((1, max_len - len(input_ids)), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device), torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)], dim=1) for input_ids in batched_input_ids]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)

            if self.batch_size == 1:
                attention_mask = None
            else:
                attention_mask = torch.zeros((batched_input_ids.shape[0], 1, max_len+self.gen_length, max_len+self.gen_length), device=self.device, dtype=torch.bool)
                for i in range(len(pad_len)):
                    attention_mask[i, :, pad_len[i]:, pad_len[i]:] = True


            stop_tokens = req.args[1]['until']
            input_ids = batched_input_ids

            # Call block batching generation
            generation_kwargs = {
                "gen_length": self.gen_length,
                "block_sizes": self.block_size_list,
                "temperature": 0,
                "remasking": self.remasking,
                "mask_id": self.mask_id,
                "threshold": self.threshold,
            }
            if str(self.generator_variant).strip().lower() in {
                "original_bulk",
                "bulk",
                "original",
                "generate_blockbatching_original_bulk",
            }:
                generation_kwargs["sync_threshold"] = self.sync_threshold
                if self.conf_threshold is not None:
                    generation_kwargs["conf_threshold"] = self.conf_threshold
                generation_kwargs["refresh_block_size"] = self.refresh_block_size

            generated_answer, total_nfe, stats = self.generate_block_batching(
                self.model,
                input_ids,
                **generation_kwargs,
            )

            nfe_init, nfe_block, nfe_refresh = self._nfe_components(stats)
            nfe = nfe_init + nfe_block + nfe_refresh
            block_results = {
                'total_nfe': nfe,
                'nfe_init': nfe_init,
                'nfe_block': nfe_block,
                'nfe_refresh': nfe_refresh,
                'final_block_size': stats.get('final_block_size', None),
                'nfe_per_block_size': stats.get('nfe_per_block_size', {}),
                'conf_threshold': stats.get('conf_threshold', self.conf_threshold),
                'refresh_block_size': self.refresh_block_size,
                'sample_id': processed_count + len(output) + 1,
            }

            # Log detailed statistics
            if self.save_dir:
                self._log_stats(stats, str(processed_count + len(output) + 1))

            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
                generated_answer_ids = generated_answer[:, input_ids.shape[1]:]
                if self.show_speed:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                batched_generated_answer = [self.tokenizer.decode(generated_answer_ids[i], skip_special_tokens=True) for i in range(len(generated_answer_ids))]
            else:
                batched_generated_answer = []
                for i in range(len(generated_answer)):
                    generated_answer_i = self.tokenizer.decode(generated_answer[i][input_ids.shape[1]:], skip_special_tokens=False)
                    for stop_seq in stop_tokens:
                        if stop_seq in generated_answer_i:
                            generated_answer_i = generated_answer_i.split(stop_seq)[0]
                    generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])
                    if self.show_speed:
                        num_tokens += (generated_answer_ids != 126081).sum()
                        num_nfe += nfe
                    generated_answer_i = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
                    batched_generated_answer.append(generated_answer_i)

            # Inject NFE and block results into doc for post-processing
            for req in batch:
                req.doc['nfe'] = nfe
                req.doc['block_results'] = block_results

            output.extend(batched_generated_answer)

            all_block_results.append(block_results)

            if self.save_dir is not None:
                # Incrementally save newly generated answers
                with open(save_path, 'a', encoding='utf-8') as f:
                    for generated_answer in batched_generated_answer:
                        record = {
                            'answer': generated_answer,
                            'nfe': nfe,
                            'nfe_init': nfe_init,
                            'nfe_block': nfe_block,
                            'nfe_refresh': nfe_refresh,
                            'wall_s': round(stats.get('total_wall_time', 0.0), 3),
                            'block_length': 'block_batching',
                            'final_block_size': stats.get('final_block_size', None),
                            'candidate_block_sizes': self.block_size_list,
                            'conf_threshold': stats.get('conf_threshold', self.conf_threshold),
                            'refresh_block_size': self.refresh_block_size,
                            'block_results': block_results,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')

            for i in range(len(batched_generated_answer)):
                print('=' * 20)
                print('answer: ', batched_generated_answer[i])
                print('nfe: ', nfe)
                print('avg nfe: ', num_nfe / len(output))
                print('=' * 20, end='\n\n')

        end_time = time.time()
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")

        if self.save_dir is not None and all_block_results:
            suffix = f'_{self.rank_suffix}' if self.rank_suffix else ''
            analysis_path = os.path.join(self.save_dir, f'block_analysis_rank_{self.rank}{suffix}.json')
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(all_block_results, f, indent=2)
            print(f"Block analysis saved to {analysis_path}")

        return output


if __name__ == "__main__":
    cli_evaluate()
