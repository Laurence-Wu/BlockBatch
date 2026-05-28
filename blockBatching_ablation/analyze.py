#!/usr/bin/env python3
"""
analyze.py -- Summarize blocksize ablation results.

Reads all rank_*.jsonl files under results/{model}/{task}/bs{N}/
and prints a table of accuracy, avg NFE, avg latency, and avg TPS per block size.

Usage:
    python analyze.py
    python analyze.py --results-dir /abs/path/to/results
    python analyze.py --model llada --task gsm8k
    python analyze.py --evaluate_accu
"""
import argparse
import glob
import json
import os
import math
import re
import shlex
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dir_label(basename):
    """'seeded_bs32' -> 32, 'bs32' -> 32, 'seeded' -> 'seeded'."""
    m = re.search(r'(\d+)$', basename)
    return int(m.group(1)) if m else basename


def _variant_label(basename):
    """Extract the grouping label from a variant directory name.

    For block-combo dirs the label is the full block-sizes string so it matches
    the key produced by _record_key (which uses block_sizes from the record):
        'block_combo_bs4-8-16' -> '4-8-16'
    For all other dirs fall back to _dir_label (trailing number):
        'sync_threshold_4' -> 4, 'refresh_block_size_32' -> 32
    """
    m = re.match(r'block_combo_bs(.+)', basename)
    if m:
        return m.group(1)
    return _dir_label(basename)


def _block_size_entry(per_block_size, block_size):
    """Return stats for one block size from JSON-loaded per-block-size stats."""
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


def _final_block_nfe_components(record):
    """Compute block-batching NFE from the selected final block-size branch only."""
    block_results = record.get('block_results') or {}
    final_block_size = block_results.get('final_block_size', record.get('final_block_size'))
    per_block_size = (
        block_results.get('nfe_per_block_size') or
        record.get('nfe_per_block_size') or
        {}
    )
    values = _block_size_entry(per_block_size, final_block_size)
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
    return final_block_size, nfe_init, nfe_block, nfe_refresh


def _normalize_block_batching_record(data):
    """Normalize older block-batching records that stored aggregate branch NFE."""
    components = _final_block_nfe_components(data)
    if components is not None:
        final_block_size, nfe_init, nfe_block, nfe_refresh = components
        data['final_block_size'] = final_block_size
        data['nfe_init'] = nfe_init
        data['nfe_block'] = nfe_block
        data['nfe_refresh'] = nfe_refresh
        data['nfe'] = nfe_init + nfe_block + nfe_refresh

    if 'block_sizes' in data:
        data.setdefault('candidate_block_sizes', data['block_sizes'])
        data.pop('block_sizes', None)
    data['block_length'] = 'block_batching'
    return data


def _task_dir(task):
    """Map logical task name to results directory name."""
    return {'math': 'math', 'mbpp': 'mbpp'}.get(task, task)


def load_records(results_dir, model, task, variant=None):
    records = []
    task_d = _task_dir(task)
    if variant:
        pattern = os.path.join(results_dir, model, task_d, variant, 'rank_*.jsonl')
    else:
        patterns = [
            os.path.join(results_dir, model, task_d, 'bs*', 'rank_*.jsonl'),
            os.path.join(results_dir, model, task_d, 'entropy', 'bs*', 'rank_*.jsonl'),
            os.path.join(results_dir, model, task_d, 'confidence', 'bs*', 'rank_*.jsonl'),
        ]
        for pattern in patterns:
            for path in sorted(glob.glob(pattern)):
                bs = int(os.path.basename(os.path.dirname(path)).lstrip('bs'))
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(data, dict):
                            data.setdefault('block_length', bs)
                            records.append(data)
        return records
    for path in sorted(glob.glob(pattern)):
        dir_name = os.path.basename(os.path.dirname(path))
        if not variant:
            bs = int(dir_name.lstrip('bs'))
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    if not variant:
                        data.setdefault('block_length', bs)
                    elif variant == 'baseline':
                        data['block_length'] = 'baseline'
                    elif variant == 'block_batching':
                        data = _normalize_block_batching_record(data)
                    else:
                        data.setdefault('block_length', _variant_label(dir_name))
                    records.append(data)
    return records


def _record_key(r, ablation_by_dir=False):
    """Grouping key: block_sizes list (block batching) or block_length (ablation).

    When ablation_by_dir=True, block_length (set from the variant directory name)
    takes priority over block_sizes. Use this when ablating a non-block-size parameter
    (e.g. sync_threshold, refresh_block_size) where all variants share the same block_sizes.
    """
    if ablation_by_dir and r.get('block_length') is not None:
        return str(r['block_length'])
    bs = r.get('block_sizes')
    if bs is not None:
        return '-'.join(str(b) for b in bs) if isinstance(bs, list) else str(bs)
    return r.get('block_length', 'unknown')


def compute_stats(records, ablation_by_dir=False):
    groups = defaultdict(list)
    for r in records:
        groups[_record_key(r, ablation_by_dir=ablation_by_dir)].append(r)

    rows = []
    for bl in sorted(groups):
        recs = groups[bl]
        n = len(recs)

        correct_vals = [r.get('correct') for r in recs if 'correct' in r]
        if correct_vals:
            acc = sum(bool(v) for v in correct_vals) / len(correct_vals)
        else:
            acc = math.nan

        nfe_vals = [r['nfe'] for r in recs if 'nfe' in r]
        avg_nfe = sum(nfe_vals) / len(nfe_vals) if nfe_vals else math.nan

        wall_vals = [r['wall_s'] for r in recs if 'wall_s' in r]
        avg_lat = sum(wall_vals) / len(wall_vals) if wall_vals else math.nan

        tok_vals = [r['tokens_generated'] for r in recs if 'tokens_generated' in r]
        avg_tok = sum(tok_vals) / len(tok_vals) if tok_vals else math.nan

        avg_tps_denom = avg_tok if not math.isnan(avg_tok) else 256.0
        avg_tps = avg_tps_denom / avg_lat if not math.isnan(avg_lat) and avg_lat > 0 else math.nan

        rows.append((bl, n, acc, avg_nfe, avg_tok, avg_lat, avg_tps))
    return rows


def fmt(x, fmt_str):
    return f'{x:{fmt_str}}' if not (isinstance(x, float) and math.isnan(x)) else '     N/A'


def print_table(model, task, rows):
    print(f"\n{'='*85}")
    print(f"  Model: {model.upper()}   Task: {task.upper()}")
    print(f"{'='*85}")
    print(f"  {'BlockLen':>8}  {'N':>5}  {'Acc':>6}  {'AvgNFE':>8}  {'AvgTok':>8}  {'AvgLat(s)':>10}  {'AvgTPS':>8}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
    for bl, n, acc, nfe, tok, lat, tps in rows:
        print(f"  {bl:>8}  {n:>5}  {fmt(acc, '6.3f')}  {fmt(nfe, '8.1f')}  {fmt(tok, '8.1f')}  {fmt(lat, '10.2f')}  {fmt(tps, '8.1f')}")


def print_accu_table(model, task, rows):
    print(f"\n{'='*50}")
    print(f"  Accuracy  |  Model: {model.upper()}   Task: {task.upper()}")
    print(f"{'='*50}")
    print(f"  {'BlockLen':>8}  {'N':>5}  {'Accuracy':>10}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*10}")
    for bl, n, acc in rows:
        print(f"  {bl:>8}  {n:>5}  {fmt(acc, '10.4f')}")


_MODEL_CODE_DIRS = {
    'llada_1p5': 'llada',
    'llada_1p5_instruct': 'llada',
}

_MODEL_ENV = {
    'llada_1p5': ('GSAI-ML/LLaDA-1.5', 'llada_1p5'),
    'llada_1p5_instruct': ('GSAI-ML/LLaDA-1.5', 'llada_1p5_instruct'),
}


def _add_model_to_path(model):
    model_dir = os.path.join(PROJECT_ROOT, _MODEL_CODE_DIRS.get(model, model))
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)


def evaluate_humaneval_accu(results_dir, model, task='humaneval', variant=None):
    """Compute pass@1 for HumanEval or MBPP via postprocess_code.py logic."""
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    _add_model_to_path(model)
    from sanitize import sanitize  # noqa: PLC0415 — intentional late import
    import evaluate as hf_evaluate
    code_eval = hf_evaluate.load("code_eval")

    task_d = _task_dir(task)
    rows = []
    if variant:
        matched = sorted(glob.glob(os.path.join(results_dir, model, task_d, variant)))
        dirs_and_labels = [(d, _variant_label(os.path.basename(d))) for d in matched if os.path.isdir(d)]
    else:
        seen = set()
        dirs_and_labels = []
        for pattern in [
            os.path.join(results_dir, model, task_d, 'bs*'),
            os.path.join(results_dir, model, task_d, 'entropy', 'bs*'),
            os.path.join(results_dir, model, task_d, 'confidence', 'bs*'),
        ]:
            for bs_dir in sorted(glob.glob(pattern),
                                 key=lambda p: int(os.path.basename(p).lstrip('bs'))):
                key = (os.path.basename(bs_dir), os.path.dirname(bs_dir))
                if key not in seen:
                    seen.add(key)
                    dirs_and_labels.append((bs_dir, int(os.path.basename(bs_dir).lstrip('bs'))))
    for bs_dir, bl in dirs_and_labels:
        if not os.path.isdir(bs_dir):
            continue
        samples_files = sorted(
            glob.glob(os.path.join(bs_dir, 'lm_eval', f'samples_{task_d}_*.jsonl')) +
            glob.glob(os.path.join(bs_dir, 'lm_eval', '*', f'samples_{task_d}_*.jsonl'))
        )
        if not samples_files:
            continue
        samples_file = samples_files[-1]

        data = []
        with open(samples_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not data:
            continue

        references = [sample['target'] for sample in data]
        predictions = [
            [sanitize(
                sample['doc']['prompt'] + "\n" +
                sample['resps'][0][0].split('```python\n', 1)[-1].split('```')[0],
                sample['doc']['entry_point']
            )]
            for sample in data
        ]

        result = code_eval.compute(references=references, predictions=predictions, k=[1])
        acc = result[0]['pass@1'] if references else math.nan
        rows.append((bl, len(data), acc))

    return rows


def evaluate_mbpp_accu(results_dir, model, variant=None):
    """Compute pass@1 for MBPP using the same code_eval logic as HumanEval.

    lm_eval MBPP samples format:
      sample['doc']['prompt']     — problem description (optional prefix for completion)
      sample['resps'][0][0]       — model's raw completion
      sample['target']            — test assertions string
    """
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    _add_model_to_path(model)
    from sanitize import sanitize  # noqa: PLC0415
    import evaluate as hf_evaluate
    code_eval = hf_evaluate.load("code_eval")

    task_d = 'mbpp'
    rows = []
    if variant:
        matched = sorted(glob.glob(os.path.join(results_dir, model, task_d, variant)))
        dirs_and_labels = [(d, _variant_label(os.path.basename(d))) for d in matched if os.path.isdir(d)]
    else:
        seen = set()
        dirs_and_labels = []
        for pattern in [
            os.path.join(results_dir, model, task_d, 'bs*'),
            os.path.join(results_dir, model, task_d, 'entropy', 'bs*'),
            os.path.join(results_dir, model, task_d, 'confidence', 'bs*'),
        ]:
            for bs_dir in sorted(glob.glob(pattern),
                                 key=lambda p: int(os.path.basename(p).lstrip('bs'))):
                key = (os.path.basename(bs_dir), os.path.dirname(bs_dir))
                if key not in seen:
                    seen.add(key)
                    dirs_and_labels.append((bs_dir, int(os.path.basename(bs_dir).lstrip('bs'))))
    for bs_dir, bl in dirs_and_labels:
        if not os.path.isdir(bs_dir):
            continue
        samples_files = sorted(
            glob.glob(os.path.join(bs_dir, 'lm_eval', 'samples_mbpp_*.jsonl')) +
            glob.glob(os.path.join(bs_dir, 'lm_eval', '*', 'samples_mbpp_*.jsonl'))
        )
        if not samples_files:
            continue
        data = []
        with open(samples_files[-1]) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if not data:
            continue

        references = [sample['target'] for sample in data]
        predictions = []
        for sample in data:
            completion = sample['resps'][0][0]
            entry_point = None
            test_list = sample['doc'].get('test_list', [])
            if test_list:
                import re as _re
                m = _re.search(r'assert\s+(\w+)\s*\(', test_list[0] if isinstance(test_list, list)
                               else test_list)
                if m:
                    entry_point = m.group(1)
            clean = completion.split('```python\n', 1)[-1].split('```')[0]
            clean = clean.split('[DONE]', 1)[0]
            predictions.append([sanitize(clean, entry_point)])

        result = code_eval.compute(references=references, predictions=predictions, k=[1])
        acc = result[0]['pass@1'] if references else math.nan
        rows.append((bl, len(data), acc))
    return rows


def _math_result_sample_count(data):
    n_samples = data.get('n-samples') or {}
    total = 0
    for value in n_samples.values():
        if isinstance(value, dict):
            total += int(value.get('effective') or value.get('original') or 0)
    return total


def _latest_math_result_files(result_files):
    shard_groups = defaultdict(list)
    for path in result_files:
        shard = next((part for part in path.split(os.sep) if re.fullmatch(r'shard\d+', part)), None)
        if shard:
            shard_groups[shard].append(path)
    if shard_groups:
        return [sorted(files)[-1] for _, files in sorted(shard_groups.items())]
    return [sorted(result_files)[-1]]


def _math_acc_from_result_files(result_files):
    weighted_acc = 0.0
    total = 0
    fallback_accs = []
    for path in _latest_math_result_files(result_files):
        with open(path) as f:
            data = json.load(f)
        math_res = data.get('results', {}).get('minerva_math', {})
        acc = math_res.get('math_verify,none')
        if acc is None:
            acc = math_res.get('exact_match,none')
        if acc is None or math.isnan(float(acc)):
            continue
        n = _math_result_sample_count(data)
        if n:
            weighted_acc += float(acc) * n
            total += n
        else:
            fallback_accs.append(float(acc))
    if total:
        return weighted_acc / total
    if fallback_accs:
        return sum(fallback_accs) / len(fallback_accs)
    return math.nan


def evaluate_math_accu(results_dir, model, variant=None):
    """Compute accuracy for Minerva Math from lm_eval result artifacts."""
    task_d = 'math'
    rows = []
    if variant:
        matched = sorted(glob.glob(os.path.join(results_dir, model, task_d, variant)))
        dirs_and_labels = [(d, _variant_label(os.path.basename(d))) for d in matched if os.path.isdir(d)]
    else:
        seen = set()
        dirs_and_labels = []
        for pattern in [
            os.path.join(results_dir, model, task_d, 'bs*'),
            os.path.join(results_dir, model, task_d, 'entropy', 'bs*'),
            os.path.join(results_dir, model, task_d, 'confidence', 'bs*'),
        ]:
            for bs_dir in sorted(glob.glob(pattern),
                                 key=lambda p: int(os.path.basename(p).lstrip('bs'))):
                key = (os.path.basename(bs_dir), os.path.dirname(bs_dir))
                if key not in seen:
                    seen.add(key)
                    dirs_and_labels.append((bs_dir, int(os.path.basename(bs_dir).lstrip('bs'))))
    for bs_dir, bl in dirs_and_labels:
        if not os.path.isdir(bs_dir):
            continue

        result_files = sorted(
            glob.glob(os.path.join(bs_dir, 'lm_eval', 'results_*.json')) +
            glob.glob(os.path.join(bs_dir, 'lm_eval', '*', 'results_*.json')) +
            glob.glob(os.path.join(bs_dir, 'lm_eval', '*', '*', 'results_*.json'))
        )
        if result_files:
            acc = _math_acc_from_result_files(result_files)
            rec_count = sum(
                1 for path in glob.glob(os.path.join(bs_dir, 'rank_*.jsonl'))
                for line in open(path) if line.strip()
            )
            rows.append((bl, rec_count, float(acc)))
            continue

        samples_files = sorted(
            glob.glob(os.path.join(bs_dir, 'lm_eval', 'samples_minerva_math_*.jsonl')) +
            glob.glob(os.path.join(bs_dir, 'lm_eval', '*', 'samples_minerva_math_*.jsonl'))
        )
        if not samples_files:
            continue
        data = []
        for sf in samples_files:
            with open(sf) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        if not data:
            continue
        acc_vals = [s.get('exact_match', s.get('acc', None)) for s in data
                    if s.get('exact_match') is not None or s.get('acc') is not None]
        acc = sum(acc_vals) / len(acc_vals) if acc_vals else math.nan
        rows.append((bl, len(data), acc))
    return rows


def _lmeval_flexible_extract(text):
    """Replicate lm_eval's flexible-extract filter: last number match, stripped."""
    matches = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
    if not matches:
        return ''
    last = matches[-1]
    val = last[0] if last[0] else last[1]
    val = val.replace(',', '').replace('$', '')
    val = re.sub(r'\.$', '', val)
    return val.strip().lower()


def _lmeval_gsm8k_gold(text):
    """Extract gold answer the way lm_eval does: strip everything up to '#### '."""
    val = re.sub(r'(?s).*#### ', '', text)
    val = val.replace(',', '').replace('$', '')
    val = re.sub(r'\.$', '', val)
    return val.strip().lower()


def _read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def load_per_sample_records(results_dir, model, task):
    """Return {sample_idx: {block_length: record}} aligned by position."""
    per_sample = defaultdict(dict)
    is_humaneval = (task == 'humaneval')
    for bs_dir in sorted(glob.glob(os.path.join(results_dir, model, task, 'bs*')),
                         key=lambda p: int(os.path.basename(p).lstrip('bs'))):
        bl = int(os.path.basename(bs_dir).lstrip('bs'))
        if is_humaneval:
            files = sorted(glob.glob(os.path.join(bs_dir, 'lm_eval', '*', 'samples_humaneval_*.jsonl')))
            if not files:
                continue
            for rec in _read_jsonl(files[-1]):
                per_sample[rec['doc_id']][bl] = rec
        else:
            idx = 0
            for path in sorted(glob.glob(os.path.join(bs_dir, 'rank_*.jsonl'))):
                for rec in _read_jsonl(path):
                    per_sample[idx][bl] = rec
                    idx += 1
    return per_sample


def evaluate_gsm8k_accu(results_dir, model, variant=None):
    """Compute GSM8K accuracy.

    Primary: read lm_eval results_*.json (written when --output_path is passed).
    Fallback: flexible-extract logic matching lm_eval's exact metric.

    variant: if set, evaluate a named subdirectory (e.g. 'seeded') instead of bs* dirs.
    """
    rows = []

    if variant:
        matched = sorted(glob.glob(os.path.join(results_dir, model, 'gsm8k', variant)))
        dirs_and_labels = [(d, _variant_label(os.path.basename(d))) for d in matched if os.path.isdir(d)]
    else:
        seen = set()
        dirs_and_labels = []
        for pattern in [
            os.path.join(results_dir, model, 'gsm8k', 'bs*'),
            os.path.join(results_dir, model, 'gsm8k', 'entropy', 'bs*'),
            os.path.join(results_dir, model, 'gsm8k', 'confidence', 'bs*'),
        ]:
            for bs_dir in sorted(glob.glob(pattern),
                                 key=lambda p: int(os.path.basename(p).lstrip('bs'))):
                key = (os.path.basename(bs_dir), os.path.dirname(bs_dir))
                if key not in seen:
                    seen.add(key)
                    dirs_and_labels.append((bs_dir, int(os.path.basename(bs_dir).lstrip('bs'))))

    for bs_dir, bl in dirs_and_labels:
        if not os.path.isdir(bs_dir):
            continue

        result_files = sorted(glob.glob(os.path.join(bs_dir, 'lm_eval', '*', 'results_*.json')))
        if result_files:
            with open(result_files[-1]) as f:
                data = json.load(f)
            gsm_res = data.get('results', {}).get('gsm8k', {})
            acc = (gsm_res.get('exact_match,flexible-extract') or
                   gsm_res.get('exact_match,strict-match') or
                   math.nan)
            rec_count = sum(
                1 for path in glob.glob(os.path.join(bs_dir, 'rank_*.jsonl'))
                for line in open(path) if line.strip()
            )
            rows.append((bl, rec_count, float(acc)))
            continue

        from datasets import load_dataset
        if not hasattr(evaluate_gsm8k_accu, '_gold'):
            gsm8k = load_dataset("gsm8k", "main", split="test")
            evaluate_gsm8k_accu._gold = [_lmeval_gsm8k_gold(item['answer']) for item in gsm8k]
        gold_answers = evaluate_gsm8k_accu._gold

        records = []
        for path in sorted(glob.glob(os.path.join(bs_dir, 'rank_*.jsonl'))):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            continue

        correct = 0
        for i, rec in enumerate(records):
            answer = rec.get('answer', '') if isinstance(rec, dict) else str(rec)
            pred = _lmeval_flexible_extract(answer)
            gold = gold_answers[i] if i < len(gold_answers) else ''
            if pred == gold:
                correct += 1

        acc = correct / len(records) if records else math.nan
        rows.append((bl, len(records), acc))

    return rows


def compute_optimal_gsm8k(results_dir, model):
    """
    Returns (opt_nfe, opt_acc, opt_acc_nfe):
      opt_nfe     – avg of per-sample min NFE across block sizes
      opt_acc     – fraction of samples correct in at least one block size
      opt_acc_nfe – avg NFE when choosing the min-NFE correct block (else min NFE)
    """
    from datasets import load_dataset
    if not hasattr(compute_optimal_gsm8k, '_gold'):
        gsm8k = load_dataset("gsm8k", "main", split="test")
        compute_optimal_gsm8k._gold = [_lmeval_gsm8k_gold(item['answer']) for item in gsm8k]
    gold = compute_optimal_gsm8k._gold

    per_sample = load_per_sample_records(results_dir, model, 'gsm8k')
    if not per_sample:
        return math.nan, math.nan, math.nan

    nfe_min_list, acc_list, acc_nfe_list = [], [], []
    for idx in sorted(per_sample):
        bl_recs = per_sample[idx]
        nfes = {bl: rec.get('nfe', math.nan) for bl, rec in bl_recs.items()}
        corrects = {
            bl: (_lmeval_flexible_extract(rec.get('answer', '') if isinstance(rec, dict) else '') == gold[idx])
            for bl, rec in bl_recs.items()
            if idx < len(gold)
        }
        valid_nfes = [v for v in nfes.values() if not math.isnan(v)]
        if not valid_nfes:
            continue
        nfe_min_list.append(min(valid_nfes))
        is_correct = any(corrects.values())
        acc_list.append(float(is_correct))
        if is_correct:
            correct_nfes = [nfes[bl] for bl, ok in corrects.items() if ok and not math.isnan(nfes.get(bl, math.nan))]
            acc_nfe_list.append(min(correct_nfes) if correct_nfes else min(valid_nfes))
        else:
            acc_nfe_list.append(min(valid_nfes))

    n = len(nfe_min_list)
    if n == 0:
        return math.nan, math.nan, math.nan
    return sum(nfe_min_list) / n, sum(acc_list) / n, sum(acc_nfe_list) / n


def compute_optimal_humaneval(results_dir, model):
    """
    Returns (opt_nfe, opt_acc, opt_acc_nfe) using code_eval per sample per block size.
    NFE is sourced from rank_0.jsonl (positionally aligned with lm_eval samples).
    """
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    _add_model_to_path(model)
    from sanitize import sanitize  # noqa: PLC0415
    import evaluate as hf_evaluate
    code_eval = hf_evaluate.load("code_eval")

    resp_per_sample = load_per_sample_records(results_dir, model, 'humaneval')
    nfe_lookup = defaultdict(dict)
    for bs_dir in sorted(glob.glob(os.path.join(results_dir, model, 'humaneval', 'bs*')),
                         key=lambda p: int(os.path.basename(p).lstrip('bs'))):
        bl = int(os.path.basename(bs_dir).lstrip('bs'))
        idx = 0
        for path in sorted(glob.glob(os.path.join(bs_dir, 'rank_*.jsonl'))):
            for rec in _read_jsonl(path):
                nfe_lookup[idx][bl] = rec.get('nfe', math.nan)
                idx += 1

    if not resp_per_sample:
        return math.nan, math.nan, math.nan

    sample_bl_pred = defaultdict(dict)
    for doc_id, bl_recs in resp_per_sample.items():
        for bl, sample in bl_recs.items():
            ref = sample['target']
            raw = sample['resps'][0][0].split('```python\n', 1)[-1].split('```')[0]
            pred = sanitize(sample['doc']['prompt'] + "\n" + raw, sample['doc']['entry_point'])
            sample_bl_pred[doc_id][bl] = (ref, pred)

    all_refs, all_preds, all_keys = [], [], []
    for doc_id in sorted(sample_bl_pred):
        for bl, (ref, pred) in sorted(sample_bl_pred[doc_id].items()):
            all_refs.append(ref)
            all_preds.append([pred])
            all_keys.append((doc_id, bl))

    if not all_refs:
        return math.nan, math.nan, math.nan

    result = code_eval.compute(references=all_refs, predictions=all_preds, k=[1])
    correct_map = defaultdict(dict)
    for i, (doc_id, bl) in enumerate(all_keys):
        passed = result[1][i][0][1].get('passed', False)
        correct_map[doc_id][bl] = passed

    nfe_min_list, acc_list, acc_nfe_list = [], [], []
    for doc_id in sorted(sample_bl_pred):
        nfes = nfe_lookup.get(doc_id, {})
        corrects = correct_map.get(doc_id, {})
        valid_nfes = [v for v in nfes.values() if not math.isnan(v)]
        if not valid_nfes:
            continue
        nfe_min_list.append(min(valid_nfes))
        is_correct = any(corrects.values())
        acc_list.append(float(is_correct))
        if is_correct:
            correct_nfes = [nfes[bl] for bl, ok in corrects.items() if ok and bl in nfes and not math.isnan(nfes[bl])]
            acc_nfe_list.append(min(correct_nfes) if correct_nfes else min(valid_nfes))
        else:
            acc_nfe_list.append(min(valid_nfes))

    n = len(nfe_min_list)
    if n == 0:
        return math.nan, math.nan, math.nan
    return sum(nfe_min_list) / n, sum(acc_list) / n, sum(acc_nfe_list) / n


_BAR_COLORS   = {'gsm8k': '#2E8B57', 'humaneval': '#982020'}
_LINE_COLORS  = {'gsm8k': '#96B050', 'humaneval': '#C86000'}
_LINE_STYLES  = {'gsm8k': ('--', 'o'), 'humaneval': ('-.', 's')}
_TASK_LABELS  = {'gsm8k': 'GSM8K',   'humaneval': 'HumanEval'}
_MODEL_TITLES = {'llada': 'LLaDA',   'dream': 'Dream'}


def generate_plot(plot_data, out_path, optimal_data=None):
    """
    2 panels side-by-side (left=LLaDA, right=Dream).
    Bars  (left y-axis)  = avg NFE,  grouped by task.
    Lines (right y-axis) = accuracy, per task.
    X-axis = block sizes + 'Optimal' group (if optimal_data provided).

    optimal_data: dict[(model, task)] -> (opt_nfe, opt_acc, opt_acc_nfe)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import numpy as np

    models = ['llada', 'dream']
    tasks  = ['gsm8k', 'humaneval']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Block Size Ablation: NFE & Accuracy', fontsize=14, fontweight='bold')

    bar_width = 0.35

    for col, model in enumerate(models):
        ax_nfe = axes[col]
        ax_acc = ax_nfe.twinx()

        all_bls = sorted(set(
            bl for task in tasks
            for bl, nfe, acc in plot_data.get((model, task), [])
        ))
        has_optimal = optimal_data and any(
            (model, task) in optimal_data for task in tasks
        )
        x_labels = (['Optimal'] if has_optimal else []) + [str(b) for b in all_bls]
        n_ticks = len(x_labels)

        if n_ticks == 0:
            ax_nfe.set_visible(False)
            continue

        x = np.arange(n_ticks)
        offsets = [-bar_width / 2, bar_width / 2]

        for t_idx, task in enumerate(tasks):
            points = {bl: (nfe, acc) for bl, nfe, acc in plot_data.get((model, task), [])}
            bar_color  = _BAR_COLORS[task]
            line_color = _LINE_COLORS[task]
            ls, mk     = _LINE_STYLES[task]

            nfe_vals = [points[bl][0] if bl in points and not math.isnan(points[bl][0]) else float('nan')
                        for bl in all_bls]
            acc_vals = [points[bl][1] if bl in points and not math.isnan(points[bl][1]) else float('nan')
                        for bl in all_bls]

            if has_optimal and (model, task) in optimal_data:
                opt_nfe, opt_acc, opt_acc_nfe = optimal_data[(model, task)]
                nfe_vals = [opt_nfe] + nfe_vals
                acc_vals = [opt_acc] + acc_vals
            elif has_optimal:
                nfe_vals = [float('nan')] + nfe_vals
                acc_vals = [float('nan')] + acc_vals

            ax_nfe.bar(x + offsets[t_idx], nfe_vals, bar_width,
                       color=bar_color, alpha=0.82, edgecolor='white', linewidth=0.5)

            cx = x + offsets[t_idx]
            bl_start = 1 if has_optimal else 0
            bl_valid = [(xi, v) for xi, v in zip(cx[bl_start:bl_start + len(all_bls)],
                                                  acc_vals[bl_start:bl_start + len(all_bls)])
                        if not math.isnan(v)]
            if bl_valid:
                lx, ly = zip(*bl_valid)
                ax_acc.plot(lx, ly, color=line_color, linestyle=ls, marker=mk,
                            markersize=6, linewidth=2.0, zorder=5)
            if has_optimal and not math.isnan(acc_vals[0]):
                ax_acc.plot(cx[0], acc_vals[0], marker='*', color=line_color,
                            markersize=12, zorder=6, linestyle='none')

        if has_optimal and len(all_bls) > 0:
            sep_x = 0.5
            ax_nfe.axvline(sep_x, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)

        ax_nfe.set_xticks(x)
        ax_nfe.set_xticklabels(x_labels, fontsize=9)
        ax_nfe.set_xlabel('Block Size', fontsize=10)
        ax_nfe.set_ylabel('Avg NFE', fontsize=10)
        ax_nfe.grid(axis='y', alpha=0.25)
        ax_nfe.spines['top'].set_visible(False)

        ax_acc.set_ylabel('Accuracy', fontsize=10, color='dimgray')
        ax_acc.set_ylim(0, 1)
        ax_acc.tick_params(axis='y', labelcolor='dimgray')
        ax_acc.spines['top'].set_visible(False)

        ax_nfe.set_title(_MODEL_TITLES[model], fontsize=12, fontweight='bold')

        legend_handles = []
        for task in tasks:
            legend_handles.append(
                mpatches.Patch(color=_BAR_COLORS[task], alpha=0.82,
                               label=f'{_TASK_LABELS[task]} NFE'))
            ls, mk = _LINE_STYLES[task]
            legend_handles.append(
                mlines.Line2D([], [], color=_LINE_COLORS[task], linestyle=ls,
                              marker=mk, markersize=5,
                              label=f'{_TASK_LABELS[task]} Accuracy'))
        if has_optimal:
            legend_handles.append(
                mlines.Line2D([], [], color='gray', marker='*', markersize=8,
                              linestyle='none', label='Optimal Accuracy'))
        ax_nfe.legend(handles=legend_handles, fontsize=8, framealpha=0.9,
                      loc='lower right')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'\nPlot saved to: {out_path}')


def generate_scatter_plot(scatter_data, out_path):
    """NFE vs accuracy scatter plot matching the dedicated ablation script style.

    scatter_data: list of (label, avg_nfe, accuracy) tuples, one per ablation variant.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    PAPER_BG   = '#f7f3ea'
    PAPER_EDGE = '#595245'
    MAIN_GREEN = '#6f8a3a'
    LIGHT_GREEN = '#b8c98a'
    MAIN_RED   = '#b33a3a'
    LIGHT_RED  = '#e8a2a2'
    TEXT_DARK  = '#2f2a24'

    scatter_data = sorted(scatter_data, key=lambda t: t[1])
    x_vals = [t[1] for t in scatter_data]
    y_vals = [t[2] for t in scatter_data]
    labels = [str(t[0]) for t in scatter_data]

    frontier = []
    best_acc = -1.0
    for item in sorted(scatter_data, key=lambda t: (t[1], -t[2])):
        if item[2] > best_acc:
            frontier.append(item)
            best_acc = item[2]

    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    fig.patch.set_facecolor(PAPER_BG)
    ax.set_facecolor(PAPER_BG)

    ax.scatter(x_vals, y_vals, s=86, color=LIGHT_GREEN, edgecolors=MAIN_GREEN,
               linewidths=1.2, alpha=0.95, zorder=2)

    if frontier:
        fx = [t[1] for t in frontier]
        fy = [t[2] for t in frontier]
        ax.plot(fx, fy, color=MAIN_RED, linewidth=1.8, alpha=0.95, zorder=3)
        ax.scatter(fx, fy, s=98, color=LIGHT_RED, edgecolors=MAIN_RED,
                   linewidths=1.4, zorder=4)

    for x_val, y_val, label in zip(x_vals, y_vals, labels):
        txt = ax.annotate(label, (x_val, y_val), textcoords='offset points',
                          xytext=(6, 6), fontsize=8, color=TEXT_DARK, zorder=5)
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground=PAPER_BG)])

    ax.set_xlabel('Average NFE', color=TEXT_DARK, fontsize=11)
    ax.set_ylabel('Accuracy', color=TEXT_DARK, fontsize=11)
    ax.grid(color=PAPER_EDGE, alpha=0.16, linewidth=0.8)
    ax.tick_params(colors=TEXT_DARK)
    for spine in ax.spines.values():
        spine.set_color(PAPER_EDGE)
        spine.set_linewidth(1.0)

    ax.scatter([], [], s=86, color=LIGHT_GREEN, edgecolors=MAIN_GREEN,
               linewidths=1.2, label='All variants')
    ax.scatter([], [], s=98, color=LIGHT_RED, edgecolors=MAIN_RED,
               linewidths=1.4, label='Accuracy/NFE frontier')
    legend = ax.legend(frameon=True, facecolor=PAPER_BG, edgecolor=PAPER_EDGE,
                       loc='lower right')
    for text in legend.get_texts():
        text.set_color(TEXT_DARK)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=PAPER_BG)
    print(f'\nScatter plot saved to: {out_path}')
    plt.close(fig)


def _print_seeded_summary(attractors_file):
    """Print avg seeded tokens (6/6-agree attractors) per sample."""
    with open(attractors_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    counts = [len(v.get('attractors', [])) for v in data.values()]
    if not counts:
        print('  [seeded] no attractor data found')
        return
    avg_seeded = sum(counts) / len(counts)
    print(f'  Avg seeded tokens (6/6-agree): {avg_seeded:.1f}  (n={len(counts)})')


_ALGO_MAP = {
    'blockBatching_dream': ('dream', 'block_batching'),
    'blockBatching_llada': ('llada', 'block_batching'),
    'blockBatching_llada_1p5': ('llada_1p5', 'block_batching'),
    'blockBatching_llada_1p5_instruct': ('llada_1p5_instruct', 'block_batching'),
}

_BASELINE_MODELS = ('llada', 'llada_1p5', 'llada_1p5_instruct', 'dream')
_BASELINE_TASKS = ('gsm8k', 'humaneval', 'mbpp', 'math')


def _baseline_result_dir(results_dir, model, task):
    return os.path.join(results_dir, model, _task_dir(task), 'baseline')


def _baseline_has_run(results_dir, model, task):
    base = _baseline_result_dir(results_dir, model, task)
    patterns = [
        os.path.join(base, 'rank_*.jsonl'),
        os.path.join(base, 'lm_eval', 'results_*.json'),
        os.path.join(base, 'lm_eval', '*', 'results_*.json'),
        os.path.join(base, 'lm_eval', 'samples_*.jsonl'),
        os.path.join(base, 'lm_eval', '*', 'samples_*.jsonl'),
    ]
    return any(glob.glob(pattern) for pattern in patterns)


def _baseline_command(model, task):
    log = os.path.join(PROJECT_ROOT, f'nohup_{model}_baseline_{task}.log')
    command = [
        'python',
        os.path.join('blockBatching_ablation', 'eval.py'),
        '--model',
        model,
        '--task',
        task,
        '--method',
        'baseline',
        '--analyze',
    ]
    model_env = _MODEL_ENV.get(model)
    if model_env:
        model_path, model_key = model_env
        command.extend(['--model-path', model_path, '--result-model', model_key])
    return f'nohup {shlex.join(command)} > {shlex.quote(log)} 2>&1 &'


def print_baseline_status(results_dir, models, tasks):
    missing = []
    print(f"\n{'='*72}")
    print('  Baseline Status')
    print(f"{'='*72}")
    print(f"  {'Model':<8}  {'Task':<10}  {'Status':<8}  Result dir")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*36}")
    for model in models:
        for task in tasks:
            status = 'DONE' if _baseline_has_run(results_dir, model, task) else 'MISSING'
            if status == 'MISSING':
                missing.append((model, task))
            print(f"  {model:<8}  {task:<10}  {status:<8}  {_baseline_result_dir(results_dir, model, task)}")

    print(f"\n{'='*72}")
    print('  Commands for Missing Baselines')
    print(f"{'='*72}")
    if not missing:
        print('  All requested baselines have result files.')
        return
    for model, task in missing:
        print(_baseline_command(model, task))


def main():
    p = argparse.ArgumentParser(
        description='Summarize block-batching ablation results: NFE, tokens generated, latency, accuracy.')
    p.add_argument('--results-dir', default=os.path.join(os.path.dirname(__file__), 'results'))
    p.add_argument('--model', choices=['llada', 'llada_1p5', 'llada_1p5_instruct', 'dream'],
                   default=None,
                   help='Filter to one model (llada / llada_1p5 / llada_1p5_instruct / dream)')
    p.add_argument('--task', choices=['gsm8k', 'humaneval', 'mbpp', 'math'], default=None,
                   help='Filter to one task. Use "math" for Minerva Math (all subjects).')
    p.add_argument('--algo', choices=list(_ALGO_MAP.keys()), default=None,
                   help='Convenience alias: blockBatching_dream, blockBatching_llada, '
                        'blockBatching_llada_1p5, or blockBatching_llada_1p5_instruct. '
                        'Sets --model and --variant automatically.')
    p.add_argument('--evaluate_accu', action='store_true',
                   help='Evaluate accuracy: HumanEval via postprocess+code_eval, GSM8K via lm_eval logic')
    p.add_argument('--plot', action='store_true',
                   help='Generate NFE-vs-accuracy bar chart (requires --evaluate_accu)')
    p.add_argument('--scatter', action='store_true',
                   help='Generate NFE-vs-accuracy scatter plot like the dedicated ablation scripts '
                        '(requires --evaluate_accu and --variant with a glob pattern)')
    p.add_argument('--plot-out', default=os.path.join(PROJECT_ROOT, 'assets', 'ablations', 'blocksize_ablation.png'),
                   help='Output path for the plot/scatter image')
    p.add_argument('--ablation-by-dir', action='store_true',
                   help='Group records by variant directory name (trailing number) instead of '
                        'block_sizes. Use when ablating a non-block-size parameter such as '
                        'sync_threshold or refresh_block_size where all variants share the same block_sizes.')
    p.add_argument('--variant', default=None,
                   help='Analyze a named result subdirectory (e.g. block_batching) instead of bs* dirs')
    p.add_argument('--attractors-file', default=None,
                   help='Path to attractors JSON; prints avg seeded tokens alongside NFE table')
    p.add_argument('--baseline-status', action='store_true',
                   help='Print baseline completion status and commands for missing model/task baselines')
    args = p.parse_args()

    if args.plot and not args.evaluate_accu:
        p.error('--plot requires --evaluate_accu')
    if args.scatter and not args.evaluate_accu:
        p.error('--scatter requires --evaluate_accu')

    if args.algo and args.algo in _ALGO_MAP:
        algo_model, algo_variant = _ALGO_MAP[args.algo]
        if args.model is None:
            args.model = algo_model
        if args.variant is None:
            args.variant = algo_variant

    _TASK_DIR = {'math': 'math', 'mbpp': 'mbpp', 'gsm8k': 'gsm8k', 'humaneval': 'humaneval'}

    models = [args.model] if args.model else ['llada', 'dream']
    tasks = [args.task] if args.task else ['gsm8k', 'humaneval', 'mbpp', 'math']

    if args.baseline_status:
        print_baseline_status(args.results_dir, models, tasks)
        if not (args.evaluate_accu or args.plot or args.variant or args.algo):
            return

    stats_map = {}
    accu_map = {}

    for model in models:
        for task in tasks:
            records = load_records(args.results_dir, model, task, variant=args.variant)
            if not records:
                print(f"\n[skip] {model}/{task}: no records found in {args.results_dir}")
                continue
            rows = compute_stats(records, ablation_by_dir=args.ablation_by_dir)
            stats_map[(model, task)] = rows
            print_table(model, task, rows)

            if args.variant:
                af = args.attractors_file or os.path.join(
                    os.path.dirname(__file__), f'attractors_{model}_{task}_full.json')
                if os.path.exists(af):
                    _print_seeded_summary(af)

            if args.evaluate_accu:
                if task == 'humaneval':
                    accu_rows = evaluate_humaneval_accu(args.results_dir, model,
                                                        task=task, variant=args.variant)
                elif task == 'mbpp':
                    accu_rows = evaluate_mbpp_accu(args.results_dir, model, variant=args.variant)
                elif task == 'math':
                    accu_rows = evaluate_math_accu(args.results_dir, model, variant=args.variant)
                else:
                    accu_rows = evaluate_gsm8k_accu(args.results_dir, model, variant=args.variant)

                if accu_rows:
                    accu_map[(model, task)] = accu_rows
                    print_accu_table(model, task, accu_rows)
                else:
                    print(f"\n[skip accuracy] {model}/{task}: no evaluation data found")
    print()

    if args.scatter and accu_map:
        scatter_points = []
        for key, accu_rows in accu_map.items():
            stat_rows = stats_map.get(key, [])
            nfe_by_bl = {str(bl): nfe for bl, _, _, nfe, _, _, _ in stat_rows}
            for bl, _, acc in accu_rows:
                nfe = nfe_by_bl.get(str(bl), math.nan)
                if not math.isnan(acc) and not math.isnan(nfe):
                    scatter_points.append((bl, nfe, acc))
        if scatter_points:
            generate_scatter_plot(scatter_points, args.plot_out)
        else:
            print('[skip scatter] no valid (NFE, accuracy) pairs found')

    if args.plot and accu_map:
        plot_data = {}
        for key, accu_rows in accu_map.items():
            stat_rows = stats_map.get(key, [])
            nfe_by_bl = {bl: nfe for bl, _, _, nfe, _, _, _ in stat_rows}
            points = [
                (bl, nfe_by_bl.get(bl, math.nan), acc)
                for bl, _, acc in accu_rows
            ]
            plot_data[key] = points

        optimal_data = {}
        for model in models:
            for task in tasks:
                if (model, task) not in plot_data:
                    continue
                print(f'\nComputing optimal for {model}/{task}...')
                try:
                    if task == 'gsm8k':
                        opt = compute_optimal_gsm8k(args.results_dir, model)
                    else:
                        opt = compute_optimal_humaneval(args.results_dir, model)
                    if not any(math.isnan(v) for v in opt):
                        optimal_data[(model, task)] = opt
                        print(f'  opt_nfe={opt[0]:.1f}  opt_acc={opt[1]:.4f}  opt_acc_nfe={opt[2]:.1f}')
                except Exception as e:
                    print(f'  [skip optimal {model}/{task}]: {e}')

        generate_plot(plot_data, args.plot_out, optimal_data=optimal_data or None)


if __name__ == '__main__':
    main()
