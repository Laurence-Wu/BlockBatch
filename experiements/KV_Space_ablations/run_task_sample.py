"""Run KV-space ablation for one task sample (gsm8k/mbpp/humaneval) and generate plots.

Prompts are built using lm-eval's doc_to_text/doc_to_target with the same few-shot
counts as the blockBatching_ablation eval scripts:
  gsm8k:    5-shot  (matches eval_gsm8k.sh --num_fewshot 5)
  mbpp:     3-shot  (matches eval_mbpp.sh  --num_fewshot 3)
  humaneval: 0-shot (raw doc["prompt"])

KV event logs (large) -> --output-dir  (default: HF cache / kv_space)
Figure PNGs   (small) -> --figures-dir (default: repo/assets/kv_space)
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
LLADA_DIR = REPO_ROOT / "llada"
TEST_DIR = REPO_ROOT / "test"
for _p in (str(THIS_DIR), str(LLADA_DIR), str(TEST_DIR), str(REPO_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
for _p in reversed((str(THIS_DIR), str(LLADA_DIR), str(TEST_DIR), str(REPO_ROOT))):
    sys.path.insert(0, _p)

from run_kv_space_ablation import run as run_kv_space  # noqa: E402

TASK_FEWSHOT = {"gsm8k": 5, "mbpp": 3, "humaneval": 0}


def get_prompt(task_name: str, doc_idx: int, seed: int = 1234) -> str:
    """Build the lm-eval few-shot context for one test doc, matching eval script format."""
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    task_obj = tm.load_task_or_group(task_name)[task_name]
    test_docs = list(task_obj.dataset["test"])
    doc = test_docs[doc_idx]

    num_fewshot = TASK_FEWSHOT[task_name]
    if num_fewshot == 0:
        return task_obj.doc_to_text(doc)

    rnd = random.Random(seed)
    for split in ("train", "validation", "dev"):
        if split in task_obj.dataset:
            train_docs = list(task_obj.dataset[split])
            break
    else:
        train_docs = []

    fewshot_docs = rnd.sample(train_docs, min(num_fewshot, len(train_docs)))
    context = ""
    for ex in fewshot_docs:
        context += task_obj.doc_to_text(ex) + task_obj.doc_to_target(ex)
    context += task_obj.doc_to_text(doc)
    return context


def _run_analyzer(args, sample_label: str) -> None:
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir) / sample_label
    cmd = [
        sys.executable,
        str(THIS_DIR / "analyze_kv_space.py"),
        "--official-dir", str(output_dir / "official_llada"),
        "--bulk-dir",     str(output_dir / "original_bulk"),
        "--output-dir",   str(figures_dir),
        "--sample-id",    sample_label,
        "--vector-source", args.vector_source,
    ]
    if args.merge_sync_step is not None:
        cmd.extend(["--merge-sync-step", str(args.merge_sync_step)])
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["gsm8k", "mbpp", "humaneval"], default="humaneval")
    parser.add_argument("--sample-id", type=int, default=1)
    parser.add_argument("--mode", choices=["official", "bulk", "both"], default="both")
    parser.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--cache-dir",  default=str(Path.home() / ".cache" / "huggingface"))
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / ".cache" / "huggingface" / "kv_space")
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "assets" / "kv_space")
    parser.add_argument("--block-sizes", default="4-8-16-32-64-128")
    parser.add_argument("--gen-length",  type=int,   default=256)
    parser.add_argument("--steps",       type=int,   default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking",   default="low_confidence")
    parser.add_argument("--mask-id",     type=int,   default=126336)
    parser.add_argument("--threshold",   type=float, default=0.9)
    parser.add_argument("--sketch-dim",  type=int,   default=256)
    parser.add_argument("--raw-snapshot-limit", type=int, default=4)
    parser.add_argument("--vector-source", choices=["sketch", "raw"], default="sketch")
    parser.add_argument("--merge-sync-step", type=int, default=None)
    parser.add_argument("--skip-analyze", action="store_true")
    args = parser.parse_args()

    prompt = get_prompt(args.task, args.sample_id)
    sample_label = f"{args.task}_{args.sample_id}"

    runner_args = argparse.Namespace(
        mode=args.mode,
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        device=args.device,
        prompt=prompt,
        apply_chat_template=True,
        sample_id=sample_label,
        run_id=sample_label,
        output_dir=args.output_dir,
        block_sizes=args.block_sizes,
        gen_length=args.gen_length,
        steps=args.steps,
        temperature=args.temperature,
        remasking=args.remasking,
        mask_id=args.mask_id,
        threshold=args.threshold,
        sketch_dim=args.sketch_dim,
        raw_snapshot_limit=args.raw_snapshot_limit,
    )
    print(f"Running KV ablation: task={args.task} sample_id={args.sample_id} label={sample_label}")
    run_kv_space(runner_args)

    if not args.skip_analyze:
        _run_analyzer(args, sample_label)


if __name__ == "__main__":
    main()
