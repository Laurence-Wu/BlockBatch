"""Sampling, formatting, and evaluation helpers for Dream quick-eval scripts."""
import re
import random
import types


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_docs(task_name: str):
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    task = tm.load_task_or_group(task_name)[task_name]
    return list(task.dataset["test"])


def format_gsm8k(doc: dict) -> dict:
    answer_line = doc["answer"].split("####")[-1].strip().replace(",", "")
    try:
        answer = int(float(answer_line))
    except ValueError:
        answer = answer_line
    return {"question": doc["question"], "answer": answer}


def make_humaneval_evaluator(doc: dict) -> dict:
    problem = {k: doc[k] for k in ("task_id", "prompt", "test", "entry_point")}

    def evaluate(solution: str, timeout: float = 5.0) -> dict:
        import os
        from human_eval.execution import check_correctness
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        res = check_correctness(problem, solution, timeout)
        return {"passed": res["passed"], "result": res["result"]}

    return {"prompt": doc["prompt"], "entry_point": doc["entry_point"],
            "evaluate": evaluate}


def sample_pairs(task_name: str, formatter, n: int = 2, seed: int = None):
    if seed is not None:
        random.seed(seed)
    docs = load_docs(task_name)
    return [formatter(d) for d in random.sample(docs, min(n, len(docs)))]


# ── Response parsing ──────────────────────────────────────────────────────────

def extract_last_number(text: str):
    nums = re.findall(r"-?[\d]+\.?\d*", text.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


# ── Model loading ─────────────────────────────────────────────────────────────

def load_dream_model(device: str, cache_dir: str = None):
    import torch
    import transformers
    from model.modeling_dream import DreamModel
    from model.generation_utils_block import DreamGenerationMixin

    model_id = "Dream-org/Dream-v0-Instruct-7B"
    model = (
        DreamModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            trust_remote_code=True, cache_dir=cache_dir,
        ).eval().to(device)
    )
    model.diffusion_generate = types.MethodType(
        DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_dir)
    return model, tokenizer


def make_chat_prompt(tokenizer, question: str) -> str:
    msgs = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False)


def tokenize_prompt(tokenizer, prompt_text: str, device: str):
    return tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
