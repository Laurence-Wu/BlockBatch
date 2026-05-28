"""Sampling, formatting, evaluation helpers shared across generate_*.py scripts."""
import re
import random


# ── Dataset helpers ──────────────────────────────────────────────────────────

def load_docs(task_name: str):
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    task = tm.load_task_or_group(task_name)[task_name]
    return list(task.dataset["test"])


def format_gsm8k(doc: dict) -> dict:
    answer_line = doc["answer"].split("####")[-1].strip()
    return {"question": doc["question"], "answer": answer_line}


def make_humaneval_evaluator(doc: dict) -> dict:
    """Return prompt + evaluate(solution) -> {passed: bool, result: str}."""
    problem = {k: doc[k] for k in ("task_id", "prompt", "test", "entry_point")}
    canonical = doc["canonical_solution"]

    def evaluate(solution: str, timeout: float = 5.0) -> dict:
        """Run human_eval check_correctness; returns {passed, result}."""
        import os
        from human_eval.execution import check_correctness
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        res = check_correctness(problem, solution, timeout)
        return {"passed": res["passed"], "result": res["result"]}

    return {"prompt": doc["prompt"], "entry_point": doc["entry_point"],
            "evaluate": evaluate, "_canonical": canonical}


def sample_pairs(task_name: str, formatter, n: int = 2, seed: int = None):
    if seed is not None:
        random.seed(seed)
    docs = load_docs(task_name)
    return [formatter(d) for d in random.sample(docs, min(n, len(docs)))]


# ── Response parsing ─────────────────────────────────────────────────────────

def extract_last_number(text: str):
    """Extract the last number from a GSM8K-style response (flexible-extract)."""
    nums = re.findall(r"-?[\d]+\.?\d*", text.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def extract_code(text: str) -> str:
    """Extract Python code from markdown fences; return raw text if none found."""
    m = re.search(r"```(?:python)?\n?(.*?)\n?```", text, re.DOTALL)
    return m.group(1) if m else text


# ── Model / tokenizer ─────────────────────────────────────────────────────────

def load_llada_model(device: str = "cuda:0", cache_dir: str = None):
    """Load LLaDA-8B-Instruct with Flash Attention. Returns (model, tokenizer)."""
    import os
    import time
    import logging
    import warnings
    import torch._dynamo
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Suppress inductor OOM during @torch.compile init — falls back to eager F.sdpa
    torch._dynamo.config.suppress_errors = True
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

    _log = logging.getLogger(__name__)

    from model.modeling_llada import LLaDAModelLM
    from transformers import AutoConfig, AutoTokenizer

    # ── Step 1: config ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    config = AutoConfig.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", cache_dir=cache_dir)
    config.flash_attention = True
    _log.info(f"[load] AutoConfig.from_pretrained       {time.perf_counter()-t0:6.2f}s")

    t0 = time.perf_counter()
    _hf_log = logging.getLogger("transformers.modeling_utils")
    _hf_prev = _hf_log.level
    _hf_log.setLevel(logging.ERROR)
    model = LLaDAModelLM.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        config=config,
        cache_dir=cache_dir,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    _hf_log.setLevel(_hf_prev)
    _log.info(f"[load] from_pretrained                   {time.perf_counter()-t0:6.2f}s")

    t0 = time.perf_counter()
    model = model.eval()
    _log.info(f"[load] .eval()                          {time.perf_counter()-t0:6.2f}s")

    # ── Step 3: tokenizer ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, cache_dir=cache_dir
    )
    _log.info(f"[load] AutoTokenizer.from_pretrained    {time.perf_counter()-t0:6.2f}s")

    # ── Step 4: first forward (triggers @torch.compile JIT — the real slow part) ─
    # Run a tiny dummy forward so compilation happens here (labelled), not silently
    # during the first real generation call.
    _log.info("[load] warming up @torch.compile (first forward) …")
    t0 = time.perf_counter()
    dummy = torch.zeros(1, 4, dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(dummy)
    _log.info(f"[load] first forward / compile warm-up  {time.perf_counter()-t0:6.2f}s  ← likely the slow one")

    return model, tokenizer


# ── Prompt helpers ────────────────────────────────────────────────────────────

def make_chat_prompt(tokenizer, question: str) -> str:
    """Apply LLaDA instruct chat template to a single user turn."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=False,
    )


def tokenize_prompt(tokenizer, prompt_text: str, device: str):
    """Tokenize a chat prompt and return a (1, L) tensor on device."""
    import torch
    return torch.tensor(
        tokenizer(prompt_text)["input_ids"], dtype=torch.long
    ).unsqueeze(0).to(device)
