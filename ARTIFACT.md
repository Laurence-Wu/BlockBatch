# BlockBatch Paper Artifact

This repository contains the code and curated artifacts used for the BlockBatch
paper experiments on diffusion language model decoding. The public tree is
intended to be reproducible without exposing local machine paths, cluster
accounts, raw logs, or private scheduler state.

## Artifact Layout

- `llada/`: LLaDA evaluation wrappers and BlockBatch generation code.
- `dream/`: Dream evaluation wrappers.
- `blockBatching_ablation/`: BlockBatch evaluation scripts, analyzers, and
  paper-facing overview figures.
- `experiements/`: Ablation utilities for sync threshold, refresh interval,
  block-size combinations, and KV-space analysis.
- `PUBLIC_RELEASE.md`: privacy and artifact inclusion policy.

Raw result trees are intentionally excluded. Recreate them by running the
evaluation scripts with `RESULTS_DIR` pointing to a local output directory.

## Environment

Use a fresh Python environment and install the repository requirements:

```bash
pip install -r requirements.txt
```

Set cache and output locations explicitly:

```bash
export PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface}"
export RESULTS_DIR="${PROJECT_ROOT}/blockBatching_ablation/results"
```

For execution-based code benchmarks:

```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

## Reproduce The Overview Figure

The intro overview figure is generated from table-derived source data, not from
raw JSONL result dumps:

```bash
python blockBatching_ablation/plot_overview_from_table.py
```

Expected outputs:

- `assets/paper/overview.png`
- `blockBatching_ablation/overview_source_data.json`

## Re-run Evaluations

Single-task examples use public defaults and repo-relative paths:

```bash
python blockBatching_ablation/eval.py --model llada --task gsm8k --method baseline --analyze
python blockBatching_ablation/eval.py --model llada --task gsm8k --method block_batching --analyze
python blockBatching_ablation/eval.py --model dream --task gsm8k --method baseline --analyze
python blockBatching_ablation/eval.py --model dream --task gsm8k --method block_batching --analyze
python blockBatching_ablation/eval.py --model llada --task math --method fast_dllm --block-length 32 --analyze
```

Summarize finished runs with the analyzer:

```bash
python blockBatching_ablation/analyze.py \
  --results-dir "${RESULTS_DIR}" \
  --model llada \
  --task gsm8k \
  --variant block_batching \
  --evaluate_accu
```

For all available summaries in a local result tree:

```bash
RESULTS_DIR="${RESULTS_DIR}" bash blockBatching_ablation/launch/evaluate_all_accuracy.sh
```

For Slurm clusters, generate and submit a machine-neutral launcher:

```bash
python blockBatching_ablation/launch/submit.py \
  --job benchmark \
  --model llada \
  --task gsm8k \
  --method block_batching \
  --partition <partition> \
  --account <account>
```

## Public-Release Checks

Run these before creating a paper artifact tag:

```bash
rg -a -n "<private-user>|<private-project-path>|<cluster-node>|<scheduler-account>|<token-name>" --hidden -g '!**/.git/**'
find . -type f \( -name '*.log' -o -name '*.out' -o -name 'rank_*.jsonl' -o -name 'samples_*.jsonl' -o -name 'nohup.out' -o -name '*.sbatch' \) -not -path './.git/*'
find . -path './.git' -prune -o -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.pdf' -o -iname '*.svg' -o -iname '*.gif' -o -iname '*.webp' -o -iname '*.mp4' \) -not -path './assets/*' -print
find . -path './.git' -prune -o -type f -name '*.py' -print0 | xargs -0 python -m py_compile
```

The first three commands should print no results.
