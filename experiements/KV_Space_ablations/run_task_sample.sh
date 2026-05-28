#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
cd "${ROOT}"

if [ -n "${CONDA_SH:-}" ]; then source "${CONDA_SH}"; fi
if [ -n "${CONDA_ENV:-}" ]; then conda activate "${CONDA_ENV}"; fi
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

TASK="${TASK:-humaneval}"
SAMPLE_ID="${SAMPLE_ID:-1}"
MODE="${MODE:-both}"
BLOCK_SIZES="${BLOCK_SIZES:-4-8-16-32-64-128}"
DEVICE="${DEVICE:-cuda}"
GEN_LENGTH="${GEN_LENGTH:-256}"
STEPS="${STEPS:-256}"
THRESHOLD="${THRESHOLD:-0.9}"
SKETCH_DIM="${SKETCH_DIM:-256}"
RAW_SNAPSHOT_LIMIT="${RAW_SNAPSHOT_LIMIT:-4}"
VECTOR_SOURCE="${VECTOR_SOURCE:-sketch}"
OUTPUT_DIR="${OUTPUT_DIR:-${HF_HOME:-${HOME}/.cache/huggingface}/kv_space}"
FIGURES_DIR="${FIGURES_DIR:-${ROOT}/assets/kv_space}"

python experiements/KV_Space_ablations/run_task_sample.py \
  --task            "${TASK}" \
  --sample-id       "${SAMPLE_ID}" \
  --mode            "${MODE}" \
  --device          "${DEVICE}" \
  --cache-dir "${CACHE_DIR:-${HF_HOME:-${HOME}/.cache/huggingface}}" \
  --output-dir      "${OUTPUT_DIR}" \
  --figures-dir     "${FIGURES_DIR}" \
  --block-sizes     "${BLOCK_SIZES}" \
  --gen-length      "${GEN_LENGTH}" \
  --steps           "${STEPS}" \
  --threshold       "${THRESHOLD}" \
  --sketch-dim      "${SKETCH_DIM}" \
  --raw-snapshot-limit "${RAW_SNAPSHOT_LIMIT}" \
  --vector-source   "${VECTOR_SOURCE}" \
  "$@"
