#!/usr/bin/env bash
# Unified accuracy/NFE summary for blockBatching_ablation results.
#
# Defaults:
#   MODELS="llada dream"
#   TASKS="gsm8k humaneval mbpp math"
#   MODES="ablation baseline block_batching"
#
# Examples:
#   bash blockBatching_ablation/launch/evaluate_all_accuracy.sh
#   TASKS="gsm8k math" bash blockBatching_ablation/launch/evaluate_all_accuracy.sh
#   MODELS="llada" MODES="ablation baseline" bash blockBatching_ablation/launch/evaluate_all_accuracy.sh

set -euo pipefail

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
RESULTS_DIR=${RESULTS_DIR:-${ROOT}/blockBatching_ablation/results}
PYTHON=${PYTHON:-python}

MODELS=${MODELS:-"llada dream"}
TASKS=${TASKS:-"gsm8k humaneval mbpp math"}
MODES=${MODES:-"ablation baseline block_batching"}

cd "${ROOT}"

run_analyze() {
    local model="$1"
    local task="$2"
    local mode="$3"
    local variant=()
    local label

    case "${mode}" in
        ablation|confidence)
            label="confidence ablation"
            ;;
        baseline)
            label="baseline"
            variant=(--variant baseline)
            ;;
        block_batching|bb)
            label="block batching"
            variant=(--variant block_batching)
            ;;
        *)
            echo "[error] unsupported mode: ${mode}" >&2
            echo "        valid modes: ablation baseline block_batching" >&2
            return 2
            ;;
    esac

    echo
    echo "=== ${model^^} ${task^^}: ${label} ==="
    "${PYTHON}" blockBatching_ablation/analyze.py \
        --results-dir "${RESULTS_DIR}" \
        --model "${model}" \
        --task "${task}" \
        "${variant[@]}" \
        --evaluate_accu
}

for model in ${MODELS}; do
    for task in ${TASKS}; do
        for mode in ${MODES}; do
            run_analyze "${model}" "${task}" "${mode}"
        done
    done
done
