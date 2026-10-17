#!/usr/bin/env bash
# =====================================================================
# Step-entropy experiment on a REASONING model (DeepSeek-R1-Distill-7B).
# =====================================================================
#
# Why this run exists
# -------------------
# The first run used Qwen2.5-7B-Instruct and inverted the paper's result:
# pruning HIGH-entropy steps preserved accuracy better than pruning low.
# The likely reason is visible in the same log - mean chain length was 4.1
# steps on GSM8K and 8.9 on MATH-500, where the paper's models produce
# dozens.
#
# With only four steps there is no redundancy to find, and low entropy
# probably marks the *conclusion* ("So the answer is 18") rather than
# filler: highly predictable given the preceding work, and fatal to
# remove. The paper's reasoning models emit long think-blocks full of
# genuinely redundant hedging, which is where low-entropy = removable
# should hold.
#
# DeepSeek-R1-Distill-Qwen-7B is the same family the paper evaluated and
# is already cached locally, so this tests that explanation directly.
#
# Note the budget change: R1 chains are an order of magnitude longer, so
# max_tokens goes 1024 -> 4096 and the context window with it. Leaving the
# old cap would truncate every chain and silently reproduce the short-chain
# regime we are trying to escape.
#
set -uo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

ENV_NAME="${ENV_NAME:-GOTComp-cu121}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
LIMIT="${LIMIT:-100}"
RATIOS="${RATIOS:-0.5 0.8}"
STRATEGIES="${STRATEGIES:-low high random}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
N_CTX="${N_CTX:-16384}"
GPU_MEM="${GPU_MEM:-0.85}"
TEMPERATURE="${TEMPERATURE:-0.6}"      # DeepSeek's own recommendation
DATASETS="${DATASETS:-gsm8k_test math500_test}"
WAIT_FOR="${WAIT_FOR:-run_entropy_pruning}"

CONDA_SH=""
for _b in "${CONDA_BASE:-}" "$HOME/Time_series_Diffusion" "$HOME/miniconda3" \
          "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
    [ -n "$_b" ] && [ -f "$_b/etc/profile.d/conda.sh" ] && { CONDA_SH="$_b/etc/profile.d/conda.sh"; break; }
done
[ -n "$CONDA_SH" ] || { echo "FATAL: conda.sh not found" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || { echo "FATAL: cannot activate $ENV_NAME" >&2; exit 1; }

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Two vLLM servers cannot share one GPU's memory budget, so wait for the
# in-flight run to release it rather than racing and OOM-ing both.
if [ -n "$WAIT_FOR" ]; then
    echo "waiting for '$WAIT_FOR' to finish before claiming the GPU..."
    while pgrep -f "$WAIT_FOR" >/dev/null 2>&1; do sleep 30; done
    echo "GPU free at $(date)"
    sleep 20                      # let CUDA memory actually be released
fi

OUT="${OUT:-results/entropy_r1_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

echo "=================================================================="
echo "Model     : $MODEL_ID   (reasoning model - long think blocks)"
echo "Datasets  : $DATASETS    Limit: $LIMIT"
echo "Budget    : max_tokens $MAX_TOKENS, ctx $N_CTX, temp $TEMPERATURE"
echo "Output    : $OUT"
echo "Started   : $(date)"
echo "=================================================================="

for DS in $DATASETS; do
    echo
    echo "################ $DS ################"
    # shellcheck disable=SC2086
    python scripts/run_entropy_pruning.py \
        --data "data/math/${DS}.csv" \
        --limit "$LIMIT" \
        --backend vllm \
        --model-id "$MODEL_ID" \
        --gpu-memory-utilization "$GPU_MEM" \
        --max-tokens "$MAX_TOKENS" \
        --n-ctx "$N_CTX" \
        --temperature "$TEMPERATURE" \
        --batch 16 \
        --ratios $RATIOS \
        --strategies $STRATEGIES \
        --out "$OUT" 2>&1 | tee "$OUT/${DS}.log"
done

echo
echo "=================================================================="
echo "COMPLETE  $(date)"
echo "Compare mean steps per chain against the Qwen2.5-Instruct run:"
echo "  if R1 produces many more steps AND low-entropy pruning now wins,"
echo "  the criterion is chain-length dependent - which is the finding."
echo "Results   $OUT"
echo "=================================================================="
