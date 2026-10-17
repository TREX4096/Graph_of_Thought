#!/usr/bin/env bash
# =====================================================================
# Step-entropy CoT compression on GSM8K and MATH-500.
# =====================================================================
#
# Runs the three-stage protocol from arXiv:2508.03346 on both benchmarks,
# at two pruning ratios, with all three selection strategies. The controls
# (high, random) are not optional: low-entropy pruning preserving accuracy
# only means something if a different selection does not.
#
# Usage:
#     bash scripts/run_entropy_experiment.sh            # foreground
#     tmux new -s entropy -d 'bash scripts/run_entropy_experiment.sh; exec bash'
#
set -uo pipefail          # NOT -e: one dataset failing must not kill the other

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

ENV_NAME="${ENV_NAME:-GOTComp-cu121}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
LIMIT="${LIMIT:-200}"
RATIOS="${RATIOS:-0.5 0.8}"
STRATEGIES="${STRATEGIES:-low high random}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
GPU_MEM="${GPU_MEM:-0.85}"
DATASETS="${DATASETS:-gsm8k_test math500_test}"

# Locate conda without relying on PATH -- a non-interactive shell has not
# sourced ~/.bashrc. Same lookup as run_tmux.sh, for the same reason.
CONDA_SH=""
for _b in "${CONDA_BASE:-}" "$HOME/Time_series_Diffusion" "$HOME/miniconda3" \
          "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
    [ -n "$_b" ] && [ -f "$_b/etc/profile.d/conda.sh" ] && { CONDA_SH="$_b/etc/profile.d/conda.sh"; break; }
done
[ -z "$CONDA_SH" ] && command -v conda >/dev/null 2>&1 && \
    CONDA_SH="$(dirname "$(dirname "$(command -v conda)")")/etc/profile.d/conda.sh"
[ -n "$CONDA_SH" ] || { echo "FATAL: conda.sh not found; set CONDA_BASE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || { echo "FATAL: cannot activate $ENV_NAME" >&2; exit 1; }
python -c "import got" 2>/dev/null || { echo "FATAL: got not installed; pip install -e ." >&2; exit 1; }

# The model is already cached; offline mode keeps a dead proxy from
# turning a GPU job into a network error ten minutes in.
export HF_HOME="${HF_HOME:-$PROJECT/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUT="${OUT:-results/entropy_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

echo "=================================================================="
echo "Model     : $MODEL_ID"
echo "Datasets  : $DATASETS"
echo "Limit     : $LIMIT problems each"
echo "Ratios    : $RATIOS     Strategies: $STRATEGIES"
echo "GPU       : $CUDA_VISIBLE_DEVICES  (mem util $GPU_MEM)"
echo "Output    : $OUT"
echo "Started   : $(date)"
echo "=================================================================="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null

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
        --n-ctx 8192 \
        --ratios $RATIOS \
        --strategies $STRATEGIES \
        --out "$OUT" 2>&1 | tee "$OUT/${DS}.log"
done

echo
echo "=================================================================="
echo "COMPLETE  $(date)"
echo "Results   $OUT"
ls -1 "$OUT"
echo "=================================================================="
