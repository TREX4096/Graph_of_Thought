#!/usr/bin/env bash
# =====================================================================
# Download DeepSeek-R1-Distill-Qwen-7B, then run the entropy experiment.
# =====================================================================
#
# The model's cache directory existed but held only config and tokenizer
# files (39 MB of a ~15 GB model), so vLLM failed with "Cannot find any
# model weights". A directory being present is not evidence that weights
# are - check `du -sh`, not `ls`.
#
# Download needs the network, so HF_HUB_OFFLINE must be off for that step
# and on for the run, otherwise a proxy that drops mid-experiment turns a
# GPU job into a network error.
#
# Usage:
#     tmux new -s r1 -d 'bash scripts/fetch_and_run_r1.sh; exec bash'
#
set -uo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

ENV_NAME="${ENV_NAME:-GOTComp-cu121}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"

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

# ---------------------------------------------------------------------
# 1. Space check before a 15 GB download onto a 95%-full shared disk
# ---------------------------------------------------------------------
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "free space: ${FREE_GB} GB"
if [ "${FREE_GB:-0}" -lt 25 ]; then
    echo "FATAL: need ~25 GB free for a 15 GB model plus working room." >&2
    echo "       This is a shared disk - clear space before retrying." >&2
    exit 1
fi

# ---------------------------------------------------------------------
# 2. Download (network ON)
# ---------------------------------------------------------------------
echo "=================================================================="
echo "Downloading $MODEL_ID  ($(date))"
echo "=================================================================="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
huggingface-cli download "$MODEL_ID" \
    --exclude "*.pth" "*.msgpack" "*.h5" \
    || { echo "FATAL: download failed - is the IITD proxy authenticated?" >&2; exit 1; }

CACHE_DIR="$HF_HOME/hub/models--${MODEL_ID//\//--}"
SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
echo "cached size: $SIZE"
# A weights-free cache is exactly the failure this script exists to avoid.
if ! find "$CACHE_DIR" -name "*.safetensors" -o -name "*.bin" | grep -q .; then
    echo "FATAL: no weight files after download. Aborting rather than" >&2
    echo "       burning GPU time on a model that cannot load." >&2
    exit 1
fi

# ---------------------------------------------------------------------
# 3. Run (network OFF, so a dropped proxy cannot kill the job)
# ---------------------------------------------------------------------
echo
echo "=================================================================="
echo "Starting entropy experiment on $MODEL_ID  ($(date))"
echo "=================================================================="
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WAIT_FOR="" bash scripts/run_entropy_r1.sh
