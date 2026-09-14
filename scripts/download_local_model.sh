#!/usr/bin/env bash
# =====================================================================
# Download a small quantised GGUF model for LOCAL CPU testing.
# =====================================================================
#
# This machine has ~3.5 GB of RAM in WSL and no CUDA GPU, so we need a
# model that runs on CPU in about 1-2 GB. Q4_K_M quantisation of a 1.5B
# model is the sweet spot: small enough to load, capable enough to follow
# a strict output format.
#
# These are for *local validation only*. Real paper-comparable numbers
# come from the HPC runs with a 8B-70B model (see scripts/slurm/).
#
# Usage:
#   bash scripts/download_local_model.sh            # default: Qwen2.5 1.5B
#   bash scripts/download_local_model.sh llama1b    # smaller alternative
#
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models}"
CHOICE="${1:-qwen1.5b}"

mkdir -p "$MODEL_DIR"

case "$CHOICE" in
  qwen1.5b)
    # ~1.1 GB. Good instruction-following for its size; handles the strict
    # "output only the list" formatting our prompts require.
    REPO="Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    FILE="qwen2.5-1.5b-instruct-q4_k_m.gguf"
    ;;
  llama1b)
    # ~0.8 GB. Use if memory is very tight; weaker at format compliance.
    REPO="bartowski/Llama-3.2-1B-Instruct-GGUF"
    FILE="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ;;
  qwen3b)
    # ~2.0 GB. Noticeably better, but close to this laptop's RAM ceiling.
    REPO="Qwen/Qwen2.5-3B-Instruct-GGUF"
    FILE="qwen2.5-3b-instruct-q4_k_m.gguf"
    ;;
  *)
    echo "Unknown model choice: $CHOICE" >&2
    echo "Options: qwen1.5b (default), llama1b, qwen3b" >&2
    exit 1
    ;;
esac

TARGET="$MODEL_DIR/$FILE"

if [ -f "$TARGET" ]; then
  echo "Already present: $TARGET"
  exit 0
fi

echo "Downloading $REPO / $FILE -> $TARGET"

# huggingface-cli ships with huggingface_hub, which is already installed.
if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" "$FILE" \
    --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
else
  # Fallback to a plain HTTPS fetch if the CLI is unavailable.
  curl -L --fail -o "$TARGET" \
    "https://huggingface.co/${REPO}/resolve/main/${FILE}?download=true"
fi

echo
echo "Done: $TARGET"
echo
echo "Try it with:"
echo "  python scripts/run_benchmark.py --task sorting \\"
echo "      --data data/smoke/sorting_32_small.csv --limit 2 \\"
echo "      --backend llamacpp --model-path $TARGET \\"
echo "      --schemes io got --out results/local"
