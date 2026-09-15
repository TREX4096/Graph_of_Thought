#!/usr/bin/env bash
# =====================================================================
# Run the GoT benchmark on a plain GPU server -- NO scheduler, NO root.
# =====================================================================
#
# Use this instead of scripts/slurm/run_got.sbatch when the machine has no
# SLURM (no `sbatch`/`sinfo`), which is the normal situation on a shared lab
# GPU box. Nothing here needs sudo: every dependency lives inside your conda
# environment, and every path is under your own home or project directory.
#
# Quick start:
#     bash scripts/run_direct.sh                       # foreground, watch it
#     bash scripts/run_direct.sh --bg                  # background + logfile
#
# Override anything via environment variables:
#     TASK=set_intersection LENGTH=32 bash scripts/run_direct.sh
#     BACKEND=mock LIMIT=5 bash scripts/run_direct.sh          # free dry run
#     MODEL_ID=Qwen/Qwen2.5-7B-Instruct AGG_K=5 bash scripts/run_direct.sh
#
set -euo pipefail

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
TASK="${TASK:-sorting}"                 # sorting | set_intersection
LENGTH="${LENGTH:-64}"                  # 32 | 64 | 128
LIMIT="${LIMIT:-100}"                   # instances (0 = all)
SCHEMES="${SCHEMES:-io cot cot_sc tot got}"
BACKEND="${BACKEND:-vllm}"              # vllm | hf | llamacpp | mock
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
TP_SIZE="${TP_SIZE:-1}"                 # GPUs for tensor parallelism
AGG_K="${AGG_K:-10}"                    # aggregation attempts -- dominant cost
NUM_CHUNKS="${NUM_CHUNKS:-4}"           # must be a power of two
BRANCH_K="${BRANCH_K:-3}"
N_CTX="${N_CTX:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-auto}"    # 'auto' = measure free memory, else 0..1
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
EXTRA_ARGS="${EXTRA_ARGS:-}"            # anything else to pass through

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-results/run_${TASK}_${LENGTH}_${STAMP}}"
LOGDIR="${LOGDIR:-logs}"
mkdir -p "$OUTDIR" "$LOGDIR"

# ---------------------------------------------------------------------
# Background mode: re-exec ourselves under nohup and exit.
# ---------------------------------------------------------------------
# A long run must survive an SSH disconnect. nohup + setsid detaches it from
# the terminal so closing the laptop does not kill three hours of GPU work.
# (tmux/screen also work; this needs neither to be installed.)
if [ "${1:-}" = "--bg" ]; then
    shift
    LOGFILE="${LOGDIR}/got_${TASK}_${LENGTH}_${STAMP}.log"
    echo "Launching in background -> $LOGFILE"
    RUN_IN_BG=0 setsid nohup bash "${BASH_SOURCE[0]}" "$@" > "$LOGFILE" 2>&1 &
    echo "PID $!"
    echo
    echo "Watch it:   tail -f $LOGFILE"
    echo "Stop it:    kill $!"
    exit 0
fi

# ---------------------------------------------------------------------
# Conda -- no root required, and no `module load` (that is a cluster thing)
# ---------------------------------------------------------------------
# If conda is already active we leave it alone. Otherwise we locate it from
# the conda binary on PATH, which is where `which conda` pointed.
if [ -z "${CONDA_PREFIX:-}" ]; then
    if [ -n "${CONDA_BASE:-}" ]; then
        CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        # .../condabin/conda  or  .../bin/conda  ->  the install root
        CONDA_SH="$(dirname "$(dirname "$(command -v conda)")")/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda not found. Set CONDA_BASE=/path/to/conda and rerun." >&2
        exit 1
    fi

    if [ ! -f "$CONDA_SH" ]; then
        echo "ERROR: no conda.sh at $CONDA_SH -- set CONDA_BASE explicitly." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "${ENV_NAME:-BTP}"
fi

# ---------------------------------------------------------------------
# Caches -- keep large downloads out of a quota-limited home directory
# ---------------------------------------------------------------------
# On a cluster this would be $SCRATCH. On a plain server there is usually no
# such variable, so default to a cache inside the project and let the user
# point HF_HOME elsewhere if the partition is tight.
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
mkdir -p "$HF_HOME"

# Shared box etiquette: pin to specific GPUs so you do not disturb someone
# else's job. Check `nvidia-smi` for which cards are free, then e.g.
#     CUDA_VISIBLE_DEVICES=1 bash scripts/run_direct.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=========================================="
echo "Host        : $(hostname)"
echo "Project     : $PROJECT_ROOT"
echo "Python      : $(which python)"
echo "Task        : $TASK   (length $LENGTH, limit $LIMIT)"
echo "Backend     : $BACKEND   Model: $MODEL_ID"
echo "Schemes     : $SCHEMES"
echo "Structure   : chunks=$NUM_CHUNKS  k=$BRANCH_K  agg_k=$AGG_K"
echo "GPUs        : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  TP=$TP_SIZE"
echo "HF cache    : $HF_HOME"
echo "Output      : $OUTDIR"
echo "Started     : $(date)"
echo "=========================================="

# ---------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------
if [ "$BACKEND" != "mock" ] && [ "$BACKEND" != "llamacpp" ]; then
    command -v nvidia-smi >/dev/null 2>&1 \
        && nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv \
        || echo "WARNING: nvidia-smi not found -- is this machine actually GPU-equipped?"

    # -----------------------------------------------------------------
    # Shared-GPU memory budget
    # -----------------------------------------------------------------
    # vLLM sizes its KV cache against the card's TOTAL memory, not the free
    # space. On a card someone else is already using, the 0.90 default will
    # try to claim more than exists -- killing this run and pressuring
    # theirs. Measure what is actually free and leave a 4 GB head-room
    # margin for the other processes to grow into.
    if [ "$GPU_MEM_UTIL" = "auto" ] && command -v nvidia-smi >/dev/null 2>&1; then
        GPU_MEM_UTIL="$(nvidia-smi --id="${CUDA_VISIBLE_DEVICES%%,*}" \
            --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null \
            | awk -F', *' '{
                  total=$1; free=$2; headroom=4000;
                  u=(free-headroom)/total;
                  if (u>0.90) u=0.90;
                  if (u<0.10) u=0;
                  printf "%.2f", u
              }')"
        [ -z "$GPU_MEM_UTIL" ] && GPU_MEM_UTIL=0.85
        echo "auto gpu-memory-utilization -> $GPU_MEM_UTIL (measured free memory, 4 GB head-room)"
    fi
    if [ "$GPU_MEM_UTIL" = "auto" ]; then GPU_MEM_UTIL=0.85; fi

    # A 7B model in fp16 needs roughly 15 GB of weights before any KV cache.
    if awk "BEGIN{exit !($GPU_MEM_UTIL < 0.10)}"; then
        echo "ERROR: GPU ${CUDA_VISIBLE_DEVICES} has almost no free memory." >&2
        echo "       Pick another card with CUDA_VISIBLE_DEVICES=N, or wait." >&2
        echo "       Do NOT kill another user's process to make room." >&2
        exit 1
    fi

    python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("ERROR: torch not installed. Run: pip install -e '.[hpc]'")
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("ERROR: torch cannot see a GPU. Check CUDA_VISIBLE_DEVICES and the driver.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")
PY
fi

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
DATA="data/${TASK}/${TASK}_${LENGTH}.csv"
if [ ! -f "$DATA" ]; then
    echo "Dataset missing, generating..."
    python scripts/generate_data.py --out data --seed 42 --small
fi
[ -f "$DATA" ] || { echo "ERROR: $DATA still missing -- check TASK/LENGTH." >&2; exit 1; }

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
MODEL_ARGS=()
case "$BACKEND" in
    vllm) MODEL_ARGS=(--model-id "$MODEL_ID" --tensor-parallel-size "$TP_SIZE"
                      --gpu-memory-utilization "$GPU_MEM_UTIL"
                      --max-num-seqs "$MAX_NUM_SEQS") ;;
    hf)   MODEL_ARGS=(--model-id "$MODEL_ID" ${LOAD_IN_4BIT:+--load-in-4bit}) ;;
    llamacpp) MODEL_ARGS=(--model-path "${MODEL_PATH:?set MODEL_PATH for llamacpp}") ;;
    mock) MODEL_ARGS=() ;;
esac

# shellcheck disable=SC2086
python scripts/run_benchmark.py \
    --task "$TASK" \
    --data "$DATA" \
    --limit "$LIMIT" \
    --schemes $SCHEMES \
    --backend "$BACKEND" \
    "${MODEL_ARGS[@]}" \
    --num-chunks "$NUM_CHUNKS" \
    --branching-factor "$BRANCH_K" \
    --aggregation-attempts "$AGG_K" \
    --temperature 1.0 \
    --n-ctx "$N_CTX" \
    --out "$OUTDIR" \
    $EXTRA_ARGS

echo "=========================================="
echo "Finished : $(date)"
echo "Results  : $OUTDIR"
echo "=========================================="
