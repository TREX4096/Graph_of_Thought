#!/usr/bin/env bash
# =====================================================================
# SLURM batch script: full GoT benchmark on an HPC GPU node.
# =====================================================================
#
# Submit with:
#     sbatch scripts/slurm/run_got.sbatch
#
# Override the defaults without editing the file:
#     sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-7B-Instruct,LENGTH=64 \
#            scripts/slurm/run_got.sbatch
#
# The paper's second use case (Section 5.2):
#     sbatch --export=ALL,TASK=set_intersection,LENGTH=32 \
#            scripts/slurm/run_got.sbatch
#
# A cost/quality sweep -- cheaper as one job than as three, because the model
# loads once:
#     sbatch --export=ALL,AGG_K=5 scripts/slurm/run_got.sbatch
#
# IMPORTANT -- adapt these before first use. Partition names, module names
# and account codes are cluster-specific; the values below are common
# defaults, not universal ones. Run `sinfo` to list partitions and
# `module avail` to list modules on your cluster.
#
#SBATCH --job-name=got-sorting
#SBATCH --output=logs/got_%j.out
#SBATCH --error=logs/got_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

# ---------------------------------------------------------------------
# Configuration (override via --export)
# ---------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.1-8B-Instruct}"
BACKEND="${BACKEND:-vllm}"
TASK="${TASK:-sorting}"          # sorting | set_intersection
LENGTH="${LENGTH:-64}"
LIMIT="${LIMIT:-100}"
SCHEMES="${SCHEMES:-io cot cot_sc tot got}"
TP_SIZE="${TP_SIZE:-1}"
AGG_K="${AGG_K:-10}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
OUTDIR="${OUTDIR:-results/hpc_${SLURM_JOB_ID:-manual}}"

mkdir -p logs "$OUTDIR"

echo "=========================================="
echo "Job ID      : ${SLURM_JOB_ID:-n/a}"
echo "Node        : $(hostname)"
echo "Model       : $MODEL_ID"
echo "Backend     : $BACKEND"
echo "Task        : $TASK"
echo "Input length: $LENGTH"
echo "Schemes     : $SCHEMES"
echo "Agg k       : $AGG_K   Chunks: $NUM_CHUNKS"
echo "Started     : $(date)"
echo "=========================================="

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
# Most clusters need CUDA loaded explicitly. Adjust the version to match
# what `module avail cuda` reports on your system.
module load cuda/12.1 2>/dev/null || echo "note: no cuda module (may already be present)"

# Activate the conda environment. Adjust the path if conda lives elsewhere
# on the cluster (check with `which conda`).
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate BTP

echo "Python : $(which python)"
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

# Keep HuggingFace caches on scratch, not in $HOME -- home quotas are small
# and a 70B model will blow straight through them.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
# Dataset filenames follow "<task>/<task>_<size>.csv" for both tasks.
DATA="data/${TASK}/${TASK}_${LENGTH}.csv"
if [ ! -f "$DATA" ]; then
  echo "Dataset missing, generating..."
  python scripts/generate_data.py --out data --seed 42
fi
if [ ! -f "$DATA" ]; then
  echo "ERROR: $DATA still missing after generation -- check TASK/LENGTH." >&2
  exit 1
fi

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
srun python scripts/run_benchmark.py \
    --task "$TASK" \
    --data "$DATA" \
    --limit "$LIMIT" \
    --schemes $SCHEMES \
    --backend "$BACKEND" \
    --model-id "$MODEL_ID" \
    --tensor-parallel-size "$TP_SIZE" \
    --num-chunks "$NUM_CHUNKS" \
    --branching-factor 3 \
    --aggregation-attempts "$AGG_K" \
    --temperature 1.0 \
    --out "$OUTDIR"

echo "=========================================="
echo "Finished: $(date)"
echo "Results : $OUTDIR"
echo "=========================================="
