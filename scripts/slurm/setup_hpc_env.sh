#!/usr/bin/env bash
# =====================================================================
# One-time environment setup on the HPC cluster.
# =====================================================================
#
# Run this ONCE on a login node (or better, an interactive compute node --
# some clusters forbid heavy pip builds on login nodes):
#
#     bash scripts/slurm/setup_hpc_env.sh
#
# It creates/updates the BTP conda environment with the GPU stack needed
# for real open-source model inference.
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-BTP}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"

echo "=== Loading CUDA module (cluster only) ==="
# `module` is a scheduler-cluster convention. Plain GPU servers do not have it,
# and do not need it -- the driver is already on the system and the CUDA
# runtime arrives with the pip torch wheel. Absence here is not an error.
if command -v module >/dev/null 2>&1; then
  module load "cuda/${CUDA_VERSION}" 2>/dev/null || \
    echo "note: could not load cuda/${CUDA_VERSION} -- check 'module avail cuda'"
else
  echo "note: no 'module' command (normal on a non-cluster server) -- skipping"
fi

echo "=== Locating conda ==="
# Try, in order: an explicit CONDA_BASE, an already-active env, then whatever
# `conda` on PATH resolves to. Hard-coding $HOME/miniconda3 breaks on shared
# lab machines where conda lives under a project directory.
if [ -n "${CONDA_BASE:-}" ]; then
  :
elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_PREFIX}/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="${CONDA_PREFIX}"
elif command -v conda >/dev/null 2>&1; then
  # .../condabin/conda or .../bin/conda -> the install root
  CONDA_BASE="$(dirname "$(dirname "$(command -v conda)")")"
else
  CONDA_BASE="$HOME/miniconda3"
fi

if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  echo "conda not found at ${CONDA_BASE}." >&2
  echo "Set CONDA_BASE=/path/to/conda and rerun, or install miniconda first" >&2
  echo "(no root needed -- it installs entirely inside your home directory):" >&2
  echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" >&2
  echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3" >&2
  exit 1
fi
echo "using conda at ${CONDA_BASE}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "=== Creating/updating env '${ENV_NAME}' ==="
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "environment exists, reusing"
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_NAME}"

echo "=== Installing GPU PyTorch ==="
# cu121 wheels. If your cluster runs CUDA 11.8, swap cu121 -> cu118.
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing the project and its HPC extras ==="
# vLLM pulls in transformers/accelerate itself, but we pin them explicitly
# via the 'hpc' extra so the versions are recorded.
pip install -e ".[hpc]"

echo "=== Verifying ==="
python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
print("GPU count      :", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory / 1e9:.1f} GB")
try:
    import vllm
    print("vLLM           :", vllm.__version__)
except ImportError:
    print("vLLM           : NOT INSTALLED (use --backend hf instead)")
import got
print("got package    :", got.__version__)
PY

echo
echo "=== Choosing a model for your GPU memory ==="
cat <<'EOF'
  GPU RAM   Recommended model                       Flags
  -------   --------------------------------------  ----------------------------
   16 GB    Qwen/Qwen2.5-7B-Instruct                --backend hf --load-in-4bit
   24 GB    meta-llama/Llama-3.1-8B-Instruct        --backend vllm
   40 GB    Qwen/Qwen2.5-32B-Instruct               --backend vllm
   80 GB    meta-llama/Llama-3.1-70B-Instruct       --backend vllm --tensor-parallel-size 2

Note: Llama models are gated on HuggingFace. Accept the licence on the model
page, then authenticate once with:  huggingface-cli login
Qwen models are ungated and need no token -- prefer them if you hit access issues.
EOF

echo
echo "Setup complete. Submit a job with:  sbatch scripts/slurm/run_got.sbatch"
