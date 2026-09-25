#!/usr/bin/env bash
# Verify the environment on the GPU server, then run a free mock benchmark.
#
# Run over SSH:
#     scp scripts/remote_check.sh server:/tmp/ && ssh server bash /tmp/remote_check.sh
#
# A non-interactive SSH session does not source ~/.bashrc, so conda is not on
# PATH and `command -v conda` finds nothing. Locate it explicitly instead of
# assuming the login shell has been set up.
set -uo pipefail

PROJECT="${PROJECT:-$HOME/GOT_Compression/IITD_GOT/Prasoon/Graph_of_Thought}"
ENV_NAME="${ENV_NAME:-GOTComp-cu121}"

cd "$PROJECT" || { echo "no project at $PROJECT"; exit 1; }

# Find conda without relying on PATH.
CONDA_SH=""
for base in "$HOME/Time_series_Diffusion" "$HOME/miniconda3" "$HOME/anaconda3" \
            "$HOME/miniforge3" /opt/conda; do
    if [ -f "$base/etc/profile.d/conda.sh" ]; then CONDA_SH="$base/etc/profile.d/conda.sh"; break; fi
done
if [ -z "$CONDA_SH" ]; then
    found="$(find "$HOME" -maxdepth 4 -name conda.sh -path '*/profile.d/*' 2>/dev/null | head -1)"
    [ -n "$found" ] && CONDA_SH="$found"
fi
[ -n "$CONDA_SH" ] || { echo "ERROR: conda.sh not found"; exit 1; }

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || { echo "ERROR: cannot activate $ENV_NAME"; exit 1; }

echo "=============================================================="
echo "host    : $(hostname)"
echo "commit  : $(git log --oneline -1)"
echo "python  : $(command -v python)"
echo "conda   : $CONDA_SH"
echo "=============================================================="

echo
echo "### torch / GPU ###"
python - <<'PY'
try:
    import torch
    print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} {p.total_memory/1e9:.1f} GB")
except Exception as exc:
    print("torch unavailable:", exc)
PY
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader 2>/dev/null

echo
echo "### test suite ###"
python -m pytest -q 2>&1 | tail -3

echo
echo "### mock benchmark -- confirms the ToT fairness fix ###"
python scripts/run_benchmark.py --task sorting \
    --data data/official/sorting/sorting_064.csv --limit 20 \
    --backend mock --schemes io cot cot_sc tot got \
    --out /tmp/totfix 2>&1 | grep -E "^(scheme|io|cot|cot_sc|tot|got|---)"

echo
echo "### claim check ###"
python scripts/compare_to_paper.py --results /tmp/totfix 2>&1 \
    | grep -E "C3|C4|TOTAL"
