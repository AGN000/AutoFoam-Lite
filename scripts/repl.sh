#!/usr/bin/env bash
# AutoFoam-Lite — Interactive REPL
#
# Loads the model once, then accepts prompts in a loop.
#
# Usage:
#   bash scripts/repl.sh           # auto-pick any free GPU
#   GPU=0 bash scripts/repl.sh     # force a specific GPU

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Auto-pick GPU with ≥3 GB free (enough for the 3B model at 4-bit)
if [ -z "${GPU:-}" ]; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null \
          | awk -F', ' '{ gsub(" MiB","",$2); if ($2+0 > 3000) print $1 }' \
          | head -1)
    if [ -z "$GPU" ]; then
        echo "No GPU with ≥3 GB free found. Falling back to CPU (USE_CPU_INFERENCE=1)."
        export USE_CPU_INFERENCE=1
        GPU=""
    fi
fi

export VLLM_GPU_MEM_FRAC=${VLLM_GPU_MEM_FRAC:-0.90}
export TORCHDYNAMO_DISABLE=1

echo "============================================"
echo "  AutoFoam-Lite REPL"
[ -n "$GPU" ] && echo "  GPU          : $GPU" || echo "  Device       : CPU"
echo "  Project      : $PROJ"
echo "  Type 'quit' or Ctrl-D to exit"
echo "============================================"

source "${OPENFOAM_BASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}" 2>/dev/null || true

cd "$PROJ"
[ -n "$GPU" ] && CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python3 -u scripts/repl.py \
              || PYTHONUNBUFFERED=1 python3 -u scripts/repl.py
