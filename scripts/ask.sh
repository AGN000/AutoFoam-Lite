#!/usr/bin/env bash
# AutoFoam-Lite — one-off simulation from the command line.
#
# Usage:
#   bash scripts/ask.sh "2D lid-driven cavity Re=1000, water, 1m square"
#
# Optional env vars:
#   GPU=0        force a specific GPU
#   TIMEOUT=300  solver wall-clock timeout in seconds
#   RETRIES=2    max self-correction retries

if [ -z "$1" ]; then
    echo "Usage: $0 \"<CFD prompt>\""
    echo "Example: $0 \"2D backward-facing step Re=800, water, step height 0.05m\""
    exit 1
fi

PROMPT="$1"
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMEOUT=${TIMEOUT:-300}
RETRIES=${RETRIES:-2}

# Auto-pick GPU with ≥3 GB free (3B model at 4-bit needs ~2 GB VRAM)
if [ -z "${GPU:-}" ]; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader 2>/dev/null \
          | awk -F', ' '{ gsub(" MiB","",$2); if ($2+0 > 3000) print $1 }' \
          | head -1)
    if [ -z "$GPU" ]; then
        echo "No GPU with ≥3 GB free. Falling back to CPU (USE_CPU_INFERENCE=1)."
        export USE_CPU_INFERENCE=1
        GPU=""
    fi
fi

export VLLM_GPU_MEM_FRAC=${VLLM_GPU_MEM_FRAC:-0.90}
export TORCHDYNAMO_DISABLE=1

echo "============================================"
echo "  AutoFoam-Lite"
[ -n "$GPU" ] && echo "  GPU     : $GPU" || echo "  Device  : CPU"
echo "  Timeout : ${TIMEOUT}s   Retries: $RETRIES"
echo "  Prompt  : $PROMPT"
echo "============================================"

source "${OPENFOAM_BASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}" 2>/dev/null || true

cd "$PROJ"
[ -n "$GPU" ] \
    && CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
       python3 -u scripts/run_agent.py run "$PROMPT" --timeout "$TIMEOUT" --retries "$RETRIES" \
    || PYTHONUNBUFFERED=1 \
       python3 -u scripts/run_agent.py run "$PROMPT" --timeout "$TIMEOUT" --retries "$RETRIES"
