#!/usr/bin/env bash
# AutoFoam-Lite — collect training data → QLoRA fine-tune → merge adapter.
#
# Usage (run inside tmux to keep it alive):
#   tmux new -s autofoam -d
#   tmux send-keys -t autofoam 'bash scripts/run_pipeline.sh' Enter
#
# Optional env vars:
#   GEN_PID=<pid>   wait for a background data-generation job to finish first
#   GPU=0           CUDA device for training and merge (default: 0)
#   EPOCHS=3        number of QLoRA training epochs

set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$PROJ/data/pipeline.log"
GEN_PID=${GEN_PID:-0}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-3}

mkdir -p "$PROJ/data"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cd "$PROJ"
log "AutoFoam-Lite pipeline started  GPU=$GPU  EPOCHS=$EPOCHS"

# ── Step 1: Wait for data generation if running in background ────────────────
if [ "$GEN_PID" -gt 0 ] && kill -0 "$GEN_PID" 2>/dev/null; then
    log "Waiting for data generation (PID $GEN_PID)..."
    while kill -0 "$GEN_PID" 2>/dev/null; do
        N=$(wc -l < data/dataset/expert_train.jsonl 2>/dev/null || echo 0)
        log "  collected $N examples so far..."
        sleep 30
    done
fi
N=$(wc -l < data/dataset/expert_train.jsonl 2>/dev/null || echo 0)
log "Data ready — $N examples"

# ── Step 2: QLoRA fine-tuning ─────────────────────────────────────────────────
log "Starting QLoRA training (epochs=$EPOCHS)..."
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/train_qlora.py \
    --epochs "$EPOCHS" --lora-r 64 --lora-alpha 128 --min-score 0.5 \
    2>&1 | tee -a "$LOG"
log "Training complete."

# ── Step 3: Merge adapter into base model ─────────────────────────────────────
log "Merging LoRA adapter..."
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/merge_adapter.py \
    2>&1 | tee -a "$LOG"
log "Merge complete. Point OPENFOAM_AGENT_LLM_OVERRIDE at the merged path to use it."
log "Pipeline finished. Full log: $LOG"
