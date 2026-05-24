#!/usr/bin/env bash
# AutoFoam-Lite — parallel full-catalog evaluation across multiple GPUs.
#
# Usage:
#   bash scripts/run_full_test_parallel.sh
#
# Optional env vars:
#   GPUS=0,1,2,3   comma-separated GPU indices (default: 0)
#   TIMEOUT=120    per-case solver timeout in seconds
set -u

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"

GPUS="${GPUS:-0}"
TIMEOUT="${TIMEOUT:-120}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/logs/full_test_${STAMP}"
mkdir -p "$OUTDIR"

source "${OPENFOAM_BASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}" 2>/dev/null || true
export TORCHDYNAMO_DISABLE=1

echo "[full-test] $N shards across GPUs: $GPUS  timeout=${TIMEOUT}s"
echo "[full-test] logs: $OUTDIR"

PIDS=()
for i in "${!GPU_ARR[@]}"; do
    G=${GPU_ARR[$i]}
    LOG="$OUTDIR/shard${i}.log"
    OUT="$OUTDIR/shard${i}.jsonl"
    CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 \
        python3 scripts/full_test_parallel.py \
            --shard "$i/$N" --timeout "$TIMEOUT" --out "$OUT" \
            > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "[full-test] shard $i → GPU $G  pid=${PIDS[-1]}"
done

echo "[full-test] waiting on ${#PIDS[@]} shards..."
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done

python3 - <<EOF
import json, glob, collections
recs = []
for f in sorted(glob.glob("$OUTDIR/shard*.jsonl")):
    for line in open(f):
        recs.append(json.loads(line))
if not recs:
    print("[full-test] no results found")
    exit()
ok   = [r for r in recs if r.get("success")]
avg  = sum(r.get("score", 0) for r in recs) / len(recs)
print(f"\n[full-test] {len(ok)}/{len(recs)} passed  avg score={avg:.3f}")
solver_ct = collections.Counter(r.get("solver", "?") for r in recs)
for s, n in solver_ct.most_common():
    print(f"  {s:<22} {n:>4}")
json.dump(recs, open("$OUTDIR/aggregated.json", "w"), indent=2)
print(f"\n[full-test] results → $OUTDIR/aggregated.json")
EOF

exit $RC
