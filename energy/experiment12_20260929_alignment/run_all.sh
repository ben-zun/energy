#!/bin/bash
# Wave runner for the matched U-vs-A long-budget training.
# Usage: run_all.sh <alignment: none|element_reference> <seed> <gpu>
set -euo pipefail
ROOT=/home/yxzhang/energy/experiment12_20260929_alignment
PY=/home/yxzhang/envs/pet_forme/bin/python
route=$1; seed=$2; gpu=$3
mkdir -p "$ROOT/logs"
log="$ROOT/logs/${route}_${seed}.log"
echo "[$(date -Is)] start $route seed=$seed gpu=$gpu -> $log"
cd "$ROOT/code"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "$PY" train_alignment_matched.py \
    --alignment "$route" --seed "$seed" --device "cuda:$gpu" >> "$log" 2>&1
echo "[$(date -Is)] done $route seed=$seed exit=$?"
