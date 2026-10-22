#!/bin/bash
# Remote orchestrator: waits for seed-20260528 to free GPU 0/1, launches the
# last seed, then runs evaluation + reports + verification once all six finish.
set -uo pipefail
ROOT=/home/yxzhang/energy/experiment12_20260929_alignment
PY=/home/yxzhang/envs/pet_forme/bin/python
cd "$ROOT"
mkdir -p logs results
LOG="$ROOT/logs/orchestrate.log"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

wait_run() {  # route seed
  while ! grep -q '"status": "complete"' "runs/$1_$2/status.json" 2>/dev/null; do sleep 60; done
  say "run $1 $2 complete"
}

say "orchestrator started"
wait_run U 20260528
wait_run A 20260528
say "GPU 0/1 free -> launching seed 20260530"
bash launch.sh none 20260530 0 >> "$LOG" 2>&1
bash launch.sh element_reference 20260530 1 >> "$LOG" 2>&1

for r in U A; do for s in 20260528 20260529 20260530; do wait_run "$r" "$s"; done; done
say "all six runs complete"

cd "$ROOT/code"
OMP_NUM_THREADS=4 "$PY" evaluate_alignment.py --device cuda:0 >> "$LOG" 2>&1 && say "evaluate ok" || say "evaluate FAILED"
OMP_NUM_THREADS=4 "$PY" build_reports.py --root "$ROOT" >> "$LOG" 2>&1 && say "reports ok" || say "reports FAILED"
OMP_NUM_THREADS=4 "$PY" verify_alignment.py >> "$LOG" 2>&1 && say "verify ok" || say "verify FAILED"
say "pipeline finished"
