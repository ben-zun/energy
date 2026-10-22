#!/bin/bash
# Step 4 launch: cross-protocol r2SCAN adaptation, routes U and A, 3 seeds each.
# Waits for free VRAM so it coexists with the EqV3 symmetric runs.
set -u
ROOT=/home/yxzhang/energy/experiment14_alignment_controls_20260929
LOGS=$ROOT/logs
mkdir -p "$LOGS"
P=/home/yxzhang/envs/pet_forme/bin/python
NEED_MB=5000
EPOCHS=60
LR=3e-4
SEEDS="20260528 20260529 20260530"

pick_gpu() {
    local need=$1 best_gpu="" best_free=-1 used total free
    while :; do
        best_gpu=""; best_free=-1
        for g in 0 1 2 3; do
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
            total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g")
            free=$(( total - used ))
            if [ "$free" -ge "$need" ] && [ "$free" -gt "$best_free" ]; then
                best_free=$free; best_gpu=$g
            fi
        done
        if [ -n "$best_gpu" ]; then echo "$best_gpu"; return 0; fi
        sleep 60
    done
}

i=0
# the routed reference must exist before any run starts
while [ ! -f "$ROOT/r2scan10k_full/r2scan_reference_fit.npz" ]; do
    echo "waiting for r2scan_reference_fit.npz ..."; sleep 60
done
for seed in $SEEDS; do
    for route in U A; do
        name="run_${route}_${seed}"
        if [ -f "$ROOT/r2scan10k_full/runs/r2scan_${route}_${seed}_lr${LR}/status.json" ]; then
            echo "skip $name (already complete)"; continue
        fi
        gpu=$(pick_gpu $NEED_MB)
        echo "launch $name on cuda:$gpu at $(date +%H:%M:%S)"
        (
            cd "$ROOT" || exit 1
            CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 "$P" code/train_r2scan_full.py \
                --route "$route" --seed "$seed" --lr "$LR" --epochs "$EPOCHS" \
                --device cuda:0 > "$LOGS/r2scanfull_${name}.log" 2>&1
            echo "$? $name" >> "$LOGS/r2scanfull_exit_codes.txt"
        ) &
        i=$(( i + 1 ))
        sleep 90
    done
done
wait
echo "r2SCAN runs finished"
cat "$LOGS/r2scanfull_exit_codes.txt" 2>/dev/null
