#!/bin/bash
# Launch the guidance-step-2 controls:
#   T  x 3 seeds x lr 3e-4            (trainable element reference)
#   U,A x 3 seeds x lr {1e-4,1e-3}    (small learning-rate selection budget)
# The lr=3e-4 U/A runs already exist in experiment12 and are reused.
set -u
ROOT=/home/yxzhang/energy/experiment14_alignment_controls_20260929
P=/home/yxzhang/envs/pet_forme/bin/python
LOGS="$ROOT/logs"
mkdir -p "$LOGS"

jobs=()
for seed in 20260528 20260529 20260530; do
    jobs+=("T:$seed:3e-4")
done
for lr in 1e-4 1e-3; do
    for seed in 20260528 20260529 20260530; do
        jobs+=("U:$seed:$lr")
        jobs+=("A:$seed:$lr")
    done
done

echo "total jobs: ${#jobs[@]}"
printf '%s\n' "${jobs[@]}" > "$LOGS/job_list.txt"

i=0
for job in "${jobs[@]}"; do
    IFS=: read -r route seed lr <<< "$job"
    gpu=$(( i % 4 ))
    name="run_${route}_${seed}_lr${lr}"
    echo "launch $name on cuda:$gpu"
    (
        cd "$ROOT" || exit 1
        CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 "$P" code/train_controls.py \
            --route "$route" --seed "$seed" --lr "$lr" --tag controls \
            --device cuda:0 > "$LOGS/${name}.log" 2>&1
        echo "$? $name" >> "$LOGS/exit_codes.txt"
    ) &
    i=$(( i + 1 ))
    sleep 2
done
wait
echo "all done"
sort "$LOGS/exit_codes.txt" 2>/dev/null
