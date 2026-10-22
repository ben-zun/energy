#!/bin/bash
# Detached launcher: launch.sh <alignment: none|element_reference> <seed> <gpu>
set -euo pipefail
ROOT=/home/yxzhang/energy/experiment12_20260929_alignment
cd "$ROOT"
mkdir -p logs runs
setsid bash run_all.sh "$1" "$2" "$3" </dev/null >>"logs/wrapper_$1_$2.log" 2>&1 &
sleep 2
echo "launched alignment=$1 seed=$2 gpu=$3"
