#!/bin/bash
# Phase 3 bandwidth wall measurement.
# Runs bench_release --query with 1..8 threads and records QPS.
# Run from the repo root: bash measurements/sweep_threads.sh
# Repeat 3 times and average the QPS column for stable results.
#
# Thread affinity (prevents OS migration between runs):
#   OMP_PLACES=cores  — bind to physical cores
#   OMP_PROC_BIND=close — fill from core 0 outward

BINARY=./build/bench_release

if [ ! -f "$BINARY" ]; then
    echo "Error: $BINARY not found. Build first with: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
    exit 1
fi

echo "threads,QPS,Recall"
for T in 1 2 3 4 5 6 7 8; do
    output=$(OMP_PLACES=cores OMP_PROC_BIND=close $BINARY --query -t $T 2>/dev/null)
    qps=$(echo "$output"    | grep "^QPS:"    | awk '{print $2}')
    recall=$(echo "$output" | grep "^Recall"  | awk '{print $2}')
    echo "$T,$qps,$recall"
done
