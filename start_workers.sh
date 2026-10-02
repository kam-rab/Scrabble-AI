#!/bin/bash
source .venv/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p data/logs

for i in $(seq 0 11); do
    nohup python -u -m ai.training.self_play \
        --worker-id $i \
        --model-path data/model.pt \
        --gaddag-path data/gaddag.pkl \
        --output-dir data/transitions/ \
        --n-games 999999 \
        --opponent greedy \
        > data/worker_logs/worker_$i.log 2>&1 &
    echo "Started worker $i (PID $!)"
done
