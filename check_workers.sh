#!/bin/bash
echo "=== Worker Health Check ==="
for i in $(seq 0 9); do
    log="data/worker_logs/worker_$i.log"
    if [ ! -f "$log" ]; then
        echo "Worker $i: NO LOG FILE"
        continue
    fi
    # Check if any game-10 print appears in last 50 lines
    recent=$(tail -n 50 "$log" | grep -c "games")
    if [ "$recent" -gt 0 ]; then
        last=$(tail -n 50 "$log" | grep "games" | tail -1)
        echo "Worker $i: OK — $last"
    else
        echo "Worker $i: STALLED (no game output in last 50 lines)"
    fi
done
