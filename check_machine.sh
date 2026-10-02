#!/bin/bash
echo "=== $(hostname) ==="
echo "CPUs:     $(nproc)"
echo "Load:     $(uptime | awk -F'load average:' '{print $2}')"
echo "RAM free: $(free -h | awk '/^Mem:/ {print $7 " free of " $2}')"
echo "Swap:     $(free -h | awk '/^Swap:/ {print $3 " used of " $2}')"
echo "My jobs:  $(pgrep -u $USER | wc -l) processes"
echo "All load: $(ps aux | awk 'NR>1 {sum += $3} END {printf "%.1f%% CPU\n", sum}')"
