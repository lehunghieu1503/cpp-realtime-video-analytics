#!/usr/bin/env bash
# Summarise a soak test from the Prometheus of a running deployment: memory
# growth, lowest frame rate and reconnects over the last HOURS hours.
# Usage: scripts/soak-report.sh HOST [HOURS]    (default: 24)
set -euo pipefail

HOST="${1:?usage: scripts/soak-report.sh HOST [HOURS]}"
HOURS="${2:-24}"

query() {
  curl -s --fail --get "http://$HOST:9090/api/v1/query" --data-urlencode "query=$1" |
    python3 -c '
import json, sys
skip = ("__name__", "instance", "job")
for r in json.load(sys.stdin)["data"]["result"]:
    labels = ",".join(k + "=" + v for k, v in r["metric"].items() if k not in skip)
    print("  %-28s %.2f" % (labels or "-", float(r["value"][1])))'
}

W="${HOURS}h"
echo "Hours of data in the window:"
query "count_over_time(va_process_resident_memory_bytes[$W]) * 5 / 3600"
echo "Resident memory, MB (first / last / max):"
query "(va_process_resident_memory_bytes offset $W) / 1e6"
query "va_process_resident_memory_bytes / 1e6"
query "max_over_time(va_process_resident_memory_bytes[$W]) / 1e6"
echo "Memory growth, MB per hour (linear fit; near 0 means no leak):"
query "deriv(va_process_resident_memory_bytes[$W]) * 3600 / 1e6"
echo "FPS per stream (min / avg):"
query "min_over_time(va_stream_fps[$W])"
query "avg_over_time(va_stream_fps[$W])"
echo "Reconnects per stream:"
query "increase(va_stream_reconnects_total[$W])"
echo "Process restarts (resets of the processed-frames counter):"
query "resets(va_stream_frames_processed_total[$W])"
echo "Power, W (avg):"
query "avg_over_time(va_power_watts[$W])"
