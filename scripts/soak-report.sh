#!/usr/bin/env bash
# Summarise a soak test from the Prometheus of a running deployment: memory
# growth, lowest frame rate and reconnects over the last HOURS hours.
# Prometheus is asked from inside its container over SSH, so this works
# whether or not its port is open to the network.
# Usage: scripts/soak-report.sh USER@HOST [REMOTE_DIR] [HOURS]
#   REMOTE_DIR  deployment directory on the board (default: video-analytics)
#   HOURS       window to summarise (default: 24)
set -euo pipefail

HOST="${1:?usage: scripts/soak-report.sh USER@HOST [REMOTE_DIR] [HOURS]}"
REMOTE_DIR="${2:-video-analytics}"
HOURS="${3:-24}"

query() {
  local encoded
  encoded=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$1")
  ssh "$HOST" "cd '$REMOTE_DIR' && docker compose exec -T prometheus wget -qO- 'http://localhost:9090/api/v1/query?query=$encoded'" |
    python3 -c '
import json, sys
skip = ("__name__", "instance", "job")
for r in json.load(sys.stdin)["data"]["result"]:
    labels = ",".join(k + "=" + v for k, v in r["metric"].items() if k not in skip)
    print("  %-28s %.2f" % (labels or "-", float(r["value"][1])))'
}

# Rates and minima across a restart would mix two processes, so the window
# is cut to the current run when that is shorter
UPTIME=$(query "va_process_uptime_seconds" | awk '{print int($2)}')
WINDOW=$((HOURS * 3600))
if [ -n "$UPTIME" ] && [ "$UPTIME" -lt "$WINDOW" ]; then WINDOW=$UPTIME; fi
# The first minute is start-up: sources connecting, buffers filling
[ "$WINDOW" -gt 120 ] && WINDOW=$((WINDOW - 60))
W="${WINDOW}s"
printf 'Current run: up %.1f h; window summarised: %.1f h\n' \
  "$(echo "${UPTIME:-0} / 3600" | bc -l)" "$(echo "$WINDOW / 3600" | bc -l)"

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
echo "Process restarts in the last $HOURS h:"
query "resets(va_process_uptime_seconds[${HOURS}h])"
echo "GPU load, % (avg / max):"
query "avg_over_time(va_gpu_utilization_percent[$W])"
query "max_over_time(va_gpu_utilization_percent[$W])"
echo "Power, W (avg):"
query "avg_over_time(va_power_watts[$W])"
