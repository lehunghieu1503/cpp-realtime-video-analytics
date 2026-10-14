#!/bin/bash
# One show_park_bev run with MODEL as the bird's-eye-view stall model; the
# last line on stdout is the run's park metrics as JSON, for
# edgebench.py run --accuracy-cmd 'scripts/park-eval.sh {onnx} {label}'
#
#   scripts/park-eval.sh MODEL.onnx LABEL [run.sh options, e.g. --gui]
#
# Always exits 0 when the run produced metrics: a failed park is a result
# ("pass": 0), not an error. The run directory is .cache/carla/edgebench-*.
set -u
[ $# -ge 2 ] || { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
MODEL=$1 LABEL=$2
shift 2
cd "$(dirname "$0")/.."
[ -f "$MODEL" ] || { echo "no such model: $MODEL" >&2; exit 1; }

CONFIG=simulation/config/.edgebench-$LABEL.yaml
trap 'rm -f "$CONFIG"' EXIT
sed "s#^  model: models/stall-bev-town05.onnx#  model: $MODEL#" simulation/config/carla.yaml >"$CONFIG"
grep -q "^  model: $MODEL\$" "$CONFIG" || { echo "bev.model not found in carla.yaml" >&2; exit 1; }

RUN_DIR=.cache/carla/edgebench-$LABEL-$(date +%Y%m%dT%H%M%S)
# run.sh's own output goes to stderr: stdout is for the JSON line only
simulation/run.sh --scenario show_park_bev --arm --config "$CONFIG" --run-dir "$RUN_DIR" "$@" >&2
[ -f "$RUN_DIR/metrics.json" ] || { echo "no metrics in $RUN_DIR" >&2; exit 1; }

python3 - "$RUN_DIR" <<'EOF'
import json, sys
run_dir = sys.argv[1]
with open(run_dir + "/metrics.json") as f:
    m = json.load(f)
park = m["show"]["park"]
size = lambda v: None if v is None else abs(v)
print(json.dumps({
    "pass": int(m["verdict"] == "pass"),
    "final_u_m": size(park.get("final_u_m")),
    "final_v_m": size(park.get("final_v_m")),
    "heading_deg": size(park.get("final_heading_deg")),
    "parked_tick": park.get("parked_tick"),
    "stall_centre_err_m": (park.get("seen_stall_error") or {}).get("centre_m"),
    "tick_to_result_p50_ms": m["latency"]["tick_to_result_ms"]["p50"],
    "run_dir": run_dir}))
EOF
