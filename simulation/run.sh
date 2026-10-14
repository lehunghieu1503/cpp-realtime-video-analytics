#!/usr/bin/env bash
# Run one CARLA scenario: CARLA and the controller on this PC, inference on
# the device.
#
# Usage: simulation/run.sh --scenario NAME [options]
#   --scenario NAME    a scenario of simulation/config/carla-scenarios.yaml
#   --device TARGET    where inference runs:
#                        USER@HOST  a Jetson that has the video-analytics:jetson
#                                   image (scripts/deploy-jetson.sh ships it)
#                        local      build/src/video_analytics on this PC
#                        none       no device: scenarios that measure the
#                                   vehicle or the sensors only
#                      (default: local)
#   --remote-dir DIR   directory on the device (default: video-analytics-carla,
#                      relative to the home directory); everything this script
#                      puts on the device is inside it
#   --sim-host ADDR    address of this PC as the device reaches it
#                      (default: the address used to reach the device)
#   --arm              arm the controller once its gates pass. Without it the
#                      vehicle stays braked for the whole run.
#   --ticks N          run N ticks instead of the scenario's number
#   --gui              one window with the whole run: the vehicle seen from
#                      behind, its own cameras and the view from above, with
#                      what the device finds in them: boxes, lane lines,
#                      parking stalls. The CARLA server itself stays off-screen.
#   --view             that window without the image from behind the vehicle
#   --web              serve the device's dashboard too: a page with each
#                      camera's video and what was found in it, at the
#                      address the script prints. Off by default: then the
#                      device draws and encodes nothing for a browser.
#   --config FILE      default simulation/config/carla.yaml
#   --scenarios FILE   default simulation/config/carla-scenarios.yaml
#   --run-dir DIR      default .cache/carla/<run id>; must not exist
#   --local-backend B  backend of a local device instead of the config's
#                      (cpu is too slow to keep up with 30 frames a second)
#   --fp16             every model of the config as its FP16 file,
#                      models/<name>_fp16.onnx, made with
#                      scripts/quantize-fp16.py. TensorRT 11 takes the
#                      precision from the file: the config's `precision: fp16`
#                      alone leaves the models FP32 there.
# Environment:
#   CARLA_ROOT   CARLA 0.9.16 (default ~/opt/carla-0.9.16); its server is
#                started off-screen when nothing listens on port 2000, and
#                left running for the next run. One that listens but does not
#                answer is stopped and started again.
#   CARLA_PYTHON python with the carla 0.9.16 client (default
#                $CARLA_ROOT/venv/bin/python, else python3)
#
# Every run gets a new run id and its own directory. The device is restarted
# with that id, so nothing of an earlier run can be joined with this one.
# Exit code: that of simulation/evaluate.py for the run (0 pass, 1 fail,
# 2 a file is missing), or 1 when the run could not be made.
# On exit the script stops the processes it started and, on a remote device,
# the container it started; nothing else is touched.
set -euo pipefail

SCENARIO="" DEVICE=local REMOTE_DIR=video-analytics-carla SIM_HOST="" ARM=""
TICKS=0 CONFIG=simulation/config/carla.yaml SCENARIOS=simulation/config/carla-scenarios.yaml
RUN_DIR="" LOCAL_BACKEND="" GUI="" WEB="" FP16=""
while [ $# -gt 0 ]; do
  case "$1" in
    --scenario) SCENARIO="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --sim-host) SIM_HOST="$2"; shift 2 ;;
    --arm) ARM=--arm; shift ;;
    --gui) GUI="--follow --view"; shift ;;
    --view) GUI="--view"; shift ;;
    --web) WEB=1; shift ;;
    --ticks) TICKS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --local-backend) LOCAL_BACKEND="$2"; shift 2 ;;
    --fp16) FP16=1; shift ;;
    *) sed -n '2,54p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
  esac
done
if [ -z "$SCENARIO" ]; then
  sed -n '2,54p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

cd "$(dirname "$0")/.."
CARLA_ROOT="${CARLA_ROOT:-$HOME/opt/carla-0.9.16}"
if [ -z "${CARLA_PYTHON:-}" ]; then
  CARLA_PYTHON=python3
  [ -x "$CARLA_ROOT/venv/bin/python" ] && CARLA_PYTHON="$CARLA_ROOT/venv/bin/python"
fi
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
[ -f "$ROS_SETUP" ] || { echo "ROS 2 not found at $ROS_SETUP (set ROS_SETUP)" >&2; exit 1; }
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u
# The run's topics stay on this PC
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}"
export PYTHONPATH="$PWD/simulation:$PWD/ros2/video_analytics_bridge:${PYTHONPATH:-}"

echo "== Checking this PC"
"$CARLA_PYTHON" - <<'PY'
import carla, rclpy, cv2, yaml  # noqa: F401
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
missing = [e for e in ("appsrc", "gdppay", "tcpserversink") if Gst.ElementFactory.find(e) is None]
if missing:
    raise SystemExit("GStreamer elements missing: " + ", ".join(missing))
print("carla client %s, GStreamer %s" % (carla.Client("127.0.0.1", 1).get_client_version(),
                                        Gst.version_string()))
PY

# A server that listens but does not answer is stopped: one left running for
# hours was found like that, and the simulator then waited on it for a minute
# before it gave up
if (exec 3<>/dev/tcp/127.0.0.1/2000) 2>/dev/null && ! "$CARLA_PYTHON" -c '
import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
client.get_server_version()' 2>/dev/null; then
  echo "== The CARLA server on port 2000 does not answer: stopping it"
  pkill -f 'CarlaUE4-Linux-Shipping' || true
  for _ in $(seq 20); do
    (exec 3<>/dev/tcp/127.0.0.1/2000) 2>/dev/null || break
    sleep 1
  done
  if (exec 3<>/dev/tcp/127.0.0.1/2000) 2>/dev/null; then
    pkill -KILL -f 'CarlaUE4-Linux-Shipping' || true
    sleep 2
  fi
fi
if ! (exec 3<>/dev/tcp/127.0.0.1/2000) 2>/dev/null; then
  [ -x "$CARLA_ROOT/CarlaUE4.sh" ] || { echo "No CARLA server on port 2000 and none at $CARLA_ROOT" >&2; exit 1; }
  echo "== Starting the CARLA server (left running after this script)"
  mkdir -p .cache/carla
  # shellcheck disable=SC2086
  nohup "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -prefernvidia \
    -carla-rpc-port=2000 >.cache/carla/server.log 2>&1 &
  # The laptop took 90 s to open the port
  for _ in $(seq 180); do
    (exec 3<>/dev/tcp/127.0.0.1/2000) 2>/dev/null && break
    sleep 1
  done
  sleep 5
fi

RUN_ID=$("$CARLA_PYTHON" -c 'from video_analytics_simulation import protocol; print(protocol.new_run_id())')
RUN_DIR="${RUN_DIR:-.cache/carla/$RUN_ID}"
[ -e "$RUN_DIR" ] && { echo "$RUN_DIR exists; a run directory is never reused" >&2; exit 1; }
mkdir -p "$RUN_DIR"
echo "== Run $RUN_ID in $RUN_DIR"
if [ -n "$FP16" ]; then
  # The config the whole run uses, kept with the run
  sed -E 's#^( *model: .*)\.onnx$#\1_fp16.onnx#' "$CONFIG" >"$RUN_DIR/config.yaml"
  # The simulator reads the map's stalls from beside the config
  cp "$(dirname "$CONFIG")/carla-stalls.yaml" "$RUN_DIR/"
  CONFIG=$RUN_DIR/config.yaml
fi

PIDS=()
REMOTE_STARTED=0
LOCAL_DEVICE=""
DEVICE_STARTS=0
start_local_device() {
  VA_RUN_ID=$RUN_ID build/src/video_analytics --config "$RUN_DIR/device.yaml" \
    --headless --port 9011 >>"$RUN_DIR/device.log" 2>&1 &
  LOCAL_DEVICE=$!
  PIDS+=("$LOCAL_DEVICE")
  DEVICE_STARTED=$SECONDS
  DEVICE_STARTS=$((DEVICE_STARTS + 1))
}
cleanup() {
  for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
  if [ "$REMOTE_STARTED" = 1 ]; then
    # shellcheck disable=SC2029
    ssh "$DEVICE" "cd '$REMOTE_DIR' && export VA_RUN_ID=$RUN_ID && docker compose logs --no-color app > app.log 2>&1; docker compose stop -t 5" >/dev/null 2>&1 || true
    scp -q "$DEVICE:$REMOTE_DIR/app.log" "$RUN_DIR/device.log" 2>/dev/null || true
  fi
}
trap cleanup EXIT

WAIT_CLIENT=600
if [ "$DEVICE" = none ]; then
  SIM_HOST=127.0.0.1 DEVICE_URL=http://127.0.0.1:9 WAIT_CLIENT=0
elif [ "$DEVICE" = local ]; then
  SIM_HOST="${SIM_HOST:-127.0.0.1}"
  DEVICE_URL=http://127.0.0.1:9011
else
  BOARD="${DEVICE#*@}"
  # This PC's address on the route to the device
  SIM_HOST="${SIM_HOST:-$(ip -4 route get "$(getent ahostsv4 "$BOARD" | awk 'NR==1{print $1}')" | sed -n 's/.* src \([0-9.]*\).*/\1/p')}"
  DEVICE_URL=http://$BOARD:9002
fi
# The device's copy of the config: the camera streams come from this PC. It
# has what this scenario uses and nothing else. The rear camera and the stall
# model are for the scenarios that park; the lane model is for the others (a
# car park has no lanes). A scenario that parks with `bev: true` also has the
# cameras of the bird's-eye view, and the stall model is then that section's,
# run on the view instead of the rear camera's image. The dashboard is served
# only with --web.
"$CARLA_PYTHON" - "$CONFIG" "$SCENARIOS" "$SCENARIO" "$SIM_HOST" "$WEB" >"$RUN_DIR/device.yaml" <<'PY'
import sys
import yaml
config_path, scenarios_path, name, host, web = sys.argv[1:]
with open(config_path) as f:
    config = yaml.safe_load(f.read().replace("SIM_HOST", host))
with open(scenarios_path) as f:
    scenarios = yaml.safe_load(f)
if name not in scenarios["scenarios"]:
    raise SystemExit("unknown scenario " + name)
scenario = dict(scenarios.get("defaults", {}), **scenarios["scenarios"][name])
view = [camera["source"] for camera in (config.get("bev") or {}).get("cameras", [])]
if scenario.get("park") and scenario.get("bev"):
    if not view:
        raise SystemExit("scenario %s wants a bird's-eye view: the config has no bev section" % name)
    for source in config["sources"]:
        source.pop("lane", None)
        source.pop("stall", None)
    config.pop("lane", None)
    config["stall"]["model"] = config["bev"].pop("model")
elif scenario.get("park"):
    config["sources"] = [s for s in config["sources"]
                         if s.get("stall") or s.get("lane") or s["id"] not in view]
    for source in config["sources"]:
        source.pop("lane", None)
    config.pop("lane", None)
    config.pop("bev", None)
else:
    config["sources"] = [s for s in config["sources"]
                         if s.get("lane") or (not s.get("stall") and s["id"] not in view)]
    config.pop("stall", None)
    config.pop("bev", None)
config["dashboard"] = bool(web)
yaml.safe_dump(config, sys.stdout, sort_keys=False)
PY

# Only the models this scenario loads have to exist
if [ -n "$FP16" ]; then
  for model in $(sed -n 's/^ *model: *//p' "$RUN_DIR/device.yaml"); do
    [ -f "$model" ] || { echo "Missing $model; make it with: scripts/quantize-fp16.py ${model%_fp16.onnx}.onnx -o $model" >&2; exit 1; }
  done
fi

echo "== Starting the controller and the bridge"
"$CARLA_PYTHON" -m video_analytics_simulation.controller --config "$CONFIG" \
  --run-dir "$RUN_DIR" --run-id "$RUN_ID" >"$RUN_DIR/controller.log" 2>&1 &
CONTROLLER=$!
PIDS+=("$CONTROLLER")
"$CARLA_PYTHON" -m video_analytics_bridge.bridge --ros-args -p "url:=$DEVICE_URL" \
  >"$RUN_DIR/bridge.log" 2>&1 &
PIDS+=($!)

echo "== Starting the simulator"
"$CARLA_PYTHON" -m video_analytics_simulation.simulator --config "$CONFIG" \
  --scenarios "$SCENARIOS" --scenario "$SCENARIO" --run-dir "$RUN_DIR" \
  --run-id "$RUN_ID" --ticks "$TICKS" --controller-pid "$CONTROLLER" \
  --wait-client "$WAIT_CLIENT" $ARM $GUI > >(tee "$RUN_DIR/simulator.log") 2>&1 &
SIMULATOR=$!
PIDS+=("$SIMULATOR")
# The simulator listens once it has written the manifest
for _ in $(seq 120); do
  [ -f "$RUN_DIR/manifest.json" ] && break
  kill -0 "$SIMULATOR" 2>/dev/null || { echo "The simulator stopped" >&2; exit 1; }
  sleep 0.5
done

[ "$DEVICE" = none ] || echo "== Starting the device ($DEVICE), camera from $SIM_HOST"
if [ "$DEVICE" = none ]; then
  :
elif [ "$DEVICE" = local ]; then
  [ -z "$LOCAL_BACKEND" ] || sed -i "s/^backend: .*/backend: $LOCAL_BACKEND/; s/^  backend: .*/  backend: $LOCAL_BACKEND/" "$RUN_DIR/device.yaml"
  start_local_device
  [ -z "$WEB" ] || echo "   dashboard: http://127.0.0.1:9011"
  [ -n "$WEB$GUI" ] || echo "   nothing to watch: --gui opens the camera window, --web serves the dashboard"
else
  MODELS=$(sed -n 's/^ *model: *//p' "$RUN_DIR/device.yaml")
  for model in $MODELS; do
    [ -f "$model" ] || { echo "Missing $model" >&2; exit 1; }
  done
  # shellcheck disable=SC2029
  ssh "$DEVICE" "mkdir -p '$REMOTE_DIR/config' '$REMOTE_DIR/models'"
  # shellcheck disable=SC2086
  rsync -tz $MODELS "$DEVICE:$REMOTE_DIR/models/"
  rsync -tz deploy/docker-compose.carla.yml "$DEVICE:$REMOTE_DIR/docker-compose.yml"
  rsync -tz "$RUN_DIR/device.yaml" "$DEVICE:$REMOTE_DIR/config/carla.yaml"
  # shellcheck disable=SC2029
  ssh "$DEVICE" "cd '$REMOTE_DIR' && VA_RUN_ID=$RUN_ID docker compose up -d --force-recreate"
  REMOTE_STARTED=1
  echo "   (the first run with a model builds its TensorRT engine: several minutes)"
fi

# A local device that stops ends the run: the simulator would wait ten
# minutes for it. One that stops while it starts is started again, up to
# eleven times: beside a CARLA server that has just loaded a map there is not
# enough video memory for its models at first. With a window and two cameras
# it took five tries, a minute; five was all it had.
while kill -0 "$SIMULATOR" 2>/dev/null; do
  if [ -n "$LOCAL_DEVICE" ] && ! kill -0 "$LOCAL_DEVICE" 2>/dev/null; then
    if [ $((SECONDS - DEVICE_STARTED)) -lt 30 ] && [ "$DEVICE_STARTS" -lt 12 ]; then
      echo "   the device stopped while starting; again in 8 s ($(tail -n 1 "$RUN_DIR/device.log" | cut -c1-80))"
      sleep 8
      start_local_device
    else
      echo "The device stopped:" >&2
      tail -n 3 "$RUN_DIR/device.log" >&2
      exit 1
    fi
  fi
  sleep 1
done
wait "$SIMULATOR" || echo "The simulator stopped with an error" >&2
echo "== Evaluating"
# The script's exit code is the evaluation's: 0 every expectation holds, 1
# one fails, 2 a file of the run is missing
CODE=0
simulation/evaluate.py "$RUN_DIR" || CODE=$?
echo "Run directory: $RUN_DIR"
exit "$CODE"
