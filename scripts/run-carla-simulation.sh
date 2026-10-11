#!/usr/bin/env bash
# Run one CARLA scenario: CARLA and the controller on this PC, inference on
# the device.
#
# Usage: scripts/run-carla-simulation.sh --scenario NAME [options]
#   --scenario NAME    a scenario of deploy/config/carla-scenarios.yaml
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
#   --config FILE      default deploy/config/carla.yaml
#   --scenarios FILE   default deploy/config/carla-scenarios.yaml
#   --run-dir DIR      default .cache/carla/<run id>; must not exist
#   --local-backend B  backend of a local device (default: cpu)
# Environment:
#   CARLA_ROOT   CARLA 0.9.16 (default ~/opt/carla-0.9.16); its server is
#                started off-screen when nothing listens on port 2000, and
#                left running for the next run
#   CARLA_PYTHON python with the carla 0.9.16 client (default
#                $CARLA_ROOT/venv/bin/python, else python3)
#
# Every run gets a new run id and its own directory. The device is restarted
# with that id, so nothing of an earlier run can be joined with this one.
# On exit the script stops the processes it started and, on a remote device,
# the container it started; nothing else is touched.
set -euo pipefail

SCENARIO="" DEVICE=local REMOTE_DIR=video-analytics-carla SIM_HOST="" ARM=""
TICKS=0 CONFIG=deploy/config/carla.yaml SCENARIOS=deploy/config/carla-scenarios.yaml
RUN_DIR="" LOCAL_BACKEND=cpu
while [ $# -gt 0 ]; do
  case "$1" in
    --scenario) SCENARIO="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --sim-host) SIM_HOST="$2"; shift 2 ;;
    --arm) ARM=--arm; shift ;;
    --ticks) TICKS="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --scenarios) SCENARIOS="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --local-backend) LOCAL_BACKEND="$2"; shift 2 ;;
    *) sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
  esac
done
if [ -z "$SCENARIO" ]; then
  sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'
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
export PYTHONPATH="$PWD/ros2/video_analytics_simulation:$PWD/ros2/video_analytics_bridge:${PYTHONPATH:-}"

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

if ! (exec 3<>/dev/tcp/127.0.0.1/2000) 2>/dev/null; then
  [ -x "$CARLA_ROOT/CarlaUE4.sh" ] || { echo "No CARLA server on port 2000 and none at $CARLA_ROOT" >&2; exit 1; }
  echo "== Starting the CARLA server (left running after this script)"
  mkdir -p .cache/carla
  nohup "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -prefernvidia \
    -carla-rpc-port=2000 >.cache/carla/server.log 2>&1 &
  for _ in $(seq 60); do
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

PIDS=()
REMOTE_STARTED=0
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
# The device's copy of the config: the camera stream comes from this PC
sed "s/SIM_HOST/$SIM_HOST/" "$CONFIG" >"$RUN_DIR/device.yaml"

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
  --wait-client "$WAIT_CLIENT" $ARM 2>&1 | tee "$RUN_DIR/simulator.log" &
SIMULATOR=$!
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
  sed -i "s/^backend: .*/backend: $LOCAL_BACKEND/; s/^  backend: .*/  backend: $LOCAL_BACKEND/" "$RUN_DIR/device.yaml"
  VA_RUN_ID=$RUN_ID build/src/video_analytics --config "$RUN_DIR/device.yaml" \
    --headless --port 9011 >"$RUN_DIR/device.log" 2>&1 &
  PIDS+=($!)
else
  MODELS=$(sed -n 's/^ *model: *//p' "$CONFIG")
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

wait "$SIMULATOR"
echo "== Evaluating"
scripts/evaluate-carla.py "$RUN_DIR" || true
echo "Run directory: $RUN_DIR"
