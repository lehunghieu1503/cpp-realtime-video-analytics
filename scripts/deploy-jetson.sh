#!/usr/bin/env bash
# Build everything on this PC and ship it to a Jetson, which only runs it.
#
# Usage: scripts/deploy-jetson.sh USER@HOST [REMOTE_DIR]
#   REMOTE_DIR  directory on the board (default: video-analytics, relative to
#               the home directory)
# Environment:
#   VIDEO_DIR   directory on the board mounted at /data/videos (default:
#               REMOTE_DIR/videos)
#   SKIP_BUILD  set to 1 to ship the image already built
#   IMAGE_ONLY  set to 1 to build and ship the application image and stop:
#               nothing in REMOTE_DIR is changed and nothing is restarted
#               (simulation/run.sh starts its own container)
#   SIM         set to 1 to run simulated RTSP cameras on the board and read
#               them with hardware decoding (docker-compose.sim.yml)
#   SECURE      set to 1 for HTTPS and passwords on the dashboard, TLS and
#               passwords on MQTT, and a Grafana login. Certificates and
#               passwords are created in deploy/secrets/ on this PC.
#
# What it does:
#   1. builds the arm64 image with docker buildx (QEMU)
#   2. pulls the arm64 Mosquitto, Prometheus and Grafana images
#   3. sends each image over SSH, leaving out the layers the board has
#   4. copies deploy/ (compose files, config) and the model
#   5. runs `docker compose up -d` on the board
# The board needs Docker with the compose plugin and the NVIDIA runtime; it
# needs no internet access and nothing is built or installed there.
set -euo pipefail

if [ $# -lt 1 ]; then
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
HOST="$1"
REMOTE_DIR="${2:-video-analytics}"
VIDEO_DIR="${VIDEO_DIR:-}"

cd "$(dirname "$0")/.."
APP_IMAGE=video-analytics:jetson
PLATFORM=linux/arm64
# Third-party images, as named in deploy/docker-compose.yml
COMPOSE_SOURCES=(deploy/docker-compose.yml)
[ "${SIM:-0}" = 1 ] && COMPOSE_SOURCES+=(deploy/docker-compose.sim.yml)
OTHER_IMAGES=$(sed -n 's/^ *image: *\([a-z][^$ ]*\)$/\1/p' "${COMPOSE_SOURCES[@]}")

# The model named in the config must exist here: the board cannot export it
MODEL=$(sed -n 's/^model: *//p' deploy/config/config.yaml)
if [ ! -f "$MODEL" ]; then
  echo "Missing $MODEL. Export it first: scripts/download-model.sh $(basename "$MODEL" .onnx)" >&2
  exit 1
fi

echo "== Checking the board"
ssh "$HOST" 'docker compose version >/dev/null && docker info --format "Docker {{.ServerVersion}}, runtimes: {{range \$k, \$v := .Runtimes}}{{\$k}} {{end}}"'

if [ "${SKIP_BUILD:-0}" != 1 ]; then
  echo "== Building $APP_IMAGE for $PLATFORM"
  docker buildx build --platform "$PLATFORM" -f Dockerfile.jetson \
    -t "$APP_IMAGE" --load .
fi

[ "${IMAGE_ONLY:-0}" = 1 ] && OTHER_IMAGES=""
for image in $OTHER_IMAGES; do
  # Pulled for the board's platform, not this PC's
  if ! docker image inspect "$image" --format '{{.Architecture}}' 2>/dev/null |
    grep -q arm64; then
    echo "== Pulling $image for $PLATFORM"
    docker pull --quiet --platform "$PLATFORM" "$image"
  fi
done

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Send one image unless the board already has exactly that image
ship_image() {
  local image="$1" archive="$TMP/image.tar"
  docker save --platform "$PLATFORM" -o "$archive" "$image"
  local id
  id=$(scripts/image-delta.py id "$archive")
  if [ "$(ssh "$HOST" "docker image inspect --format '{{.Id}}' '$image' 2>/dev/null" || true)" = "$id" ]; then
    echo "== $image: already on the board"
  else
    echo "== $image: sending the layers the board lacks"
    # Every layer of every image on the board; unquoted on purpose
    local have
    have=$(ssh "$HOST" 'docker image ls -q | sort -u | xargs -r docker image inspect --format "{{range .RootFS.Layers}}{{println .}}{{end}}" | sort -u')
    # shellcheck disable=SC2086
    scripts/image-delta.py strip "$archive" $have | ssh "$HOST" 'docker load'
  fi
  rm -f "$archive"
}

ship_image "$APP_IMAGE"
if [ "${IMAGE_ONLY:-0}" = 1 ]; then
  echo "== $APP_IMAGE is on the board; nothing else was changed"
  exit 0
fi
for image in $OTHER_IMAGES; do ship_image "$image"; done

echo "== Copying configuration and model to $HOST:$REMOTE_DIR"
# Created here, by the login user: a directory Docker creates for a bind
# mount would belong to root and the container could not write to it
ssh "$HOST" "mkdir -p '$REMOTE_DIR/models' '$REMOTE_DIR/snapshots' '$REMOTE_DIR/videos'"
rsync -rtz --delete deploy/ "$HOST:$REMOTE_DIR/" \
  --exclude models --exclude snapshots --exclude videos --exclude .env \
  --exclude secrets --exclude prometheus-data

COMPOSE_FILE=docker-compose.yml:docker-compose.jetson.yml
[ "${SIM:-0}" = 1 ] && COMPOSE_FILE=$COMPOSE_FILE:docker-compose.sim.yml
if [ "${SECURE:-0}" = 1 ]; then
  echo "== Creating and copying secrets"
  scripts/make-secrets.sh "${HOST#*@}"
  # Only secrets/run: the CA key stays on this PC
  ssh "$HOST" "mkdir -p '$REMOTE_DIR/prometheus-data' && mkdir -p -m 700 '$REMOTE_DIR/secrets'"
  rsync -rtz --delete --chmod=F600 deploy/secrets/run/ "$HOST:$REMOTE_DIR/secrets/"
  COMPOSE_FILE=$COMPOSE_FILE:docker-compose.secure.yml
fi
rsync -tz --progress "$MODEL" "$HOST:$REMOTE_DIR/models/"
# An INT8 calibration table, if one was generated for this model
CALIBRATION="${MODEL%.onnx}.calibration"
if [ -f "$CALIBRATION" ]; then
  ssh "$HOST" "mkdir -p '$REMOTE_DIR/models/trt-cache'"
  rsync -tz "$CALIBRATION" "$CALIBRATION".cache "$HOST:$REMOTE_DIR/models/trt-cache/"
fi

# Read by docker compose on the board, so `docker compose ...` there needs
# no flags
ssh "$HOST" "cat > '$REMOTE_DIR/.env'" <<ENV
COMPOSE_FILE=$COMPOSE_FILE
VA_IMAGE=$APP_IMAGE
DEPLOY_UID_GID=$(ssh "$HOST" 'echo $(id -u):$(id -g)')
${VIDEO_DIR:+VIDEO_DIR=$VIDEO_DIR}
ENV

echo "== Starting"
ssh "$HOST" "cd '$REMOTE_DIR' && docker compose up -d --remove-orphans && docker compose ps"

if [ "${SECURE:-0}" = 1 ]; then
  # Grafana reads its admin password from the environment only when it
  # creates its database; on a board deployed before, set it explicitly
  ssh "$HOST" "cd '$REMOTE_DIR' && for i in \$(seq 30); do docker compose exec -T grafana grafana cli admin reset-admin-password \"\$(cat secrets/grafana.password)\" >/dev/null 2>&1 && break; sleep 2; done"
fi

BOARD="${HOST#*@}"
cat <<DONE

Deployed. The first start with a new model builds its TensorRT engine, which
takes several minutes; follow it with:
  ssh $HOST 'cd $REMOTE_DIR && docker compose logs -f app'

DONE
if [ "${SECURE:-0}" = 1 ]; then
  R=deploy/secrets/run
  cat <<DONE
  Dashboard   https://$BOARD:9001   user admin, password in $R/http.password
              (the browser will not know the CA: import $R/ca.crt, or accept the warning)
  Grafana     http://$BOARD:3000    user admin, password in $R/grafana.password
  Prometheus  only on the board, at 127.0.0.1:9090
  MQTT        mosquitto_sub -h $BOARD -p 8883 --cafile $R/ca.crt \\
                -u va -P "\$(cat $R/mqtt.password)" -t 'video-analytics/#' -v
DONE
else
  cat <<DONE
  Dashboard   http://$BOARD:9001
  Grafana     http://$BOARD:3000
  Prometheus  http://$BOARD:9090
  MQTT        mosquitto_sub -h $BOARD -t 'video-analytics/#' -v

Nothing above asks for a password. SECURE=1 adds TLS and logins.
DONE
fi
