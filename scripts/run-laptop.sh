#!/usr/bin/env bash
# Run the whole stack in Docker on this x86 machine: analytics on the Intel
# NPU, Mosquitto, Prometheus and Grafana.
#
# Usage: scripts/run-laptop.sh [up|down|logs]    (default: up)
# Needs models/yolo11s.onnx (scripts/download-model.sh yolo11s) and test
# frames under data/videos/. Edit deploy/config/laptop.yaml to change the
# sources or the backend.
set -euo pipefail

cd "$(dirname "$0")/.."
RENDER_GID=$(getent group render | cut -d: -f3)
export RENDER_GID
export VA_IMAGE=video-analytics:latest
compose() {
  docker compose --project-directory deploy -p video-analytics \
    -f deploy/docker-compose.yml -f deploy/docker-compose.laptop.yml "$@"
}

case "${1:-up}" in
  up)
    docker build -t "$VA_IMAGE" .
    # Created here, by this user: a directory Docker creates for a bind
    # mount would belong to root and the container could not write to it
    mkdir -p deploy/snapshots
    compose up -d --remove-orphans
    compose ps
    cat <<DONE

  Dashboard   http://localhost:9001
  Grafana     http://localhost:3000
  Prometheus  http://localhost:9090
  MQTT        mosquitto_sub -p 1884 -t 'video-analytics/#' -v
DONE
    ;;
  down) compose down ;;
  logs) compose logs -f app ;;
  *)
    echo "usage: $0 [up|down|logs]" >&2
    exit 1
    ;;
esac
