#!/usr/bin/env bash
# Download YOLOv8 weights and export them to ONNX in models/.
# Usage: scripts/download-model.sh [MODEL]    (default: yolov8n)
set -euo pipefail

MODEL="${1:-yolov8n}"

if ! command -v yolo >/dev/null; then
  echo "Install dependencies: pip install ultralytics" >&2
  exit 1
fi

# Work inside models/ so the downloaded .pt lands in a gitignored directory
mkdir -p "$(dirname "$0")/../models"
cd "$(dirname "$0")/../models"

if [ -f "${MODEL}.onnx" ]; then
  echo "models/${MODEL}.onnx already exists"
  exit 0
fi

# opset 17: the default follows the installed PyTorch and will outgrow what
# ONNX Runtime 1.17.0 can load (opset 20)
yolo export model="${MODEL}.pt" format=onnx opset=17

echo "Saved models/${MODEL}.onnx"
