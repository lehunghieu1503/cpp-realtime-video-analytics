#!/usr/bin/env bash
# Benchmark several backends on one video and print a comparison table:
# FPS, latency percentiles, time per stage, power, FPS per watt and mAP.
#
# Usage: scripts/benchmark.sh VIDEO BACKEND[@PRECISION][=MODEL]...
#   scripts/benchmark.sh clip.mp4 cpu openvino:CPU openvino:GPU \
#       openvino:NPU=models/yolo11s_int8.xml tensorrt@fp16 tensorrt@int8
# Environment:
#   VA_BIN   the binary, or a command that runs it (default:
#            build/src/video_analytics)
#   MODEL    default model (default: models/yolov8n.onnx)
#   FRAMES   frames to measure per run (default: 600)
#   OUT      output directory (default: data/bench)
#   COCO     image directory for mAP (default: data/coco/val2017); mAP is
#            skipped when it does not exist
# Plug the machine in and fix its power mode first: both change the numbers.
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi
VIDEO="$1"
shift
VA_BIN="${VA_BIN:-build/src/video_analytics}"
MODEL="${MODEL:-models/yolov8n.onnx}"
FRAMES="${FRAMES:-600}"
OUT="${OUT:-data/bench}"
COCO="${COCO:-data/coco/val2017}"

mkdir -p "$OUT"
rm -f "$OUT/speed.csv" "$OUT/map.csv"

for spec in "$@"; do
  model="$MODEL"
  [[ "$spec" == *=* ]] && model="${spec#*=}" && spec="${spec%%=*}"
  precision=fp16
  [[ "$spec" == *@* ]] && precision="${spec#*@}" && spec="${spec%%@*}"
  backend="$spec"
  # Only tensorrt builds for a precision; the others take it from the model
  [ "$backend" = tensorrt ] || precision=model
  label="$(basename "${model%.*}"),$backend,$precision"
  echo "== $label"

  # shellcheck disable=SC2086
  $VA_BIN --benchmark --input "$VIDEO" --frames "$FRAMES" --warmup 60 \
    --model "$model" --backend "$backend" --precision "${precision/model/fp16}" \
    --csv "$OUT/speed.csv" >"$OUT/run.log" 2>&1 ||
    { echo "   failed, see $OUT/run.log"; continue; }

  if [ -d "$COCO" ]; then
    # shellcheck disable=SC2086
    $VA_BIN --eval-coco "$COCO" --eval-out "$OUT/detections.json" \
      --model "$model" --backend "$backend" --precision "${precision/model/fp16}" \
      >>"$OUT/run.log" 2>&1 &&
      scripts/eval-map.py "$OUT/detections.json" --images "$COCO" \
        --csv "$OUT/map.csv" --label "$label" >>"$OUT/run.log" 2>&1 ||
      echo "   mAP failed, see $OUT/run.log"
  fi
done

# Join the two files on model,backend,precision into a Markdown table
python3 - "$OUT" <<'PY' | tee "$OUT/results.md"
import csv, os, sys
out = sys.argv[1]
maps = {}
if os.path.exists(f"{out}/map.csv"):
    for row in csv.reader(open(f"{out}/map.csv")):
        maps[tuple(row[:3])] = row[3:]
print("| Model | Backend | Precision | FPS | P50 ms | P95 ms | P99 ms | Infer ms | W | FPS/W | mAP50-95 | mAP50 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in csv.DictReader(open(f"{out}/speed.csv")):
    m = maps.get((r["model"], r["backend"], r["precision"]), ["n/a", "n/a"])
    print(f"| {r['model']} | {r['backend']} | {r['precision']} | {r['fps']} | {r['p50_ms']} "
          f"| {r['p95_ms']} | {r['p99_ms']} | {r['infer_ms']} | {r['watts'] or 'n/a'} "
          f"| {r['fps_per_watt'] or 'n/a'} | {m[0]} | {m[1]} |")
PY
