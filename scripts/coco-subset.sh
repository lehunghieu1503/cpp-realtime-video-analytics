#!/usr/bin/env bash
# Download the COCO val2017 annotations and the first N validation images
# into data/coco/, for scripts/eval-map.py and INT8 calibration.
# Usage: scripts/coco-subset.sh [N]    (default: 500)
set -euo pipefail

N="${1:-500}"
DIR="$(dirname "$0")/../data/coco"
mkdir -p "$DIR/val2017"
cd "$DIR"

if [ ! -f annotations/instances_val2017.json ]; then
  curl -L --fail -o annotations.zip \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q -o annotations.zip annotations/instances_val2017.json
  rm annotations.zip
fi

# Image file names in the order of the annotation file, so every run picks
# the same subset
python3 - "$N" <<'PY' >images.txt
import json, sys
images = json.load(open("annotations/instances_val2017.json"))["images"]
for image in sorted(images, key=lambda i: i["id"])[: int(sys.argv[1])]:
    print(image["file_name"])
PY

# 8 downloads at a time; files already present are skipped
xargs -P 8 -I{} sh -c \
  '[ -f val2017/{} ] || curl -s --fail -o val2017/{} http://images.cocodataset.org/val2017/{}' \
  <images.txt

echo "data/coco/val2017: $(ls val2017 | wc -l) images"
