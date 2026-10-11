#!/usr/bin/env python3
"""Compute COCO mAP for detections written by `video_analytics --eval-coco`.

  build/src/video_analytics --eval-coco data/coco/val2017 \
      --backend tensorrt --eval-out detections.json
  scripts/eval-map.py detections.json

Only the images present in the detections' directory are scored, so a subset
of val2017 works (see scripts/coco-subset.sh).
"""
import argparse
import json
import sys

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    sys.exit("Install dependencies: pip install pycocotools")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("detections", help="JSON from --eval-coco")
    parser.add_argument(
        "--annotations",
        default="data/coco/annotations/instances_val2017.json",
        help="COCO annotation file (default: %(default)s)",
    )
    parser.add_argument(
        "--images",
        default="data/coco/val2017",
        help="directory that was evaluated (default: %(default)s)",
    )
    parser.add_argument(
        "--csv", help="append 'label,mAP50-95,mAP50' to this file", default=None
    )
    parser.add_argument("--label", help="first CSV column", default="")
    args = parser.parse_args()

    import os

    coco = COCO(args.annotations)
    # The model numbers its 80 classes 0-79; COCO category ids have gaps
    category_ids = sorted(coco.getCatIds())
    by_file = {img["file_name"]: img["id"] for img in coco.imgs.values()}
    evaluated = [by_file[f] for f in os.listdir(args.images) if f in by_file]

    results = [
        {
            "image_id": by_file[d["file"]],
            "category_id": category_ids[d["class"]],
            "bbox": d["bbox"],
            "score": d["score"],
        }
        for d in json.load(open(args.detections))
        if d["file"] in by_file
    ]
    if not results:
        sys.exit("No detections match the annotation file")

    evaluation = COCOeval(coco, coco.loadRes(results), "bbox")
    evaluation.params.imgIds = evaluated
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()

    map_50_95, map_50 = evaluation.stats[0], evaluation.stats[1]
    print(f"\n{len(evaluated)} images: mAP50-95 = {map_50_95:.3f}, mAP50 = {map_50:.3f}")
    if args.csv:
        with open(args.csv, "a") as out:
            out.write(f"{args.label},{map_50_95:.4f},{map_50:.4f}\n")


if __name__ == "__main__":
    main()
