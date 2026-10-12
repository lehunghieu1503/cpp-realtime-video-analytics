#!/usr/bin/env python3
"""Train the parking stall model and export it to ONNX.

An Ultralytics pose model, started from its COCO weights: two classes (a
stall that is empty, one with a vehicle in it) and four keypoints, the
stall's corners. Trained on the labelled images of simulation/stall-dataset.py
and exported with a fixed input the shape of the camera's image:

    input   float32 [1, 3, 544, 960]   RGB, scaled to [0, 1]
    output  float32 [1, 18, N]         cx, cy, w, h, 2 class scores,
                                       4 x (x, y, confidence), in input pixels

Usage:
    simulation/stall-dataset.py make --out data/stalls       # needs CARLA
    scripts/train-stall-model.py --data data/stalls --output models/stall-town05.onnx

Next to the ONNX file it writes <output>.json: checksums, tool versions, the
validation scores, and how far the corners it finds are from the labelled
ones. That last number is measured on the ONNX file the way the device runs
it: the image stretched to the input, the same threshold, the same NMS.

The images are of one car park, the one the runs use: the scores say how
well the model knows that car park, not car parks.

Needs ultralytics, torch and onnxruntime; they are not needed to run the
model. Ultralytics imports matplotlib: it must be one built for the numpy
that is installed (numpy 2 needs matplotlib 3.9 or later).
"""

import argparse
import glob
import hashlib
import json
import os
import shutil

import numpy as np

INPUT = (544, 960)  # height, width: 16:9, multiples of 32
FEATURES = 18


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(output, confidence, nms, width, height):
    """What StallEngine::decode does, for the check below: rows of
    (class, score, 4 x (x, y, confidence)) in pixels of the image."""
    import cv2
    rows = output[0].T  # (N, 18)
    scores = rows[:, 4:6].max(axis=1)
    rows, scores = rows[scores >= confidence], scores[scores >= confidence]
    sx, sy = width / INPUT[1], height / INPUT[0]
    boxes = [[int((r[0] - r[2] / 2) * sx), int((r[1] - r[3] / 2) * sy),
              max(1, int(r[2] * sx)), max(1, int(r[3] * sy))] for r in rows]
    kept = cv2.dnn.NMSBoxes(boxes, scores.tolist(), confidence, nms)
    stalls = []
    for i in np.asarray(kept).reshape(-1):
        corners = rows[i, 6:].reshape(4, 3) * [sx, sy, 1.0]
        stalls.append((int(rows[i, 5] > rows[i, 4]), float(scores[i]), corners))
    return stalls


def corner_errors(onnx_path, data, confidence=0.5, nms=0.5):
    """Pixels between each labelled, visible corner of the validation images
    and the same corner of the stall the model found there."""
    import cv2
    import onnxruntime
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    errors, labelled, found, class_right = [], 0, 0, 0
    for image_path in sorted(glob.glob(os.path.join(data, "images", "val", "*.jpg"))):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (INPUT[1], INPUT[0]),
                                     swapRB=True, crop=False)
        stalls = decode(session.run(None, {name: blob})[0], confidence, nms, width, height)
        label_path = image_path.replace("/images/", "/labels/")[:-4] + ".txt"
        with open(label_path) as f:
            labels = [[float(v) for v in line.split()] for line in f if line.strip()]
        for label in labels:
            truth = np.array(label[5:]).reshape(4, 3) * [width, height, 1.0]
            seen = truth[:, 2] > 0
            labelled += 1
            # The stall found for this label: the one whose corners are nearest
            best = None
            for cls, _, corners in stalls:
                distance = np.hypot(*(corners[seen, :2] - truth[seen, :2]).T)
                if best is None or distance.mean() < best[1].mean():
                    best = (cls, distance)
            if best is None or best[1].mean() > 40.0:
                continue
            found += 1
            class_right += int(best[0] == int(label[0]))
            errors += best[1].tolist()
    errors = np.array(errors)
    return {"labelled_stalls": labelled, "found": found,
            "class_right": class_right,
            "corner_error_px_median": float(np.median(errors)),
            "corner_error_px_p95": float(np.percentile(errors, 95)),
            "corner_error_px_max": float(errors.max()),
            "image_size": [width, height]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="directory of stall-dataset.py make")
    parser.add_argument("--output", required=True, help="the ONNX file to write")
    parser.add_argument("--weights", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--work", default=".cache/stall-training",
                        help="where the training run is kept")
    parser.add_argument("--skip-training", action="store_true",
                        help="export and check the weights of an earlier run in --work")
    args = parser.parse_args()

    import onnx
    import onnxruntime
    import torch
    import ultralytics
    from ultralytics import YOLO

    data = os.path.abspath(args.data)
    work = os.path.abspath(args.work)
    output = os.path.abspath(args.output)
    # The library downloads the starting weights, and others to check the
    # GPU with, into the working directory: not into the repository
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    best = os.path.join(work, "stall", "weights", "best.pt")
    if not args.skip_training:
        # Trained square, with the library's own augmentation; no rotation
        # or shear: the camera is rigid, the ground is flat
        YOLO(args.weights).train(
            data=os.path.join(data, "data.yaml"), imgsz=INPUT[1], epochs=args.epochs,
            batch=args.batch, project=work, name="stall", exist_ok=True,
            degrees=0.0, shear=0.0, perspective=0.0, seed=0, plots=False)
    model = YOLO(best)
    scores = model.val(data=os.path.join(data, "data.yaml"), imgsz=INPUT[1], plots=False,
                       project=work, name="val", exist_ok=True)
    exported = model.export(format="onnx", imgsz=list(INPUT), opset=17, simplify=True,
                            dynamic=False)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    shutil.copy(exported, output)

    session = onnxruntime.InferenceSession(output, providers=["CPUExecutionProvider"])
    shape = session.get_outputs()[0].shape
    if list(session.get_inputs()[0].shape) != [1, 3, INPUT[0], INPUT[1]] \
            or len(shape) != 3 or shape[1] != FEATURES:
        raise SystemExit("the exported model is not [1, 3, %d, %d] -> [1, %d, N]: %s" % (
            INPUT[0], INPUT[1], FEATURES, shape))
    check = corner_errors(output, data)
    with open(os.path.join(data, "data.yaml")) as f:
        dataset = f.read()
    report = {
        "model": "Ultralytics pose, from %s" % args.weights,
        "classes": ["empty", "occupied"],
        "corners": "mouth at +v, mouth at -v, back at -v, back at +v (parking.Stall.corners)",
        "training_images": len(glob.glob(os.path.join(data, "images", "train", "*.jpg"))),
        "validation_images": len(glob.glob(os.path.join(data, "images", "val", "*.jpg"))),
        "dataset": "simulation/stall-dataset.py make; one car park",
        "dataset_yaml_sha256": hashlib.sha256(dataset.encode()).hexdigest(),
        "epochs": args.epochs,
        "onnx_sha256": sha256(output),
        "input": {"shape": [1, 3, INPUT[0], INPUT[1]], "dtype": "float32"},
        "output": {"shape": [1, FEATURES, shape[2]], "dtype": "float32"},
        "opset": 17,
        "validation": {
            "box_map50_95": float(scores.box.map), "box_map50": float(scores.box.map50),
            "pose_map50_95": float(scores.pose.map), "pose_map50": float(scores.pose.map50)},
        "onnx_as_the_device_runs_it": check,
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
    }
    with open(output + ".json", "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report["validation"]))
    print(json.dumps(check))
    print("wrote %s and %s.json" % (output, output))


if __name__ == "__main__":
    main()
