#!/usr/bin/env python3
"""Train the parking stall model and export it to ONNX.

An Ultralytics pose model, started from its COCO weights, with four classes
and four keypoints:

    0 empty, 1 occupied   a stall, with its four corners
    2 mouth, 3 back       where a divider between two stalls ends: at the
                          mouth of the stalls, or at the line across their
                          back. Its four keypoints are all that one place.

Trained on the labelled images of simulation/stall-dataset.py and exported
with a fixed input the shape of the camera's image:

    input   float32 [1, 3, 544, 960]   RGB, scaled to [0, 1]
    output  float32 [1, 20, N]         cx, cy, w, h, 4 class scores,
                                       4 x (x, y, confidence), in input pixels

Usage:
    simulation/stall-dataset.py make --out data/stalls       # needs CARLA
    scripts/train-stall-model.py --data data/stalls --output models/stall-town05.onnx

Next to the ONNX file it writes <output>.json: checksums, tool versions, the
validation scores, and two numbers measured on the ONNX file the way the
device runs it (the image stretched to the input, the same threshold, the
same NMS): how many of the labelled stalls it finds with the right class,
and how many pixels the divider ends it finds are from the labelled ones.

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
import math
import os
import shutil

import numpy as np

INPUT = (544, 960)  # height, width: 16:9, multiples of 32
FEATURES = 20
NAMES = ["empty", "occupied", "mouth", "back"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(output, confidence, nms, width, height):
    """What StallEngine::decode does, for the check below: the stalls as
    (class, box x, y, w, h) and the ends as (class, x, y), in pixels of the
    image. NMS runs among the stalls and among the ends apart."""
    import cv2
    rows = output[0].T  # (N, 20)
    classes = rows[:, 4:8].argmax(axis=1)
    scores = rows[:, 4:8].max(axis=1)
    sx, sy = width / INPUT[1], height / INPUT[0]
    found = [[], []]
    for group, members in enumerate(((0, 1), (2, 3))):
        keep = (scores >= confidence) & np.isin(classes, members)
        part, cls, score = rows[keep], classes[keep], scores[keep]
        boxes = [[int((r[0] - r[2] / 2) * sx), int((r[1] - r[3] / 2) * sy),
                  max(1, int(r[2] * sx)), max(1, int(r[3] * sy))] for r in part]
        for i in np.asarray(cv2.dnn.NMSBoxes(boxes, score.tolist(), confidence,
                                             nms)).reshape(-1):
            if group == 0:
                found[0].append((int(cls[i]), boxes[i]))
            else:
                points = part[i, 8:].reshape(4, 3)
                found[1].append((int(cls[i]), float(points[:, 0].mean() * sx),
                                 float(points[:, 1].mean() * sy)))
    return found


def overlap(a, b):
    """Intersection over union of two boxes x, y, w, h."""
    x = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return x * y / (a[2] * a[3] + b[2] * b[3] - x * y)


def check_onnx(onnx_path, data, confidence=0.5, nms=0.5):
    """The validation images through the ONNX file as the device runs it,
    against their labels."""
    import cv2
    import onnxruntime
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    stalls = {"labelled": 0, "found": 0, "class_right": 0}
    # Ends by the width of their box: 0.8 m of ground is 40 pixels wide at
    # about 7 m from the camera
    ends = {"near": [], "far": []}
    labelled = {"near": 0, "far": 0}
    for image_path in sorted(glob.glob(os.path.join(data, "images", "val", "*.jpg"))):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (INPUT[1], INPUT[0]),
                                     swapRB=True, crop=False)
        found_stalls, found_ends = decode(session.run(None, {name: blob})[0], confidence,
                                          nms, width, height)
        with open(image_path.replace("/images/", "/labels/")[:-4] + ".txt") as f:
            labels = [[float(v) for v in line.split()] for line in f if line.strip()]
        for label in labels:
            cls = int(label[0])
            box = [(label[1] - label[3] / 2) * width, (label[2] - label[4] / 2) * height,
                   label[3] * width, label[4] * height]
            if cls < 2:
                stalls["labelled"] += 1
                best = max(found_stalls, key=lambda s: overlap(box, s[1]), default=None)
                if best is not None and overlap(box, best[1]) >= 0.5:
                    stalls["found"] += 1
                    stalls["class_right"] += int(best[0] == cls)
            else:
                size = "near" if box[2] >= 40 else "far"
                labelled[size] += 1
                x, y = label[5] * width, label[6] * height
                apart = [math.hypot(px - x, py - y) for c, px, py in found_ends if c == cls]
                # Farther than this it is not this end that was found
                if apart and min(apart) <= 0.5 * box[2]:
                    ends[size].append(min(apart))
    report = {"image_size": [width, height], "stalls": stalls}
    for size, errors in ends.items():
        errors = np.array(errors) if errors else np.array([math.nan])
        report["ends_" + size] = {
            "labelled": labelled[size], "found": int(np.isfinite(errors).sum()),
            "error_px_median": float(np.median(errors)),
            "error_px_p95": float(np.percentile(errors, 95)),
            "error_px_max": float(errors.max())}
    return report


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
    parser.add_argument("--resume", action="store_true",
                        help="go on with a training run in --work that was stopped")
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
    if args.resume:
        YOLO(os.path.join(work, "stall", "weights", "last.pt")).train(resume=True)
    elif not args.skip_training:
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
    check = check_onnx(output, data)
    with open(os.path.join(data, "data.yaml")) as f:
        dataset = f.read()
    report = {
        "model": "Ultralytics pose, from %s" % args.weights,
        "classes": NAMES,
        "keypoints": "of a stall: mouth at +v, mouth at -v, back at -v, back at +v "
                     "(parking.Stall.corners); of an end: that place, four times",
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
