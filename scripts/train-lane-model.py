#!/usr/bin/env python3
"""Train the lane model on the simulator's labelled images and export it to ONNX.

The network of Ultra Fast Lane Detection (Qin, Wang, Li, 2020) with a
ResNet-18: for each of 56 rows of the image and each of four lines, which
of 100 columns the line is in, or that it is not in that row. The input and
the output are those of scripts/export-lane-model.py, so the device runs
this file as it runs that one:

    input   float32 [1, 3, 288, 800]   RGB, scaled to [0, 1], then
                                       (x - mean) / std with the ImageNet values
    output  float32 [1, 101, 56, 4]    100 column bins + "no lane",
                                       56 row anchors, 4 lanes

It starts from torchvision's ImageNet weights of the ResNet-18, as the
authors started theirs; their TuSimple checkpoint is not used. Besides the
class of each cell it is taught the place the decoder reads from the cell
(the mean of the columns): that is what the lane is measured from.

Usage:
    simulation/lane-dataset.py --out data/lanes/town04      # needs CARLA
    scripts/train-lane-model.py --data data/lanes/town04 --output models/lane-town04.onnx
    # the same check on another model or other images, without training:
    scripts/train-lane-model.py --data data/lanes/town04 --check-only models/ufld-tusimple18.onnx

Next to the ONNX file it writes <output>.json: checksums, tool versions and
the check below. The check runs the val images through the ONNX file the
way the device does (JPEG at the quality of the runs, the whole image
shrunk to the input, the reference decoder) and compares with the labels:

    lines   each labelled side of the vehicle's own lane against the line
            the model gives nearest to it: in how many of the labelled rows
            it has a point, and how far off in pixels
    lanes   the lane in metres that simulation/video_analytics_simulation/lane.py
            makes of the model's lines against the one it makes of the
            labels: valid where the labels give one, and how far its centre
            and width are off

The images are of the simulator's maps: the numbers say how well the model
knows those roads, not roads.

Needs torch, torchvision, onnx, onnxruntime and opencv; they are not needed
to run the model.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "simulation"))
from video_analytics_simulation import lane  # noqa: E402
from video_analytics_simulation.calibration import Rig  # noqa: E402

INPUT_SHAPE = (1, 3, lane.MODEL_HEIGHT, lane.MODEL_WIDTH)
OUTPUT_SHAPE = (1, lane.GRID + 1, len(lane.ROW_ANCHORS), 4)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# The shared configuration sends the images at this quality
CHECK_JPEG_QUALITY = 30


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(directories, part):
    rows = []
    for directory in directories:
        with open(os.path.join(directory, part + ".jsonl")) as f:
            for line in f:
                row = json.loads(line)
                row["path"] = os.path.join(directory, row["image"])
                rows.append(row)
    return rows


def to_tensor(bgr):
    """What LaneEngine::preprocess does."""
    import cv2
    shrinks = bgr.shape[1] > lane.MODEL_WIDTH and bgr.shape[0] > lane.MODEL_HEIGHT
    resized = cv2.resize(bgr, (lane.MODEL_WIDTH, lane.MODEL_HEIGHT),
                         interpolation=cv2.INTER_AREA if shrinks else cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1))


def as_sent(bgr, quality):
    import cv2
    return cv2.imdecode(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1],
                        cv2.IMREAD_COLOR)


class Images:
    """The labelled images as training examples: mirrored half of the time,
    lighter or darker, and through JPEG at a quality a run may use."""

    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2
        row = self.rows[index]
        image = cv2.imread(row["path"])
        columns = np.array(row["lanes"], dtype=np.float32)  # (4, rows)
        if self.train:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                # Mirrored, left is right: the lines change places too
                image = image[:, ::-1]
                columns = np.where(columns[::-1] >= 0, row["width"] - 1 - columns[::-1], -1.0)
            image = np.clip(image.astype(np.float32) * rng.uniform(0.6, 1.4)
                            + rng.uniform(-25.0, 25.0), 0, 255).astype(np.uint8)
            image = as_sent(image, int(rng.integers(20, 91)))
        else:
            image = as_sent(image, CHECK_JPEG_QUALITY)
        seen = columns >= 0
        place = np.where(seen, lane.column_of(columns, row["width"]), 0.0)
        cell = np.where(seen, np.clip(np.rint(place) - 1, 0, lane.GRID - 1), lane.GRID)
        return (to_tensor(image), cell.T.astype(np.int64), place.T.astype(np.float32))


def network():
    import torch
    import torchvision

    class LaneNet(torch.nn.Module):
        """The authors' parsingNet without its auxiliary segmentation head."""

        def __init__(self):
            super().__init__()
            resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
            self.pool = torch.nn.Conv2d(512, 8, 1)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(1800, 2048), torch.nn.ReLU(),
                torch.nn.Linear(2048, int(np.prod(OUTPUT_SHAPE[1:]))))

        def forward(self, x):
            x = self.pool(self.backbone(x)).flatten(1)
            return self.head(x).view(-1, *OUTPUT_SHAPE[1:])

    return LaneNet()


def loss_of(logits, cell, place):
    """Which cell, and for a cell with a line the place the decoder reads:
    the mean of the columns under the softmax of the 100 bins."""
    import torch
    import torch.nn.functional as F
    logits = logits.float()
    which = F.cross_entropy(logits, cell)
    bins = torch.arange(1, lane.GRID + 1, device=logits.device).view(1, -1, 1, 1)
    read = (logits[:, :lane.GRID].softmax(dim=1) * bins).sum(dim=1)
    seen = cell < lane.GRID
    where = F.smooth_l1_loss(read[seen], place[seen]) if seen.any() else logits.sum() * 0.0
    return which + where, which.item(), where.item()


def train(args, work):
    import torch
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = {
        part: torch.utils.data.DataLoader(
            Images(read_rows(args.data, part), part == "train"), batch_size=args.batch,
            shuffle=part == "train", drop_last=part == "train", num_workers=args.workers,
            persistent_workers=True)
        for part in ("train", "val")}
    net = network().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(loader["train"])
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps,
                                                   pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    best, log = float("inf"), []
    for epoch in range(args.epochs):
        net.train()
        seen = np.zeros(3)
        for image, cell, place in loader["train"]:
            image, cell, place = image.to(device), cell.to(device), place.to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                logits = net(image)
            loss, which, where = loss_of(logits, cell, place)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            seen += (which, where, 1)
        net.eval()
        val = np.zeros(3)
        with torch.no_grad():
            for image, cell, place in loader["val"]:
                image, cell, place = image.to(device), cell.to(device), place.to(device)
                with torch.autocast(device, enabled=device == "cuda"):
                    logits = net(image)
                _, which, where = loss_of(logits, cell, place)
                val += (which, where, 1)
        row = {"epoch": epoch + 1, "train_cell": seen[0] / seen[2], "train_place": seen[1] / seen[2],
               "val_cell": val[0] / val[2], "val_place": val[1] / val[2]}
        log.append(row)
        print(json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)
        if row["val_cell"] + row["val_place"] < best:
            best = row["val_cell"] + row["val_place"]
            torch.save(net.state_dict(), os.path.join(work, "best.pt"))
    net.load_state_dict(torch.load(os.path.join(work, "best.pt"), map_location="cpu"))
    return net.cpu().eval(), log


def export(net, output):
    import onnx
    import onnxruntime
    import torch
    torch.onnx.export(net, torch.zeros(INPUT_SHAPE), output, input_names=["image"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(output))
    session = onnxruntime.InferenceSession(output, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal(INPUT_SHAPE).astype(np.float32)
    with torch.no_grad():
        expected = net(torch.from_numpy(x)).numpy()
    actual = session.run(None, {"image": x})[0]
    if actual.shape != OUTPUT_SHAPE:
        raise SystemExit("the exported model gives %s, not %s" % (actual.shape, OUTPUT_SHAPE))
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
    return float(np.abs(actual - expected).max())


def summary(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"count": 0}
    return {"count": int(len(values)), "median": round(float(np.median(values)), 3),
            "p90": round(float(np.percentile(values, 90)), 3),
            "p99": round(float(np.percentile(values, 99)), 3),
            "max": round(float(values.max()), 3)}


def check_onnx(onnx_path, directories, config):
    """The val images through the ONNX file as the device runs it, against
    their labels."""
    import cv2
    import onnxruntime
    rig = Rig.from_config(config["calibration"])
    lane_config = dict(lane.DEFAULTS)
    lane_config.update({k: v for k, v in (config.get("lane") or {}).items()
                        if k in lane_config})
    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    groups = {}

    def group(key):
        return groups.setdefault(key, {
            "images": 0, "rows_labelled": 0, "rows_found": 0, "rows_within_20_px": 0,
            "error_px": [], "label_gives_a_lane": 0, "model_gives_it": 0,
            "model_gives_one_the_label_does_not": 0, "centre_off_m": [], "width_off_m": []})

    def width_at(result):
        x = result["reference_x_m"]
        return (lane.evaluate(result["right"]["coefficients"], x)
                - lane.evaluate(result["left"]["coefficients"], x))

    for row in read_rows(directories, "val"):
        rows = lane.anchor_rows(row["height"])
        image = as_sent(cv2.imread(row["path"]), CHECK_JPEG_QUALITY)
        logits = session.run(None, {name: to_tensor(image)[None]})[0][0]
        found = lane.decode_ufld(logits, row["width"], row["height"])
        columns = [dict((y, x) for x, y in points) for points in found]
        labelled = [[(x, y) for x, y in zip(line, rows) if x >= 0] for line in row["lanes"]]
        errors, labelled_rows = [], 0
        for line in labelled[1:3]:  # the two sides of the vehicle's lane
            # The line the model gives that lies on most of this one,
            # whatever its place among the four
            best = []
            for candidate in columns:
                off = [abs(candidate[y] - x) for x, y in line if y in candidate]
                if sum(e <= 20 for e in off) > sum(e <= 20 for e in best):
                    best = off
            labelled_rows += len(line)
            errors += best
        truth = lane.estimate_lane(labelled, rig, lane_config)
        given = lane.estimate_lane(found, rig, lane_config)
        # Near the middle of a lane and along it, or anywhere across it
        keeping = abs(row["across_m"]) <= 0.8
        for g in (group("all"), group("light_" + row["light"]),
                  group("keeping_lane" if keeping else "across_lane")):
            g["images"] += 1
            g["rows_labelled"] += labelled_rows
            g["rows_found"] += len(errors)
            g["rows_within_20_px"] += int(sum(e <= 20 for e in errors))
            g["error_px"] += errors
            if truth["valid"]:
                g["label_gives_a_lane"] += 1
                if given["valid"]:
                    g["model_gives_it"] += 1
                    g["centre_off_m"].append(abs(given["lateral_error_m"]
                                                 - truth["lateral_error_m"]))
                    g["width_off_m"].append(abs(width_at(given) - width_at(truth)))
            elif given["valid"]:
                g["model_gives_one_the_label_does_not"] += 1
    for g in groups.values():
        for key in ("error_px", "centre_off_m", "width_off_m"):
            g[key] = summary(g[key])
    return {"jpeg_quality": CHECK_JPEG_QUALITY,
            "lane": {k: lane_config[k] for k in sorted(lane_config)},
            **{key: groups[key] for key in sorted(groups)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, nargs="+",
                        help="directories of simulation/lane-dataset.py")
    parser.add_argument("--output", help="the ONNX file to write")
    parser.add_argument("--check-only", metavar="ONNX",
                        help="no training: check this model on the val images of --data")
    parser.add_argument("--config", default=os.path.join(HERE, "..", "simulation", "config",
                                                         "carla.yaml"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--work", default=".cache/lane-training",
                        help="where the best weights of the run are kept")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.check_only:
        print(json.dumps(check_onnx(args.check_only, args.data, config), indent=2))
        return
    if not args.output:
        parser.error("--output or --check-only")

    import onnx
    import onnxruntime
    import torch
    import torchvision
    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    net, log = train(args, work)
    difference = export(net, args.output)
    check = check_onnx(args.output, args.data, config)
    report = {
        "model": "Ultra Fast Lane Detection network, ResNet-18, trained here from "
                 "torchvision's ImageNet weights",
        "data": [os.path.relpath(d) for d in args.data],
        "dataset": "simulation/lane-dataset.py; the simulator's maps",
        "training_images": len(read_rows(args.data, "train")),
        "validation_images": len(read_rows(args.data, "val")),
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "loss_by_epoch": log,
        "onnx_sha256": sha256(args.output),
        "input": {"name": "image", "shape": INPUT_SHAPE, "dtype": "float32"},
        "output": {"name": "logits", "shape": OUTPUT_SHAPE, "dtype": "float32"},
        "opset": 17,
        "pytorch_vs_onnxruntime_max_abs_difference": difference,
        "onnx_as_the_device_runs_it": check,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
    }
    with open(args.output + ".json", "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(check["all"], indent=2))
    print("wrote %s and %s.json" % (args.output, args.output))


if __name__ == "__main__":
    main()
