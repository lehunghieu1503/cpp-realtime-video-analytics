"""The numbers the C++ lane code and this package must agree on.

tests/fixtures/lane-calibration.json is written from the Python reference:

    python3 simulation/test/fixtures.py

test_calibration.py checks the file still matches the Python code, and
tests/test-lane.cpp checks the C++ code against the same file.
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from video_analytics_simulation import lane  # noqa: E402
from video_analytics_simulation.calibration import Rig  # noqa: E402

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "tests",
                    "fixtures", "lane-calibration.json")

CALIBRATION = {"width": 1280, "height": 720, "fov_deg": 52.0,
               "camera": {"x": 1.5, "y": 0.0, "z": 1.6, "pitch_deg": 3.0},
               "ground_z": 0.0}


def synthetic_logits(case):
    """Logits with a peak per lane and row; the C++ test builds the same.

    case["lanes"]: per lane [column at row 0, columns per row, sharpness,
    first row without a lane, last row without a lane].
    """
    logits = np.zeros((lane.GRID + 1, len(lane.ROW_ANCHORS), 4), dtype=np.float32)
    for l, (c0, slope, sharp, none_from, none_to) in enumerate(case["lanes"]):
        for r in range(len(lane.ROW_ANCHORS)):
            centre = c0 + slope * r
            for g in range(lane.GRID):
                logits[g, r, l] = np.float32(-sharp * (g - centre) * (g - centre))
            logits[lane.GRID, r, l] = np.float32(
                5.0 if none_from <= r <= none_to else -50.0)
    return logits


DECODER_CASES = [
    # Two lanes converging, one at the left edge, one absent everywhere
    {"width": 1280, "height": 720,
     "lanes": [[20.0, 0.45, 0.5, 0, 5], [80.0, -0.45, 0.5, 0, 5],
               [0.2, 0.0, 2.0, 50, 55], [50.0, 0.0, 0.5, 0, 55]]},
    # Right edge, a flat distribution (a tie between bins), two points only
    {"width": 640, "height": 360,
     "lanes": [[99.0, 0.0, 2.0, 0, 0], [50.0, 0.0, 0.0, 10, 40],
               [30.0, 0.5, 0.3, 0, 53], [60.0, -0.2, 1.0, 20, 30]]},
]


def lane_pixels(rig, offset_m, curvature=0.0, heading=0.0, xs=None):
    """Pixels of a road marking y(x) = curvature x^2 + heading x + offset."""
    xs = np.arange(7.0, 40.0, 1.5) if xs is None else xs
    points = np.stack([xs, curvature * xs ** 2 + heading * xs + offset_m,
                       np.full_like(xs, rig.ground_z)], axis=1)
    uv, _ = rig.project(points)
    return [[float(round(u)), float(round(v))] for u, v in uv
            if 0 <= round(u) < rig.width and 0 <= round(v) < rig.height]


def build():
    rig = Rig.from_config(CALIBRATION)
    pixels = [[640, 700], [200, 650], [1100, 500], [640, 400], [640, 330],
              [640, 300], [0, 719], [1279, 719], [-5, 600], [640.5, 450.25]]
    xy, valid = rig.uv_to_ground([p[0] for p in pixels], [p[1] for p in pixels])
    ground = [{"uv": p, "xy": [float(x), float(y)] if ok else None}
              for p, (x, y), ok in zip(pixels, xy, valid)]

    decoder = []
    for case in DECODER_CASES:
        points = lane.decode_ufld(synthetic_logits(case), case["width"], case["height"])
        decoder.append(dict(case, points=[[list(p) for p in l] for l in points]))

    lanes = []
    for name, markings in [
        ("straight, centred", [(-1.75, 0, 0), (1.75, 0, 0)]),
        ("ego 0.4 m left of centre, four markings",
         [(-4.85, 0, 0), (-1.35, 0, 0), (2.15, 0, 0), (5.65, 0, 0)]),
        ("curve to the right with a heading", [(-1.75, 0.002, 0.03), (1.75, 0.002, 0.03)]),
        ("curve to the left", [(-1.6, -0.003, 0.0), (1.9, -0.003, 0.0)]),
        ("one marking only", [(-1.75, 0, 0)]),
        ("too narrow", [(-0.6, 0, 0), (0.6, 0, 0)]),
        ("crossing", [(-1.5, 0, 0.1), (1.75, 0, -0.1)]),
        ("no markings", []),
    ]:
        px = [lane_pixels(rig, *m) for m in markings]
        lanes.append({"name": name, "pixels": px,
                      "expected": lane.estimate_lane(px, rig)})
    # Short: the far half of the points is missing
    px = [lane_pixels(rig, o, xs=np.arange(7.0, 11.0, 1.0)) for o in (-1.75, 1.75)]
    lanes.append({"name": "too short", "pixels": px,
                  "expected": lane.estimate_lane(px, rig)})

    return {"comment": "Written by simulation/test/fixtures.py",
            "calibration": CALIBRATION, "calibration_id": rig.calibration_id,
            "focal_px": rig.focal, "lane_config": lane.DEFAULTS,
            "ground": ground, "decoder": decoder, "lanes": lanes}


if __name__ == "__main__":
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w") as f:
        json.dump(build(), f, indent=1)
    print("wrote", os.path.normpath(PATH))
