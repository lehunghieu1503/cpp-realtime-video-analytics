"""Lane geometry: the reference decoder of the lane model, lane boundaries in
metres, and moving a lane from the pose it was seen from to the current one.

The device decodes and fits the lane in C++ (src/inference/lane-engine.cpp,
src/analytics/lane-geometry.cpp). The functions here are the reference those
are tested against, through tests/fixtures/lane-calibration.json, and what
the controller uses to re-express a lane in the current ego frame.

A lane boundary is y(x) = a x^2 + b x + c in the ego frame: x forward, y
right, metres. Heading and curvature are positive to the right.
"""

import math

import numpy as np

# TuSimple row anchors of the model, in its 288-pixel-high input
ROW_ANCHORS = list(range(64, 288, 4))
GRID = 100
MODEL_WIDTH = 800
MODEL_HEIGHT = 288

DEFAULTS = {
    "roi_x_min_m": 6.0,     # nearer road is below the image with this rig
    "roi_x_max_m": 30.0,
    "reference_x_m": 8.0,   # where the errors are reported
    "min_points": 5,        # per boundary, inside the region of interest
    "min_support_m": 5.0,   # length both boundaries must share
    "max_residual_m": 0.25, # RMS distance of the points to the fitted curve
    "width_min_m": 2.5,
    "width_max_m": 4.6,
}


def decode_ufld(logits, width, height):
    """The authors' decoder (demo.py), from logits [101, 56, 4] to four lists
    of (x, y) pixels in an image of width x height.

    Kept as they wrote it: rows are reversed, the expected column is counted
    from 1, and a lane with fewer than 3 points is dropped.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (GRID + 1, len(ROW_ANCHORS), 4):
        raise ValueError("lane logits have shape %s" % (logits.shape,))
    if not np.isfinite(logits).all():
        raise ValueError("lane logits are not finite")
    out = logits[:, ::-1, :]
    bins = out[:GRID]
    e = np.exp(bins - bins.max(axis=0, keepdims=True))
    prob = e / e.sum(axis=0, keepdims=True)
    loc = (prob * (np.arange(GRID) + 1).reshape(-1, 1, 1)).sum(axis=0)
    loc[np.argmax(out, axis=0) == GRID] = 0
    step = (MODEL_WIDTH - 1) / (GRID - 1)
    rows = len(ROW_ANCHORS)
    lanes = []
    for lane in range(4):
        points = []
        if np.sum(loc[:, lane] != 0) > 2:
            for k in range(rows):
                if loc[k, lane] > 0:
                    x = int(loc[k, lane] * step * width / MODEL_WIDTH) - 1
                    y = int(height * (ROW_ANCHORS[rows - 1 - k] / MODEL_HEIGHT)) - 1
                    if 0 <= x < width and 0 <= y < height:
                        points.append((x, y))
        lanes.append(points)
    return lanes


def fit_boundary(xy, config=DEFAULTS):
    """Quadratic through ground points, or None when they do not support one."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    keep = (xy[:, 0] >= config["roi_x_min_m"]) & (xy[:, 0] <= config["roi_x_max_m"])
    xy = xy[keep]
    if len(xy) < config["min_points"] or np.ptp(xy[:, 0]) < config["min_support_m"]:
        return None

    def fit(points):
        A = np.stack([points[:, 0] ** 2, points[:, 0], np.ones(len(points))], axis=1)
        coefficients = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
        return coefficients, A @ coefficients - points[:, 1]

    coefficients, residual = fit(xy)
    # One pass of outlier rejection: a point of another marking, a kerb
    limit = max(0.15, 2.5 * float(np.median(np.abs(residual))))
    inliers = np.abs(residual) <= limit
    if not inliers.all() and inliers.sum() >= config["min_points"]:
        xy = xy[inliers]
        coefficients, residual = fit(xy)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    if (rms > config["max_residual_m"]
            or np.ptp(xy[:, 0]) < config["min_support_m"]):
        return None
    return {"coefficients": [float(c) for c in coefficients],
            "x_min_m": float(xy[:, 0].min()), "x_max_m": float(xy[:, 0].max()),
            "points": int(len(xy)), "rms_m": rms}


def evaluate(coefficients, x):
    a, b, c = coefficients
    return a * x * x + b * x + c


def _invalid(reason, config, calibration_id):
    return {"valid": False, "confidence": 0.0, "reason": reason,
            "calibration_id": calibration_id, "left": None, "right": None,
            "centerline": None, "reference_x_m": config["reference_x_m"],
            "lateral_error_m": None, "heading_rad": None, "curvature_inv_m": None}


def lane_from_boundaries(fits, config=DEFAULTS, calibration_id=""):
    """Chooses the boundaries either side of the ego vehicle among the fitted
    ones and derives the centreline and the errors a controller steers on."""
    left = right = None
    for fit in fits:
        if fit is None:
            continue
        x = min(max(config["reference_x_m"], fit["x_min_m"]), fit["x_max_m"])
        y = evaluate(fit["coefficients"], x)
        if y < 0 and (left is None or y > left[0]):
            left = (y, fit)
        if y >= 0 and (right is None or y < right[0]):
            right = (y, fit)
    if left is None or right is None:
        # One boundary is not a lane: no width is assumed to make a second
        return _invalid("missing_boundary", config, calibration_id)
    left, right = left[1], right[1]
    x_min = max(left["x_min_m"], right["x_min_m"])
    x_max = min(left["x_max_m"], right["x_max_m"])
    if x_max - x_min < config["min_support_m"]:
        return _invalid("insufficient_support", config, calibration_id)
    widths = [evaluate(right["coefficients"], x) - evaluate(left["coefficients"], x)
              for x in (x_min, (x_min + x_max) / 2, x_max)]
    if min(widths) <= 0:
        return _invalid("crossing", config, calibration_id)
    if min(widths) < config["width_min_m"] or max(widths) > config["width_max_m"]:
        return _invalid("width", config, calibration_id)

    a, b, c = [(l + r) / 2 for l, r in zip(left["coefficients"],
                                           right["coefficients"])]
    x = min(max(config["reference_x_m"], x_min), x_max)
    slope = 2 * a * x + b
    worst = max(left["rms_m"], right["rms_m"])
    confidence = (min(1.0, min(left["points"], right["points"]) / 15.0)
                  * (1.0 - worst / config["max_residual_m"]))

    def boundary(fit):
        return {k: fit[k] for k in ("coefficients", "x_min_m", "x_max_m")}

    return {"valid": True, "confidence": round(float(confidence), 3),
            "reason": "", "calibration_id": calibration_id,
            "left": boundary(left), "right": boundary(right),
            "centerline": {"coefficients": [a, b, c], "x_min_m": x_min,
                           "x_max_m": x_max},
            "reference_x_m": x,
            "lateral_error_m": evaluate((a, b, c), x),
            "heading_rad": math.atan(slope),
            "curvature_inv_m": 2 * a / (1 + slope * slope) ** 1.5}


def boundaries(lanes_px, rig, config=DEFAULTS):
    """Every marking of one image that supports a curve, as fitted on the
    road in the ego frame the image was captured from."""
    fits = []
    for points in lanes_px:
        if len(points) == 0:
            continue
        points = np.asarray(points, dtype=float)
        xy, valid = rig.uv_to_ground(points[:, 0], points[:, 1])
        fit = fit_boundary(xy[valid], config)
        if fit is not None:
            fits.append(fit)
    return fits


def estimate_lane(lanes_px, rig, config=DEFAULTS):
    """Lane pixels of one image to the lane envelope, in the ego frame the
    image was captured from."""
    return lane_from_boundaries(boundaries(lanes_px, rig, config), config,
                                rig.calibration_id)


def rebase_boundary(boundary, capture_pose, control_pose, ground_z=0.0,
                    config=DEFAULTS, step_m=1.0):
    """A boundary seen from capture_pose, fitted again in the ego frame at
    control_pose over the range it was observed on; None when what is left
    of it does not support a curve."""
    moved_config = dict(config)
    # The points come from a curve, not from pixels: region and count checks
    # were made at capture; what is left to check is the support that remains
    moved_config["roi_x_min_m"] = 0.0
    moved_config["min_points"] = 3
    moved_config["min_support_m"] = min(config["min_support_m"], 3.0)
    count = max(2, int(round((boundary["x_max_m"] - boundary["x_min_m"]) / step_m)) + 1)
    x = np.linspace(boundary["x_min_m"], boundary["x_max_m"], count)
    points = np.stack([x, evaluate(boundary["coefficients"], x),
                       np.full_like(x, ground_z)], axis=1)
    moved = control_pose.from_parent(capture_pose.to_parent(points))
    return fit_boundary(moved[:, :2], moved_config), moved_config


def rebase_lane(lane, capture_pose, control_pose, ground_z=0.0, config=DEFAULTS,
                step_m=1.0):
    """A lane seen from capture_pose, expressed in the ego frame at
    control_pose: the vehicle has moved since the image was taken.

    The boundaries are sampled over the range they were observed on, moved
    through the world frame and fitted again, so the result has its own,
    shorter support. Nothing is extended beyond what was seen.
    """
    if not lane or not lane.get("valid"):
        return _invalid("no_lane", config, (lane or {}).get("calibration_id", ""))
    fits = []
    for side in ("left", "right"):
        fit, moved_config = rebase_boundary(lane[side], capture_pose, control_pose,
                                            ground_z, config, step_m)
        fits.append(fit)
    if fits[0] is None or fits[1] is None:
        return _invalid("insufficient_support", config, lane.get("calibration_id", ""))
    return lane_from_boundaries(fits, moved_config, lane.get("calibration_id", ""))
