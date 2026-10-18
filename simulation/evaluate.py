#!/usr/bin/env python3
"""Evaluate one CARLA run from the files in its run directory.

Usage: simulation/evaluate.py RUN_DIR [--replay] [--lane-model MODEL.onnx]

Reads what the simulator and the controller wrote (see
simulation/video_analytics_simulation/simulator.py) and
writes RUN_DIR/metrics.json and RUN_DIR/summary.md. Exit code 0 when every
expectation of the scenario holds, 1 when one fails, 2 when a file the
evaluation needs is missing.

This is the only place where the true state of the other actors and the
map's lane centre (truth.jsonl) are read.

How things are counted:

  frames      every image the camera produced is in the ledger, whatever
              became of it. Rates are over those, not over what arrived.
  recall      two numbers per distance bin. "received": of the ground-truth
              objects in frames whose result arrived. "end to end": of the
              ground-truth objects in every frame the camera produced; a
              frame without a result counts as a miss.
  matching    a prediction and a ground-truth object of the same class with
              IoU >= 0.5, one to one; a second prediction on the same object
              is a false positive.
  ground truth box: the actor's 3-D bounding box projected with the run's
              calibration and clipped to the image. An object is eligible
              when any of it is inside the image and it is in front of the
              camera. Scenes are built so that objects do not hide each
              other; occlusion is not tested.
  distance    from the camera to the centre of the actor's bounding box, in
              the road plane. Bins: 0-30, 30-60, 60-90, 90-120 m; an object
              at exactly 30 m is in 30-60.
  latency     differences of this PC's monotonic clock only: from the tick
              that produced a frame to the arrival of its result, and to the
              tick at which a command resting on it was applied. The
              device's clock is never subtracted from this PC's.

--lane-model MODEL.onnx runs the lane model on this PC's CPU on the JPEGs kept
from the run and compares the lane pixels with those the device reported.

--replay recomputes the metric tracking from the stored artifacts alone
(results as received, lossless depth, ego states, calibration) and compares
it with what the live run computed. A missing or corrupt artifact is
reported and fails the replay; nothing is taken from the simulator.
"""

import argparse
import json
import math
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_analytics_simulation import protocol  # noqa: E402
from video_analytics_simulation.calibration import Pose, Rig, decode_depth  # noqa: E402
from video_analytics_simulation.controller import stopping_distance  # noqa: E402
from video_analytics_simulation.metric_tracker import MetricTracker  # noqa: E402
from video_analytics_simulation.parking import Stall  # noqa: E402

BINS = [(0, 30), (30, 60), (60, 90), (90, 120)]
IOU_MATCH = 0.5


class MissingArtifact(Exception):
    pass


def read_jsonl(path, required=True):
    if not os.path.exists(path):
        if required:
            raise MissingArtifact(path)
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile_summary(values):
    if len(values) == 0:
        return {"count": 0}
    values = np.asarray(values, dtype=float)
    return {"count": int(len(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max()),
            "mean": float(values.mean())}


def error_summary(errors):
    if len(errors) == 0:
        return {"count": 0}
    errors = np.asarray(errors, dtype=float)
    return {"count": int(len(errors)), "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mean": float(errors.mean()),
            "p95_abs": float(np.percentile(np.abs(errors), 95)),
            "max_abs": float(np.abs(errors).max())}


def bin_of(distance):
    for low, high in BINS:
        if low <= distance < high or (high == BINS[-1][1] and distance == high):
            return "%d-%d" % (low, high)
    return None


def iou(a, b):
    """Boxes as (x0, y0, x1, y1)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(truths, predictions):
    """One-to-one matching of ground truth and predictions of one frame.

    truths: [{"class", "box"}], predictions: [{"class", "box"}]. Returns
    (pairs, unmatched prediction indices): pairs is [(truth index,
    prediction index)], best IoU first, IoU >= 0.5, same class.
    """
    candidates = []
    for t, truth in enumerate(truths):
        for p, prediction in enumerate(predictions):
            if truth["class"] != prediction["class"]:
                continue
            overlap = iou(truth["box"], prediction["box"])
            if overlap >= IOU_MATCH:
                candidates.append((overlap, t, p))
    pairs, used_t, used_p = [], set(), set()
    for overlap, t, p in sorted(candidates, reverse=True):
        if t in used_t or p in used_p:
            continue
        used_t.add(t)
        used_p.add(p)
        pairs.append((t, p))
    return pairs, [p for p in range(len(predictions)) if p not in used_p]


# ── Ground truth geometry ────────────────────────────────────────────────────

def actor_corners_ego(actor, ego_pose):
    """The 8 corners of an actor's bounding box in the ego frame."""
    ex, ey, ez = actor["extent"]
    cx, cy, cz = actor["box_centre"]
    corners = np.array([[cx + sx * ex, cy + sy * ey, cz + sz * ez]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return ego_pose.from_parent(Pose.from_dict(actor["pose"]).to_parent(corners))


def truth_object(actor, ego_pose, rig):
    """Ground truth of one actor in one frame, or None when it is not there
    or no part of it is in the image."""
    if not actor.get("present"):
        return None
    corners = actor_corners_ego(actor, ego_pose)
    uv, depth = rig.project(corners)
    if depth.min() <= 0.1:
        return None  # beside or behind the camera
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    box = (max(0.0, x0), max(0.0, y0), min(float(rig.width), x1),
           min(float(rig.height), y1))
    if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
        return None
    centre = corners.mean(axis=0)
    full = (x1 - x0) * (y1 - y0)
    return {"id": actor["id"], "name": actor["name"], "class": actor["class"],
            "box": box,
            "distance_m": float(math.hypot(centre[0] - rig.x, centre[1] - rig.y)),
            "near_x_m": float(corners[:, 0].min()),
            "left_m": float(corners[:, 1].min()), "right_m": float(corners[:, 1].max()),
            "centre_ego": [float(centre[0]), float(centre[1])],
            "tiny": bool(box[3] - box[1] < 10.0),
            "truncated": bool((box[2] - box[0]) * (box[3] - box[1]) < 0.9 * full)}


def truth_kinematics(truth_rows):
    """World velocity and acceleration of every actor at every frame, by
    central differences of its scripted positions."""
    series = {}
    for row in truth_rows:
        for actor in row["actors"]:
            series.setdefault(actor["id"], []).append(
                (row["sim_ns"], actor["pose"]["x"], actor["pose"]["y"],
                 actor.get("present", False)))
    out = {}
    for actor_id, rows in series.items():
        rows.sort()
        t = np.array([r[0] for r in rows]) * 1e-9
        xy = np.array([[r[1], r[2]] for r in rows])
        present = np.array([r[3] for r in rows])
        velocity = np.full_like(xy, np.nan)
        accel = np.full_like(xy, np.nan)
        for i in range(1, len(rows) - 1):
            # A teleport (a sweep placement, an actor appearing) is not motion
            if not (present[i - 1] and present[i] and present[i + 1]):
                continue
            step = np.linalg.norm(xy[i + 1] - xy[i - 1])
            if step > 5.0:
                continue
            velocity[i] = (xy[i + 1] - xy[i - 1]) / (t[i + 1] - t[i - 1])
        for i in range(1, len(rows) - 1):
            if np.isfinite(velocity[i - 1]).all() and np.isfinite(velocity[i + 1]).all():
                accel[i] = (velocity[i + 1] - velocity[i - 1]) / (t[i + 1] - t[i - 1])
        for i, row in enumerate(rows):
            out[(actor_id, row[0])] = (velocity[i], accel[i])
    return out


def lane_truth_in_ego(lane, ego_pose, reference_x):
    """The map's lane centre at reference_x in the ego frame: (y, heading).
    None when the centre line does not reach that far."""
    points = np.array([[p[0], p[1], ego_pose.z] for p in lane["centre_ahead"]])
    ego = ego_pose.from_parent(points)
    x, y = ego[:, 0], ego[:, 1]
    if len(x) < 2 or reference_x < x.min() or reference_x > x.max():
        return None
    order = np.argsort(x)
    x, y = x[order], y[order]
    centre = float(np.interp(reference_x, x, y))
    i = int(np.clip(np.searchsorted(x, reference_x), 1, len(x) - 1))
    heading = math.atan2(y[i] - y[i - 1], x[i] - x[i - 1])
    return centre, heading


def ego_lateral_deviation(row):
    """Signed distance of the ego origin from the map's lane centre, metres,
    positive to the right of the centre line."""
    yaw = math.radians(row["lane"]["centre_yaw_deg"])
    dx = row["ego"]["pose"]["x"] - row["lane"]["centre"][0]
    dy = row["ego"]["pose"]["y"] - row["lane"]["centre"][1]
    return -math.sin(yaw) * dx + math.cos(yaw) * dy


# ── Sections of the report ───────────────────────────────────────────────────

def frames_section(frames, ticks):
    run_ticks = {f for f, t in ticks.items() if t.get("tick", -1) >= 0}
    # Images of the ticks before the run (the vehicle settling) are not its
    # frames, even when one of them arrives after the run began
    generated = {f: e for f, e in frames.items() if e["registered"] and f in run_ticks}
    statuses = {}
    for entry in generated.values():
        key = entry["status"] if entry["status"] != "dropped" else \
            "dropped:" + entry.get("reason", "?")
        statuses[key] = statuses.get(key, 0) + 1
    before = {e["sim_ns"] for f, e in frames.items()
              if e["registered"] and f not in run_ticks and e["status"] == "sent"}
    return generated, before, {
        "ticks": len(run_ticks),
        "rgb_generated": len(generated),
        # The tick happened and the camera gave no image: a loss of its own
        "rgb_never_generated": len(run_ticks - set(generated)),
        "status": statuses,
        "sent": statuses.get("sent", 0),
        "jpeg_bytes": percentile_summary([e["jpeg_bytes"] for e in generated.values()
                                          if e["status"] == "sent"]),
    }


def identity_section(perception, generated, run_id, before_run=()):
    """Every result received must name a frame this run sent, once.
    before_run: timestamps of the frames sent while waiting for the device,
    before the first tick of the run; their results are set aside."""
    by_ns = {e["sim_ns"]: f for f, e in generated.items() if e["status"] == "sent"}
    seen = set()
    counts = {"received": len(perception), "matched": 0, "before_run": 0,
              "wrong_run": 0,
              "unknown_frame": 0, "repeated": 0, "out_of_order": 0,
              "bad_identity": 0}
    usable = {}
    last = -1
    for record in perception:
        simulation = record["envelope"].get("simulation") or {}
        sim_ns = simulation.get("sim_ns")
        if simulation.get("run_id") != run_id:
            counts["wrong_run"] += 1
            continue
        if not isinstance(sim_ns, int) or isinstance(sim_ns, bool):
            counts["bad_identity"] += 1
            continue
        if sim_ns in before_run:
            counts["before_run"] += 1
            continue
        if sim_ns not in by_ns:
            # A timestamp that is not one the simulator sent: the join key
            # was altered on the way. This must stay at zero.
            counts["unknown_frame"] += 1
            continue
        if sim_ns in seen:
            counts["repeated"] += 1
            continue
        if sim_ns < last:
            counts["out_of_order"] += 1
        last = max(last, sim_ns)
        seen.add(sim_ns)
        counts["matched"] += 1
        usable[sim_ns] = record
    counts["delivery_rate"] = (counts["matched"] / len(generated)) if generated else 0.0
    counts["rejected_by_controller"] = {}
    for record in perception:
        if record["rejected"]:
            counts["rejected_by_controller"][record["rejected"]] = \
                counts["rejected_by_controller"].get(record["rejected"], 0) + 1
    return usable, counts


def detection_section(truth_by_ns, usable, generated, rig):
    bins = {"%d-%d" % b: {"eligible_received": 0, "tp": 0, "eligible_generated": 0,
                          "tiny": 0, "truncated": 0} for b in BINS}
    false_positives = 0
    predictions_total = 0
    matches = {}  # sim_ns -> [(truth object, detection)]
    beyond = 0
    for entry in generated.values():
        sim_ns = entry["sim_ns"]
        row = truth_by_ns.get(sim_ns)
        if row is None:
            continue
        ego_pose = Pose.from_dict(row["ego"]["pose"])
        truths = [t for t in (truth_object(a, ego_pose, rig) for a in row["actors"]) if t]
        for t in truths:
            t["bin"] = bin_of(t["distance_m"])
        beyond += sum(1 for t in truths if t["bin"] is None)
        truths = [t for t in truths if t["bin"]]
        for t in truths:
            bins[t["bin"]]["eligible_generated"] += 1
            bins[t["bin"]]["tiny"] += t["tiny"]
            bins[t["bin"]]["truncated"] += t["truncated"]
        record = usable.get(sim_ns)
        if record is None:
            continue  # no result: every object in it is an end-to-end miss
        detections = record["envelope"].get("detections", [])
        predictions = [{"class": d["class"],
                        "box": (d["bbox"]["x"], d["bbox"]["y"],
                                d["bbox"]["x"] + d["bbox"]["w"],
                                d["bbox"]["y"] + d["bbox"]["h"])} for d in detections]
        pairs, unmatched = match(truths, predictions)
        predictions_total += len(predictions)
        false_positives += len(unmatched)
        for t in truths:
            bins[t["bin"]]["eligible_received"] += 1
        for t, p in pairs:
            bins[truths[t]["bin"]]["tp"] += 1
        matches[sim_ns] = [(truths[t], detections[p]) for t, p in pairs]
    for b in bins.values():
        b["recall_received"] = b["tp"] / b["eligible_received"] \
            if b["eligible_received"] else None
        b["recall_end_to_end"] = b["tp"] / b["eligible_generated"] \
            if b["eligible_generated"] else None
    tp = sum(b["tp"] for b in bins.values())
    return matches, {
        "bins": bins, "true_positives": tp, "false_positives": false_positives,
        "predictions": predictions_total,
        "precision": tp / predictions_total if predictions_total else None,
        "truth_beyond_120_m": beyond,
        "note": "false positives include detections of things that are not "
                "scripted actors (the scene itself)",
    }


def metric_section(metric_rows, matches, kinematics):
    """Metric observations against the truth of the actor each belongs to."""
    by_ns = {m["simulation"]["sim_ns"]: m for m in metric_rows}
    bins = {"%d-%d" % b: {"range": [], "lateral": [], "speed": [], "accel": [],
                          "matched": 0, "range_valid": 0, "velocity_valid": 0,
                          "accel_valid": 0, "invalid": {}} for b in BINS}
    for sim_ns, pairs in matches.items():
        metric = by_ns.get(sim_ns)
        if metric is None:
            continue
        for truth, detection in pairs:
            b = bins[truth["bin"]]
            b["matched"] += 1
            obs = next((o for o in metric["observations"]
                        if not o.get("predicted_only") and o.get("bbox") == detection["bbox"]),
                       None)
            if obs is None or not obs["valid"]:
                reason = obs["reason"] if obs else "no_observation"
                b["invalid"][reason] = b["invalid"].get(reason, 0) + 1
                continue
            b["range_valid"] += 1
            # Same reference on both sides: the near face of the object
            b["range"].append(obs["near_x_m"] - truth["near_x_m"])
            b["lateral"].append(obs["position_ego_m"][1] - truth["centre_ego"][1])
            velocity, accel = kinematics.get((truth["id"], sim_ns), (None, None))
            if obs.get("velocity_world_mps") is not None and velocity is not None \
                    and np.isfinite(velocity).all():
                b["velocity_valid"] += 1
                b["speed"].append(float(np.linalg.norm(
                    np.array(obs["velocity_world_mps"]) - velocity)))
            if obs.get("acceleration_world_mps2") is not None and accel is not None \
                    and np.isfinite(accel).all():
                b["accel_valid"] += 1
                b["accel"].append(float(np.linalg.norm(
                    np.array(obs["acceleration_world_mps2"]) - accel)))
    out = {}
    for name, b in bins.items():
        out[name] = {
            "matched_detections": b["matched"], "range_valid": b["range_valid"],
            "velocity_valid": b["velocity_valid"], "acceleration_valid": b["accel_valid"],
            "invalid": b["invalid"],
            "range_error_m": error_summary(b["range"]),
            "lateral_error_m": error_summary(b["lateral"]),
            # Magnitude of the velocity difference: the "error" is never negative
            "velocity_error_mps": error_summary(b["speed"]),
            "acceleration_error_mps2": error_summary(b["accel"]),
        }
    return out


def lane_section(usable, truth_by_ns):
    eligible = valid = 0
    lateral, heading = [], []
    reasons = {}
    for sim_ns, record in usable.items():
        lane = record["envelope"].get("lane")
        row = truth_by_ns.get(sim_ns)
        if lane is None or row is None:
            continue
        eligible += 1
        if not lane.get("valid"):
            reasons[lane.get("reason", "?")] = reasons.get(lane.get("reason", "?"), 0) + 1
            continue
        valid += 1
        truth = lane_truth_in_ego(row["lane"], Pose.from_dict(row["ego"]["pose"]),
                                  lane["reference_x_m"])
        if truth is None:
            continue
        lateral.append(lane["lateral_error_m"] - truth[0])
        heading.append(lane["heading_rad"] - truth[1])
    return {"eligible_frames": eligible, "valid": valid,
            "coverage": valid / eligible if eligible else None,
            "invalid_reasons": reasons,
            "centre_error_m": error_summary(lateral),
            "heading_error_rad": error_summary(heading)}


def latency_section(usable, generated, ticks, controls, commands):
    by_ns = {e["sim_ns"]: (f, e) for f, e in generated.items()}
    tick_to_result, sent_to_result = [], []
    for sim_ns, record in usable.items():
        frame, entry = by_ns[sim_ns]
        tick = ticks.get(frame)
        if tick:
            tick_to_result.append((record["received_mono_ns"] - tick["mono_ns"]) / 1e6)
        if "sent_mono_ns" in entry:
            sent_to_result.append((record["received_mono_ns"] - entry["sent_mono_ns"]) / 1e6)
    basis_of = {c["command"]["command_seq"]: c["command"]["basis_sim_ns"]
                for c in commands if "command" in c}
    tick_mono = {t["sim_ns"]: t["mono_ns"] for t in ticks.values()}
    source_to_apply, age_at_apply = [], []
    for applied in controls:
        if applied.get("event") != "applied" or applied["reason"] != "":
            continue
        basis = basis_of.get(applied.get("command_seq"))
        if basis in tick_mono:
            source_to_apply.append((applied["apply_mono_ns"] - tick_mono[basis]) / 1e6)
            age_at_apply.append((applied["sim_ns_after"] - basis) / 1e6)
    return {
        "tick_to_result_ms": percentile_summary(tick_to_result),
        "jpeg_sent_to_result_ms": percentile_summary(sent_to_result),
        "tick_to_control_applied_ms": percentile_summary(source_to_apply),
        "perception_age_when_applied_sim_ms": percentile_summary(age_at_apply),
        "note": "this PC's monotonic clock only; applied commands that drove "
                "the vehicle only",
    }


def control_section(controls, truth_rows, manifest):
    applied = [c for c in controls if c.get("event") == "applied"]
    reasons = {}
    for c in applied:
        reasons[c["reason"] or "driving"] = reasons.get(c["reason"] or "driving", 0) + 1
    driving = [c for c in applied if c["reason"] == ""]
    first_drive = driving[0]["tick"] if driving else None
    last_drive = driving[-1]["tick"] if driving else None
    stop = next((c for c in applied if first_drive is not None
                 and c["tick"] > last_drive), None)
    both = sum(1 for c in applied if c["throttle"] > 0 and c["brake"] > 0)

    bumper = manifest["vehicle"]["front_bumper_m"]
    half_width = manifest["vehicle"]["width_m"] / 2.0
    deviations, speeds, gaps = [], [], {}
    events = {"collision": [], "lane_invasion": []}
    for row in truth_rows:
        for event in row["events"]:
            events.setdefault(event["event"], []).append(
                dict(event, tick=row["tick"]))
        speeds.append(row["ego"]["speed_mps"])
        if first_drive is not None and first_drive <= row["tick"] <= last_drive:
            deviations.append(ego_lateral_deviation(row))
        ego_pose = Pose.from_dict(row["ego"]["pose"])
        for actor in row["actors"]:
            if not actor.get("present"):
                continue
            corners = actor_corners_ego(actor, ego_pose)
            # Ahead of the bumper and overlapping the vehicle's width
            if corners[:, 1].max() < -half_width or corners[:, 1].min() > half_width:
                continue
            gap = float(corners[:, 0].min()) - bumper
            if gap > -4.0:  # not an actor the vehicle has already passed
                gaps[actor["name"]] = min(gaps.get(actor["name"], 1e9), gap)
    return {
        "applied_reasons": reasons, "first_driving_tick": first_drive,
        "last_driving_tick": last_drive,
        "stop_reason": stop["reason"] if stop else None,
        "stop_tick": stop["tick"] if stop else None,
        "throttle_and_brake_together": both,
        "max_speed_mps": max(speeds) if speeds else None,
        "final_speed_mps": speeds[-1] if speeds else None,
        "ego_lateral_deviation_m": error_summary(deviations),
        "min_bumper_gap_m": gaps,
        "collisions": events["collision"], "lane_invasions": events["lane_invasion"],
        "commands_rejected": {},
    }


def footprints_overlap(a, b):
    """Two convex quadrilaterals, (4, 2) each, corners in order: whether
    they share any area (separating axes)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    for polygon in (a, b):
        for i in range(4):
            edge = polygon[(i + 1) % 4] - polygon[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True


def ego_footprint(vehicle):
    """The vehicle's box on the road, in its own frame."""
    front = vehicle["front_bumper_m"]
    # Runs written before the rear bumper was recorded: as long behind as ahead
    rear = vehicle.get("rear_bumper_m", -front)
    half = vehicle["width_m"] / 2.0
    return np.array([[front, -half], [front, half], [rear, half], [rear, -half]])


def actor_footprint(actor, ego_pose):
    """An actor's box on the road, in the ego frame, corners in order."""
    corners = actor_corners_ego(actor, ego_pose)
    # Built as sx, sy, sz in (-1, 1): the lower face is every second corner
    low = corners[::2, :2]
    return low[[0, 1, 3, 2]]


def show_section(controls, commands, truth_rows, manifest, matches):
    """What the drive show and the park show are judged on: whether the
    vehicle drove on after a brake it could release, the lane change, the
    stretch it followed another vehicle on, whether its box ever shared
    ground with another's, and where and how it ended a park.

    A brake that was released is read from the controller's own record of
    it (debug.hold in commands.jsonl), not from the reason of the last
    stop: a vehicle that drives on has no last stop.
    """
    scenario = manifest["scenario_config"]
    vehicle = manifest["vehicle"]
    applied = sorted((c for c in controls if c.get("event") == "applied"),
                     key=lambda c: c["tick"])
    tick_of = {c["sim_ns_after"]: c["tick"] for c in applied}
    out = {}

    # Holds: runs of commands the controller marked as a brake it can release
    holds = []
    for record in commands:
        command, debug = record.get("command"), record.get("debug") or {}
        if not command or "control_sim_ns" not in command:
            continue
        holding = debug.get("hold") is True or debug.get("state") == "HOLDING"
        tick = tick_of.get(command["control_sim_ns"])
        if tick is None:
            continue
        if holding and command["brake"] >= 1.0:
            if holds and holds[-1]["last_tick"] >= tick - 2:
                holds[-1]["last_tick"] = tick
                holds[-1]["reasons"].add(command["reason"])
            else:
                holds.append({"first_tick": tick, "last_tick": tick,
                              "reasons": {command["reason"]}})
    driving = [c["tick"] for c in applied if c["reason"] == "" and c["throttle"] > 0.0
               and not c.get("reverse")]
    for hold in holds:
        hold["reasons"] = sorted(hold["reasons"])
        hold["drove_before"] = any(t < hold["first_tick"] for t in driving)
        # The command of a tick is applied in the next one
        hold["drove_on_tick"] = next((t for t in driving if t > hold["last_tick"] + 1), None)
    out["holds"] = holds

    # Lane change: the last state the controller logged
    states = [(tick_of.get(r["command"]["control_sim_ns"]), r["debug"])
              for r in commands if r.get("command") and (r.get("debug") or {}).get("lane_change")]
    lane_change = {"state": None}
    if states:
        lane_change["state"] = states[-1][1]["lane_change"]
        for wanted in ("active", "done", "skipped"):
            lane_change[wanted + "_tick"] = next(
                (t for t, d in states if d["lane_change"] == wanted), None)
        lane_change["skipped_reason"] = next(
            (d["lane_change_skipped"] for _, d in states if d.get("lane_change_skipped")),
            None)
        lane_change["side"] = next(
            (d["lane_change_side"] for _, d in reversed(states)
             if d.get("lane_change_side")), None)
    out["lane_change"] = lane_change
    out["lane_ids"] = [row["lane"]["lane_id"] for row in truth_rows
                       if isinstance(row.get("lane"), dict) and "lane_id" in row["lane"]]

    # Following, and boxes that share ground
    ego_box = ego_footprint(vehicle)
    bumper = vehicle["front_bumper_m"]
    half = vehicle["width_m"] / 2.0
    brake = (manifest.get("control") or {}).get("brake")
    follow_gap = scenario.get("follow_gap_m")
    kinematics = truth_kinematics(truth_rows)
    followed, overlaps = {}, {}
    for row in truth_rows:
        ego_pose = Pose.from_dict(row["ego"]["pose"])
        speed = row["ego"]["speed_mps"]
        for actor in row["actors"]:
            if not actor.get("present"):
                continue
            if footprints_overlap(ego_box, actor_footprint(actor, ego_pose)):
                overlaps.setdefault(actor["name"], []).append(row["tick"])
            if not (follow_gap and brake) or actor["class"] == "person":
                continue
            corners = actor_corners_ego(actor, ego_pose)
            if corners[:, 1].max() < -half or corners[:, 1].min() > half:
                continue  # not ahead of the vehicle's own width
            gap = float(corners[:, 0].min()) - bumper
            velocity = kinematics.get((actor["id"], row["sim_ns"]), (None,))[0]
            if velocity is None or not np.isfinite(velocity).all():
                continue
            # Evidence of following only where following is what is asked
            # for: farther than the stopping distance, nearer than the gap
            if stopping_distance(speed, brake) < gap < follow_gap:
                entry = followed.setdefault(actor["name"], {"ticks": 0, "matched": 0,
                                                            "gaps": []})
                entry["ticks"] += 1
                entry["gaps"].append(gap)
                if abs(speed - float(np.linalg.norm(velocity))) <= 2.0:
                    entry["matched"] += 1
    dt = manifest["fixed_delta_seconds"]
    out["follow"] = {name: {"seconds_in_the_gap": e["ticks"] * dt,
                            "seconds_at_its_speed": e["matched"] * dt,
                            "gap_m": percentile_summary(e["gaps"])}
                     for name, e in followed.items()}
    out["box_overlap"] = {name: {"ticks": len(t), "first_tick": t[0]}
                          for name, t in overlaps.items()}

    # Park: judged against the stalls as they are painted (the manifest's
    # truth), which the controller never saw
    if scenario.get("park"):
        stalls = [(Stall.from_dict(spec), spec.get("occupied", False))
                  for spec in (manifest.get("truth") or {}).get("stalls", [])]

        def stall_at(pose):
            return next((i for i, (stall, _) in enumerate(stalls)
                         if stall.contains([pose["x"], pose["y"]])), None)

        poses = {row["tick"]: row["ego"] for row in truth_rows}
        last = truth_rows[-1]["ego"]["pose"] if truth_rows else None
        index = stall_at(last) if last else None
        stall = stalls[index][0] if index is not None else None
        tail = []
        for c in reversed(applied):
            ego = poses.get(c["tick"])
            if ego is None or stall is None:
                break
            pose = ego["pose"]
            ok = (c["reason"] == "parked" and c["brake"] >= 1.0 and c["throttle"] == 0.0
                  and c.get("reverse") is False and c.get("hand_brake") is True
                  and stall.contains([pose["x"], pose["y"]])
                  and abs(stall.heading_error(pose["yaw"])) < math.radians(15.0)
                  and ego["speed_mps"] <= 0.2)
            if not ok:
                break
            tail.append(c["tick"])
        inside = [row["tick"] for row in truth_rows if stall_at(row["ego"]["pose"]) is not None]
        reasons = sorted({c["reason"] for c in applied if c["reason"].startswith("park_abort")})
        # The stall as the controller last put it, from the rear camera
        believed = next((r["debug"]["park_stall"] for r in reversed(commands)
                         if (r.get("debug") or {}).get("park_stall")), None)
        out["park"] = {
            "stalls": len(stalls), "stalls_occupied": sum(1 for _, taken in stalls if taken),
            "parked_ticks_at_the_end": len(tail),
            "parked_tick": tail[-1] if tail else None,
            "parked_command_ticks": sum(1 for c in applied if c["reason"] == "parked"
                                        or c.get("hand_brake")),
            "reverse_ticks": sum(1 for c in applied if c.get("reverse")),
            "abort_reasons": reasons,
            "abort_tick": next((c["tick"] for c in applied
                                if c["reason"].startswith("park_abort")), None),
            "ticks_inside_a_stall": len(inside),
            "final_inside": stall is not None,
            "max_reverse_speed_mps": max(
                [poses[c["tick"]]["speed_mps"] for c in applied
                 if c.get("reverse") and c["tick"] in poses], default=None),
        }
        if stall is not None:
            u, v = stall.frame([last["x"], last["y"]])
            # The stall it passed just before this one: its front swung out
            # over that one while it turned in. Which side that is, from
            # where the vehicle was when the run began.
            first = truth_rows[0]["ego"]["pose"]
            side = -1.0 if stall.frame([first["x"], first["y"]])[1] < 0 else 1.0
            before = stall_at(dict(zip("xy", stall.world(0.0, side * stall.width))))
            out["park"].update(
                stall=dict(stall.as_dict(), index=index), stall_was_occupied=stalls[index][1],
                stall_before_was_occupied=stalls[before][1] if before is not None else None,
                final_u_m=u, final_v_m=v,
                final_heading_deg=math.degrees(stall.heading_error(last["yaw"])))
            if believed:
                seen = Stall.from_dict(believed)
                out["park"]["seen_stall_error"] = {
                    "centre_m": float(np.hypot(*(seen.centre - stall.centre))),
                    "direction_deg": math.degrees(stall.heading_error(seen.yaw_out)),
                    "width_m": seen.width - stall.width,
                    "length_m": seen.length - stall.length}
    return out


def fault_section(controls, scenario, dt):
    """For each fault that must stop the vehicle: ticks from its start to the
    first braked tick."""
    applied = {c["tick"]: c for c in controls if c.get("event") == "applied"}
    out = []
    for fault in scenario.get("faults", []):
        start = fault.get("from_tick", 0)
        if start not in applied:
            continue
        braked = next((t for t in sorted(applied) if t >= start
                       and applied[t]["brake"] >= 1.0 and applied[t]["throttle"] == 0.0),
                      None)
        entry = {"kind": fault["kind"], "from_tick": start, "brake_tick": braked}
        if braked is not None:
            entry["reason"] = applied[braked]["reason"]
            entry["sim_ms_to_brake"] = (braked - start) * dt * 1e3
            entry["wall_ms_to_brake"] = (applied[braked]["apply_mono_ns"]
                                         - applied[start]["apply_mono_ns"]) / 1e6
            entry["was_driving_before"] = any(
                applied[t]["reason"] == "" for t in applied if t < start)
        out.append(entry)
    rejected = {}
    for c in controls:
        if c.get("event") == "rejected":
            rejected[c["reason"]] = rejected.get(c["reason"], 0) + 1
    return out, rejected


def vehicle_section(controls, ego_rows, manifest):
    """Open-loop scenarios: what the vehicle does for a given command.

    Steering: sign and gain of the response to a constant steer. Braking:
    the delay from the brake command to the first loss of speed, and the
    deceleration between 90 % and 10 % of the speed it had.
    """
    applied = {c["sim_ns_after"]: c for c in controls
               if c.get("event") == "applied" and c["reason"] == "open_loop"}
    if not applied:
        return None
    rows = [(e, applied[e["sim_ns"]]) for e in ego_rows if e["sim_ns"] in applied]
    wheelbase = manifest["vehicle"]["wheelbase_m"]
    max_steer = math.radians(manifest["vehicle"]["max_steer_deg"])
    out = {"steer": [], "brake": None}
    segments = {}
    paths = {}
    for ego, command in rows:
        # Backwards at this throttle the vehicle stays under 1 m/s
        if command["steer"] != 0.0 and ego["speed_mps"] > (
                0.3 if command.get("reverse") else 1.0):
            # Speed is unsigned: in reverse the yaw rate over it has the
            # sign of the turn, not of the path's curvature
            key = (bool(command.get("reverse")), command["steer"])
            segments.setdefault(key, []).append(ego["yaw_rate_radps"] / ego["speed_mps"])
            yaw = math.radians(ego["pose"]["yaw"])
            paths.setdefault(key, []).append(
                (ego["pose"]["x"] - wheelbase / 2.0 * math.cos(yaw),
                 ego["pose"]["y"] - wheelbase / 2.0 * math.sin(yaw)))
    for (reverse, steer), curvatures in sorted(segments.items()):
        # The first second is the vehicle turning in
        settled = curvatures[len(curvatures) // 2:]
        curvature = float(np.mean(settled))
        # The yaw rate over the speed is the curvature of the origin's path;
        # the wheel angle follows from the rear axle's, half a wheelbase back
        rear = math.sqrt(max(1e-9, 1.0 / curvature ** 2 - (wheelbase / 2.0) ** 2)) \
            if curvature else float("inf")
        path = np.array(paths[(reverse, steer)])
        if abs(steer) >= 0.2 and len(path) >= 30:
            # A turn this tight covers enough of its circle to fit one to
            # the rear axle's path: the radius a manoeuvre is planned with
            A = np.c_[2 * path, np.ones(len(path))]
            cx, cy, k = np.linalg.lstsq(A, (path ** 2).sum(axis=1), rcond=None)[0]
            rear = math.sqrt(max(1e-9, k + cx * cx + cy * cy))
        wheel = math.atan(wheelbase / rear) * (1 if curvature > 0 else -1)
        out["steer"].append({
            "steer": steer, "reverse": reverse, "curvature_inv_m": curvature,
            "wheel_angle_rad": wheel, "turns_right": curvature > 0,
            # Backwards, the same wheel angle turns the vehicle the other way
            "sign_ok": (curvature > 0) == ((steer > 0) != reverse),
            "rear_axle_radius_m": rear if curvature else None,
            # 1.0 would be: steer 1 turns the wheels by max_steer_deg
            "gain": abs(wheel) / (abs(steer) * max_steer)})
    braking = [(ego, c) for ego, c in rows if c["brake"] >= 1.0]
    moving = [i for i, (ego, c) in enumerate(rows) if c["brake"] >= 1.0
              and i > 0 and rows[i - 1][1]["brake"] < 1.0 and ego["speed_mps"] > 1.0]
    if moving:
        start = moving[0]
        v0 = rows[start][0]["speed_mps"]
        t0 = rows[start][0]["sim_ns"]
        series = [(e["sim_ns"] - t0, e["speed_mps"], e["pose"]) for e, _ in rows[start:]]
        onset = next((t for t, v, _ in series if v < v0 - 0.05), None)
        t90 = next((t for t, v, _ in series if v <= 0.9 * v0), None)
        t10 = next((t for t, v, _ in series if v <= 0.1 * v0), None)
        stopped = next((i for i, (t, v, _) in enumerate(series) if v < 0.05), None)
        distance = None
        if stopped is not None:
            a, b = series[0][2], series[stopped][2]
            distance = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        out["brake"] = {
            "speed_at_brake_mps": v0,
            # The command is applied before the tick whose state this is
            "delay_to_first_deceleration_s": None if onset is None else onset * 1e-9 + 1 / 30,
            "deceleration_90_to_10_mps2": (0.8 * v0 / ((t10 - t90) * 1e-9))
            if t90 is not None and t10 is not None and t10 > t90 else None,
            "stopping_distance_m": distance,
            "stopping_time_s": None if stopped is None else series[stopped][0] * 1e-9}
    del braking
    return out


def depth_fixture_section(run_dir, generated, truth_by_ns, rig, every=15,
                          front_bumper_m=None):
    """What the depth image measures, checked on the road itself.

    The road is a known plane: z = ground_z in the ego frame. Each sampled
    road pixel is turned into a point twice, reading depth as the distance
    along the optical axis and as the distance along the ray; the reading
    that puts the points on the road is the right one. With it, the point
    must also agree with the ray-plane intersection of the same pixel.

    Then the silhouette of each scripted object in the depth image against
    its projected bounding box: left, right and bottom edge, in pixels.
    """
    axis, radial, plane_xy, edges = [], [], [], []
    frames = 0
    for entry in sorted(generated.values(), key=lambda e: e["sim_ns"])[::every]:
        path = entry.get("depth_artifact")
        row = truth_by_ns.get(entry["sim_ns"])
        if not path or row is None:
            continue
        with open(os.path.join(run_dir, path), "rb") as f:
            raw = zlib.decompress(f.read())
        h, w = entry["depth_shape"]
        bgr = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        depth = decode_depth(np.concatenate(
            [bgr, np.full((h, w, 1), 255, np.uint8)], axis=2)).astype(np.float64)
        frames += 1
        ego_pose = Pose.from_dict(row["ego"]["pose"])
        objects = [t for t in (truth_object(a, ego_pose, rig) for a in row["actors"]) if t]

        # Road pixels on a grid over the lower image, away from the objects
        v, u = np.mgrid[h // 2 + 20:h:16, 8:w:16]
        keep = np.ones(u.shape, bool)
        for t in objects:
            x0, y0, x1, y1 = t["box"]
            keep &= ~((u >= x0 - 20) & (u <= x1 + 20) & (v >= y0 - 20) & (v <= y1 + 60))
        u, v = u[keep], v[keep]
        d = depth[v, u]
        expected, on_road = rig.uv_to_ground(u + 0.5, v + 0.5, 60.0)
        # Only where the road is the nearest thing: the lane ahead, within 60 m
        on_road &= np.abs(expected[:, 1]) < 5.0
        if front_bumper_m is not None:
            # ... and not the vehicle's own bonnet, which a camera pitched
            # down has in its lowest rows. A bonnet up to half the camera's
            # height at the bumper hides the road as far as twice the
            # distance to the bumper.
            on_road &= expected[:, 0] > 2.0 * front_bumper_m - rig.x + 0.25
        u, v, d, expected = u[on_road], v[on_road], d[on_road], expected[on_road]
        along_axis = rig.depth_to_ego(u + 0.5, v + 0.5, d)
        along_ray = rig.depth_to_ego(u + 0.5, v + 0.5, d, radial=True)
        axis.extend(along_axis[:, 2] - rig.ground_z)
        radial.extend(along_ray[:, 2] - rig.ground_z)
        plane_xy.extend(np.linalg.norm(along_axis[:, :2] - expected, axis=1)
                        / np.maximum(1.0, expected[:, 0]))

        for t in objects:
            x0, y0, x1, y1 = [int(round(c)) for c in t["box"]]
            pad = 12
            ys, xs = np.mgrid[max(0, y0 - pad):min(h, y1 + pad),
                              max(0, x0 - pad):min(w, x1 + pad)]
            points = rig.depth_to_ego(xs + 0.5, ys + 0.5, depth[ys, xs])
            solid = (points[..., 2] > rig.ground_z + 0.15) & \
                    (np.abs(points[..., 0] - t["near_x_m"]) < 6.0)
            if solid.sum() < 20:
                continue
            columns = xs[solid]
            rows_ = ys[solid]
            edges.append({"name": t["name"], "distance_m": t["distance_m"],
                          "left_px": float(columns.min() - t["box"][0]),
                          "right_px": float(columns.max() + 1 - t["box"][2]),
                          "bottom_px": float(rows_.max() + 1 - t["box"][3])})
    if not frames:
        return None
    axis, radial = np.array(axis), np.array(radial)
    edge_values = [abs(e[k]) for e in edges for k in ("left_px", "right_px", "bottom_px")]
    return {
        "frames": frames, "road_pixels": int(len(axis)),
        "road_height_error_m": {"depth_along_optical_axis": error_summary(axis),
                                "depth_along_ray": error_summary(radial)},
        "depth_is": "along_optical_axis"
        if np.sqrt(np.mean(axis ** 2)) < np.sqrt(np.mean(radial ** 2)) else "along_ray",
        "road_point_error_relative": percentile_summary(plane_xy),
        "silhouette_edges_px": percentile_summary(edge_values),
        "silhouettes": len(edges),
        "note": "silhouette edges compare the rendered mesh with the actor's "
                "bounding box, which is larger than the mesh",
    }


def lane_parity_section(run_dir, usable, model_path):
    """The device's lane pixels against the same model on this PC's CPU.

    For every JPEG kept from the run (rgb/, as it was sent): ONNX Runtime on
    the CPU, FP32, with the device's pre-processing (area resize), and the
    reference decoder; then the difference in pixels to what the device
    reported for that frame, row by row. Rows only one side found count as
    disagreements. A second pass uses the pre-processing of the model's
    authors (PIL resize) to show what the choice of resize costs.
    """
    import cv2
    import onnxruntime
    from video_analytics_simulation import lane as lane_reference

    session = onnxruntime.InferenceSession(model_path,
                                           providers=["CPUExecutionProvider"])
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    def run(rgb_resized):
        x = (rgb_resized.astype(np.float32) / 255.0 - mean) / std
        return session.run(None, {"image": x.transpose(2, 0, 1)[None]})[0][0]

    def compare(a, b):
        """Per-row differences in x of two decoded results; and how many rows
        only one of them has."""
        differences, only_one = [], 0
        for lane_a, lane_b in zip(a, b):
            rows_a = {y: x for x, y in lane_a}
            rows_b = {y: x for x, y in lane_b}
            for y in set(rows_a) | set(rows_b):
                if y in rows_a and y in rows_b:
                    differences.append(abs(rows_a[y] - rows_b[y]))
                else:
                    only_one += 1
        return differences, only_one

    directory = os.path.join(run_dir, "rgb")
    device, authors, mask_device, mask_authors, images = [], [], 0, 0, 0
    for name in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
        if not name.endswith(".jpg"):
            continue
        record = usable.get(int(name[:-4]))
        if record is None or "lane" not in record["envelope"]:
            continue
        bgr = cv2.imread(os.path.join(directory, name))
        h, w = bgr.shape[:2]
        reference = lane_reference.decode_ufld(
            run(cv2.resize(bgr, (800, 288), interpolation=cv2.INTER_AREA)[:, :, ::-1]),
            w, h)
        reported = [[tuple(p) for p in l] for l in record["envelope"]["lane"]["pixels"]]
        d, only = compare(reported, reference)
        device += d
        mask_device += only
        try:
            from PIL import Image
            pil = Image.fromarray(bgr[:, :, ::-1]).resize((800, 288), Image.BILINEAR)
            d, only = compare(reference, lane_reference.decode_ufld(run(np.asarray(pil)), w, h))
            authors += d
            mask_authors += only
        except ImportError:
            pass
        images += 1
    if not images:
        return None
    return {"images": images, "model": model_path,
            "device_vs_cpu_fp32_px": percentile_summary(device),
            "device_vs_cpu_rows_only_one_has": mask_device,
            "area_resize_vs_authors_resize_px": percentile_summary(authors),
            "area_resize_vs_authors_rows_only_one_has": mask_authors}


def reprojection_section(run_dir, truth_by_ns, rig):
    """Red markers drawn at known 3-D points against where the calibration
    projects those points: an error in pixels that does not go through the
    depth image or the detector.

    Each marker is found in the kept JPEGs as the centroid of the red pixels
    near its projection (within 12 px); a marker that cannot be found is
    counted, not skipped.
    """
    import cv2
    path = os.path.join(run_dir, "markers.json")
    directory = os.path.join(run_dir, "rgb")
    if not os.path.exists(path) or not os.path.isdir(directory):
        return None
    with open(path) as f:
        world = np.array(json.load(f)["points_world_m"])
    errors, missing, images = [], 0, 0
    for name in sorted(os.listdir(directory)):
        row = truth_by_ns.get(int(name[:-4])) if name.endswith(".jpg") else None
        # Only frames of the empty road: a car would hide some of the markers
        if row is None or any(a.get("present") for a in row["actors"]):
            continue
        bgr = cv2.imread(os.path.join(directory, name)).astype(np.int32)
        # Red, and nothing else in this scene is: R high, G and B low
        red = (bgr[:, :, 2] > 150) & (bgr[:, :, 1] < 90) & (bgr[:, :, 0] < 90)
        ego = Pose.from_dict(row["ego"]["pose"]).from_parent(world)
        uv, depth = rig.project(ego)
        images += 1
        for (u, v), d in zip(uv, depth):
            if d <= 0 or not (12 <= u < rig.width - 12 and 12 <= v < rig.height - 12):
                continue
            u0, v0 = int(round(u)), int(round(v))
            window = red[v0 - 12:v0 + 13, u0 - 12:u0 + 13]
            ys, xs = np.nonzero(window)
            if len(xs) < 3:
                missing += 1
                continue
            # Pixel centres: pixel i covers [i, i + 1)
            found_u = xs.mean() + u0 - 12 + 0.5
            found_v = ys.mean() + v0 - 12 + 0.5
            errors.append(math.hypot(found_u - u, found_v - v))
    return {"images": images, "markers": int(len(world)),
            "error_px": percentile_summary(errors), "not_found": missing}


def replay(run_dir, manifest, perception, ego_rows, generated, metric_live):
    """Recomputes the metric tracking from the stored artifacts alone."""
    rig = Rig.from_config(manifest["calibration"])
    tracker = MetricTracker(rig, manifest.get("metric") or {})
    ego_by_ns = {e["sim_ns"]: e for e in ego_rows}
    artifact = {e["sim_ns"]: e for e in generated.values()}
    live = {m["simulation"]["sim_ns"]: m for m in metric_live}
    accepted = [r["envelope"] for r in perception if r["rejected"] == ""]
    result = {"frames_in_live_run": len(accepted), "replayed": 0, "compared_values": 0,
              "max_abs_difference": 0.0, "errors": []}
    for envelope in accepted:
        sim_ns = envelope["simulation"]["sim_ns"]
        live_metric = live.get(sim_ns)
        if live_metric is None:
            result["errors"].append("%d: no live metric" % sim_ns)
            break
        # The pose the live filter used, not a fresh read of the truth log.
        pose_used = live_metric.get("pose_used")
        ego = ego_by_ns.get(sim_ns)
        if pose_used is None and ego is not None:
            pose_used = ego["pose"]
        if pose_used is None:
            result["errors"].append("%d: no pose for the filter" % sim_ns)
            break
        ego = dict(ego or {}, pose=Pose.from_dict(pose_used), sim_ns=sim_ns,
                   run_id=envelope["simulation"]["run_id"])
        if "velocity_world_mps" not in ego:
            ego["velocity_world_mps"] = [0.0, 0.0, 0.0]
        again = tracker.update(envelope, ego)
        result["replayed"] += 1
        for a, b in zip(again["observations"], live[sim_ns]["observations"]):
            for key in ("position_ego_m", "position_world_m", "velocity_world_mps",
                        "acceleration_world_mps2", "covariance"):
                if a.get(key) is None or b.get(key) is None:
                    if a.get(key) != b.get(key):
                        result["errors"].append("%d: %s differs in validity" % (sim_ns, key))
                    continue
                difference = float(np.max(np.abs(np.array(a[key]) - np.array(b[key]))))
                result["max_abs_difference"] = max(result["max_abs_difference"], difference)
                result["compared_values"] += len(a[key])
    result["tolerance"] = 1e-6
    result["pass"] = (not result["errors"] and result["replayed"] > 0
                      and result["max_abs_difference"] <= result["tolerance"])
    return result


# ── Expectations ─────────────────────────────────────────────────────────────

def check_expectations(expect, metrics):
    """The scenario's `expect` block against the measured values.
    Returns [(name, passed, detail)]."""
    control = metrics["control"]
    results = []

    def add(name, passed, detail):
        results.append((name, bool(passed), detail))

    for key, value in (expect or {}).items():
        if key == "collisions":
            add("collisions == %d" % value, len(control["collisions"]) == value,
                "%d" % len(control["collisions"]))
        elif key == "drives":
            add("vehicle drives: %s" % value,
                (control["first_driving_tick"] is not None) == value,
                "first driving tick %s" % control["first_driving_tick"])
        elif key == "stops":
            add("vehicle is stopped by the controller: %s" % value,
                (control["stop_reason"] is not None) == value,
                "stop reason %s" % control["stop_reason"])
        elif key == "stop_reason_prefix":
            reason = control["stop_reason"] or ""
            add("stop reason starts with %s" % value, reason.startswith(value), reason)
        elif key == "no_stop_reason_prefix":
            reason = control["stop_reason"] or ""
            add("stop reason does not start with %s" % value,
                not reason.startswith(value), reason or "none")
        elif key == "max_gap_m":
            gaps = control["min_bumper_gap_m"]
            worst = min(gaps.values()) if gaps else None
            add("stopped within %.0f m of the obstacle" % value,
                worst is not None and worst <= value,
                "gap %s" % (None if worst is None else round(worst, 2)))
        elif key == "min_gap_m":
            gaps = control["min_bumper_gap_m"]
            worst = min(gaps.values()) if gaps else None
            add("bumper gap >= %.2f m" % value, worst is not None and worst >= value,
                "min %s" % (None if worst is None else round(worst, 2)))
        elif key == "max_lateral_deviation_m":
            d = control["ego_lateral_deviation_m"]
            add("ego lateral deviation max <= %.2f m" % value,
                d["count"] > 0 and d["max_abs"] <= value, str(d.get("max_abs")))
        elif key == "lateral_rmse_m":
            d = control["ego_lateral_deviation_m"]
            add("ego lateral deviation RMSE <= %.2f m" % value,
                d["count"] > 0 and d["rmse"] <= value, str(d.get("rmse")))
        elif key == "lane_coverage":
            c = metrics["lane"]["coverage"]
            add("lane valid in >= %.0f %% of frames" % (value * 100),
                c is not None and c >= value, str(c))
        elif key == "lane_centre_rmse_m":
            e = metrics["lane"]["centre_error_m"]
            add("lane centre RMSE <= %.2f m" % value,
                e["count"] > 0 and e["rmse"] <= value, str(e.get("rmse")))
        elif key == "unknown_frames":
            add("results naming a frame that was not sent == %d" % value,
                metrics["identity"]["unknown_frame"] == value,
                str(metrics["identity"]["unknown_frame"]))
        elif key == "fault_brake_by_deadline":
            budget = metrics["control_budget"]
            for fault in metrics["faults"]:
                # Data stopped: the age limit, in simulation time. Controller
                # gone: the wall-time lease. One tick for the tick itself.
                wall = fault["kind"] == "kill_controller"
                limit = (budget["wall_lease_ms"] if wall else budget["age_limit_ms"]) + 34
                got = fault.get("wall_ms_to_brake" if wall else "sim_ms_to_brake")
                add("%s: braked within %d ms of %s" % (
                    fault["kind"], limit, "wall time" if wall else "simulation"),
                    got is not None and got <= limit and fault.get("was_driving_before"),
                    "%s ms, driving before: %s" % (
                        None if got is None else round(got), fault.get("was_driving_before")))
            if not metrics["faults"]:
                add("fault injected", False, "no fault in this run")
        elif key == "fault_brake_within_sim_ms":
            for fault in metrics["faults"]:
                ok = fault.get("sim_ms_to_brake") is not None \
                    and fault["sim_ms_to_brake"] <= value
                add("%s: braked within %d ms of simulation" % (fault["kind"], value),
                    ok, str(fault.get("sim_ms_to_brake")))
        elif key == "commands_rejected":
            got = metrics["commands_rejected_by_simulator"]
            for reason in value:
                add("simulator rejected a %s command" % reason, got.get(reason, 0) > 0,
                    str(got.get(reason, 0)))
        elif key == "range_error_max_m":
            for name, b in metrics["metric"].items():
                e = b["range_error_m"]
                if e["count"]:
                    # The larger of the limit and 2 % of the bin's near edge
                    limit = max(value, 0.02 * float(name.split("-")[0]))
                    add("range error %s m <= %.2f m" % (name, limit),
                        e["max_abs"] <= limit, "max %.3f" % e["max_abs"])
        elif key == "speed_rmse_mps":
            e = metrics["metric"]["0-30"]["velocity_error_mps"]
            add("world speed RMSE 0-30 m <= %.2f m/s" % value,
                e["count"] > 0 and e["rmse"] <= value, str(e.get("rmse")))
        elif key == "accel_rmse_mps2":
            e = metrics["metric"]["0-30"]["acceleration_error_mps2"]
            add("acceleration RMSE 0-30 m <= %.2f m/s2" % value,
                e["count"] > 0 and e["rmse"] <= value, str(e.get("rmse")))
        elif key == "steer_sign":
            for item in (metrics.get("vehicle") or {}).get("steer", []):
                right = (item["steer"] > 0) != item.get("reverse", False)
                add("steer %+.2f%s turns %s" % (
                    item["steer"], " in reverse" if item.get("reverse") else "",
                    "right" if right else "left"),
                    item["sign_ok"], "curvature %.4f 1/m, gain %.2f" % (
                        item["curvature_inv_m"], item["gain"]))
            if not (metrics.get("vehicle") or {}).get("steer"):
                add("steering response measured", False, "no steering segment")
        elif key == "brake_deceleration_min_mps2":
            brake = (metrics.get("vehicle") or {}).get("brake") or {}
            got = brake.get("deceleration_90_to_10_mps2")
            add("full brake decelerates at >= %.1f m/s2" % value,
                got is not None and got >= value, str(got))
        elif key == "depth_is":
            fixture = metrics.get("depth_fixture") or {}
            add("depth is measured %s" % value, fixture.get("depth_is") == value,
                str(fixture.get("depth_is")))
        elif key == "road_height_rmse_m":
            fixture = metrics.get("depth_fixture") or {}
            e = (fixture.get("road_height_error_m") or {}).get(
                "depth_along_optical_axis", {})
            add("road from depth within %.2f m RMSE of the calibrated plane" % value,
                e.get("count", 0) > 0 and e["rmse"] <= value, str(e.get("rmse")))
        elif key == "reprojection_max_px":
            r = metrics.get("reprojection") or {}
            e = r.get("error_px") or {}
            add("markers reproject within %.1f px" % value,
                e.get("count", 0) > 0 and e["max"] <= value and r["not_found"] == 0,
                "max %s px over %s, %s not found" % (
                    e.get("max"), e.get("count", 0), r.get("not_found")))
        elif key == "resumes":
            # true: any brake that was released; a word: one whose reason
            # has it as a field ("person") or starts with it ("obstacle")
            holds = [h for h in metrics["show"]["holds"] if value is True or any(
                value in r.split(":") or r.startswith(value) for r in h["reasons"])]
            done = [h for h in holds if h["drove_before"] and h["drove_on_tick"] is not None]
            add("drove on after a brake%s" % ("" if value is True else " for " + str(value)),
                bool(value) == bool(done),
                "; ".join("held ticks %d-%d (%s), drove on at %s" % (
                    h["first_tick"], h["last_tick"], ", ".join(h["reasons"]),
                    h["drove_on_tick"]) for h in holds) or "no such brake")
        elif key == "lane_change":
            change = metrics["show"]["lane_change"]
            if isinstance(value, dict):
                state_ok = change["state"] == value.get("state", change["state"])
                side_ok = change.get("side") == value.get("side", change.get("side"))
                add("lane change %s" % value, state_ok and side_ok,
                    "state %s side %s" % (change["state"], change.get("side")))
                continue
            add("lane change: %s" % value, change["state"] == value,
                "%s%s" % (change["state"], " (%s)" % change["skipped_reason"]
                          if change.get("skipped_reason") else
                          " at tick %s" % change.get("done_tick")
                          if change.get("done_tick") is not None else ""))
        elif key == "lane_to":
            show = metrics["show"]
            ids = show.get("lane_ids") or []
            side = (show.get("lane_change") or {}).get("side")
            # CARLA follows OpenDRIVE: the lane to the right has the smaller id.
            moved = (len(ids) >= 2 and ids[-1] < ids[0]) if value == "right" \
                else (len(ids) >= 2 and ids[-1] > ids[0])
            add("moved %s by lane id" % value, bool(moved) and side == value,
                "ids %s -> %s, logged side %s" % (
                    ids[0] if ids else None, ids[-1] if ids else None, side))
        elif key == "lane_report":
            report = metrics.get("lane_report") or {}
            safe = len(control["collisions"]) == 0 and not report.get(
                "drove_invalid_too_long") and (
                    report.get("finished_with_lane") or report.get("latched_lane")
                    or report.get("never_armed"))
            add("lane report %s" % value, value == "safe" and safe,
                "valid %.2f latched %s never_armed %s invalid_run %s" % (
                    report.get("valid_fraction", 0.0), report.get("latched_lane"),
                    report.get("never_armed"), report.get("drove_invalid_too_long")))
        elif key == "follows":
            best = max(metrics["show"]["follow"].values(),
                       key=lambda f: f["seconds_at_its_speed"], default=None)
            add("followed a vehicle at its speed for >= %.1f s, between the stopping "
                "distance and the follow gap" % value,
                best is not None and best["seconds_at_its_speed"] >= value,
                "none" if best is None else "%.1f s of %.1f s in the gap, gap %.1f m" % (
                    best["seconds_at_its_speed"], best["seconds_in_the_gap"],
                    best["gap_m"]["p50"]))
        elif key == "no_box_overlap":
            overlap = metrics["show"]["box_overlap"]
            names = sorted(overlap) if value is True else list(value)
            hit = {n: overlap[n] for n in names if n in overlap}
            add("the vehicle's box never over %s's" % (
                "another" if value is True else ", ".join(value)), not hit,
                "; ".join("%s: %d ticks from %d" % (n, o["ticks"], o["first_tick"])
                          for n, o in hit.items()) or "0 ticks")
        elif key == "parked":
            park = metrics["show"].get("park") or {}
            needed = 15 if value is True else int(value)
            along = park.get("final_u_m")
            across = park.get("final_v_m")
            heading = park.get("final_heading_deg")
            ok = park.get("parked_ticks_at_the_end", 0) >= needed \
                and park.get("stall_was_occupied") is False \
                and park.get("stall_before_was_occupied") is False \
                and along is not None and abs(along) <= 0.40 \
                and across is not None and abs(across) <= 0.40 \
                and heading is not None and abs(heading) <= 5.0
            add("parked for the last %d ticks: in a stall that was empty, as was the one "
                "passed before it; within 0.40 m and 5 deg, standing, reverse off, "
                "brakes on" % needed,
                ok, "%d ticks; %s; %.2f m along, %.2f m beside its axis, %.1f deg" % (
                    park.get("parked_ticks_at_the_end", 0),
                    "in no stall" if not park.get("final_inside") else
                    "the stall had a vehicle in it" if park.get("stall_was_occupied") else
                    "no stall before it: the end of a row"
                    if park.get("stall_before_was_occupied") is None else
                    "the stall before it had a vehicle in it"
                    if park.get("stall_before_was_occupied") else
                    "stall at (%.2f, %.2f)" % (park["stall"]["x"], park["stall"]["y"]),
                    park.get("final_u_m", math.nan), park.get("final_v_m", math.nan),
                    park.get("final_heading_deg", math.nan)))
        elif key == "park_abort":
            park = metrics["show"].get("park") or {}
            given = park.get("abort_reasons", [])
            wanted = None if value is True else "park_abort:" + str(value)
            checks = [
                ("no parked tick", park.get("parked_command_ticks", 1) == 0),
                ("given up: %s" % (", ".join(given) or "no"),
                 bool(given) and (wanted is None or given == [wanted])),
                ("never in a stall", park.get("ticks_inside_a_stall", 1) == 0),
                ("no collision", not control["collisions"]),
                ("no box over another", not metrics["show"]["box_overlap"]),
            ]
            if wanted in (None, "park_abort:no_free_stall"):
                # Nothing to back into: it never went backwards
                checks.append(("never in reverse", park.get("reverse_ticks", 1) == 0))
            add("park given up%s" % (" (%s)" % value if wanted else ""),
                all(ok for _, ok in checks),
                "; ".join("%s%s" % (name, "" if ok else " FAILED") for name, ok in checks))
        elif key == "evaluator_must_fail":
            pass  # handled by the caller: a negative control
        else:
            add("unknown expectation %s" % key, False, "not implemented")
    return results


def lane_report_section(perception, controls, commands):
    """Whether a lane run stayed safe: finished on a valid lane, latched, or
    never armed. Driving on through more than five invalid results fails."""
    accepted = [r for r in perception if r.get("rejected") == ""]
    streak = worst = valid = 0
    for record in accepted:
        if (record.get("envelope") or {}).get("lane", {}).get("valid"):
            valid += 1
            streak = 0
        else:
            streak += 1
            worst = max(worst, streak)
    drove = any(c.get("event") == "applied" and c.get("throttle", 0) > 0
                for c in controls)
    latched = any(
        (r.get("debug") or {}).get("state") == "STOP_LATCHED"
        and str((r.get("command") or {}).get("reason", "")).startswith("lane")
        for r in commands)
    n = len(accepted)
    return {
        "valid_fraction": (valid / n) if n else 0.0,
        "finished_with_lane": bool(n) and valid == n and drove,
        "latched_lane": latched,
        "never_armed": not drove and not latched,
        "drove_invalid_too_long": worst > 5 and drove and not latched,
    }


def odometry_section(commands, truth_rows):
    """How far the dead-reckoned pose is from the simulator pose. A report,
    not a pass/fail. Missing dr_pose (older logs) yields no samples."""
    by_ns = {row["sim_ns"]: row["ego"]["pose"] for row in truth_rows}
    errors = []
    for record in commands:
        dr = (record.get("debug") or {}).get("dr_pose")
        truth = by_ns.get((record.get("command") or {}).get("control_sim_ns"))
        if not dr or not truth:
            continue
        errors.append(math.hypot(dr["x"] - truth["x"], dr["y"] - truth["y"]))
    if not errors:
        return {"samples": 0}
    return {"samples": len(errors), "final_m": errors[-1], "max_m": max(errors)}


def evaluate(run_dir, do_replay=False, lane_model=None):
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise MissingArtifact(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    rig = Rig.from_config(manifest["calibration"])
    frames, ticks = protocol.read_ledger(os.path.join(run_dir, "frames.jsonl"))
    # Ticks before the run began (tick -1) are not part of it
    truth_rows = [r for r in read_jsonl(os.path.join(run_dir, "truth.jsonl"))
                  if r["tick"] >= 0]
    controls = [c for c in read_jsonl(os.path.join(run_dir, "controls.jsonl"))
                if c.get("tick", 0) >= 0]
    ego_rows = read_jsonl(os.path.join(run_dir, "ego.jsonl"))
    perception = read_jsonl(os.path.join(run_dir, "perception.jsonl"))
    metric_rows = read_jsonl(os.path.join(run_dir, "metric.jsonl"))
    commands = read_jsonl(os.path.join(run_dir, "commands.jsonl"))
    summary_path = os.path.join(run_dir, "simulator-summary.json")
    simulator = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            simulator = json.load(f)

    truth_by_ns = {row["sim_ns"]: row for row in truth_rows}
    generated, before_run, frame_metrics = frames_section(frames, ticks)
    usable, identity = identity_section(perception, generated, manifest["run_id"],
                                        before_run)
    matches, detection = detection_section(truth_by_ns, usable, generated, rig)
    faults, rejected = fault_section(controls, manifest["scenario_config"],
                                     manifest["fixed_delta_seconds"])
    metrics = {
        "run_id": manifest["run_id"], "scenario": manifest["scenario"],
        "commit": manifest["commit"], "calibration_id": manifest["calibration_id"],
        "source": {"width": rig.width, "height": rig.height, "fov_deg": rig.fov_deg,
                   "jpeg_quality": manifest["jpeg_quality"]},
        "detector": manifest["detector"],
        "frames": frame_metrics, "identity": identity, "detection": detection,
        "metric": metric_section(metric_rows, matches, truth_kinematics(truth_rows)),
        "lane": lane_section(usable, truth_by_ns),
        "latency": latency_section(usable, generated, ticks, controls, commands),
        "control": control_section(controls, truth_rows, manifest),
        "show": show_section(controls, commands, truth_rows, manifest, matches),
        "lane_report": lane_report_section(perception, controls, commands),
        "odometry": odometry_section(commands, truth_rows),
        "faults": faults, "commands_rejected_by_simulator": rejected,
        "control_budget": {k: manifest.get("control", {}).get(k) for k in (
            "age_limit_ms", "wall_lease_ms", "pose_budget_ms")},
        "rates": {
            "simulated_fps": simulator.get("simulated_fps"),
            "wall_fps": simulator.get("wall_fps"),
            "results_per_wall_second": (identity["matched"] / simulator["wall_seconds"])
            if simulator.get("wall_seconds") else None,
            "jpeg_mbit_per_s": simulator.get("jpeg_mbit_per_s"),
            # Paced at 30: anything short of it is the host not keeping up
            "real_time_30_fps": bool(simulator.get("wall_fps", 0) >= 29.95),
        },
        "simulator": simulator,
    }
    if lane_model:
        metrics["lane_parity"] = lane_parity_section(run_dir, usable, lane_model)
    vehicle = vehicle_section(controls, ego_rows, manifest)
    if vehicle:
        metrics["vehicle"] = vehicle
    if manifest["scenario_config"].get("depth_fixture"):
        metrics["depth_fixture"] = depth_fixture_section(
            run_dir, generated, truth_by_ns, rig,
            front_bumper_m=manifest["vehicle"].get("front_bumper_m"))
        metrics["reprojection"] = reprojection_section(run_dir, truth_by_ns, rig)
    if do_replay:
        # Every frame of the ledger: the tracker's state at the first tick of
        # the run comes from the frames before it
        metrics["replay"] = replay(run_dir, manifest, perception, ego_rows,
                                   {f: e for f, e in frames.items() if "sim_ns" in e},
                                   metric_rows)

    expect = manifest["scenario_config"].get("expect")
    results = check_expectations(expect, metrics)
    if do_replay:
        results.append(("replay from artifacts reproduces the live metric tracking",
                        metrics["replay"]["pass"],
                        "%d frames, max difference %.2e, %s" % (
                            metrics["replay"]["replayed"],
                            metrics["replay"]["max_abs_difference"],
                            "; ".join(metrics["replay"]["errors"][:2]) or "no errors")))
    failed = [r for r in results if not r[1]]
    if (expect or {}).get("evaluator_must_fail"):
        # A negative control: the run did something wrong on purpose, and the
        # evaluation is right only if it noticed
        verdict = "pass" if failed else "fail"
    else:
        verdict = "fail" if failed else ("pass" if results else "no expectations")
    metrics["expectations"] = [{"name": n, "pass": p, "measured": d} for n, p, d in results]
    metrics["verdict"] = verdict
    return metrics


def summary_markdown(m):
    def fmt(value, digits=2):
        return "-" if value is None else ("%.*f" % (digits, value))

    lines = ["# Run %s: %s" % (m["run_id"], m["scenario"]), "",
             "Verdict: **%s**" % m["verdict"], ""]
    if m["expectations"]:
        lines += ["| Expectation | Result | Measured |", "|---|---|---|"]
        lines += ["| %s | %s | %s |" % (e["name"], "pass" if e["pass"] else "FAIL",
                                        e["measured"]) for e in m["expectations"]]
        lines.append("")
    f, i, r = m["frames"], m["identity"], m["rates"]
    lines += [
        "## Frames", "",
        "- %d ticks; %d RGB images produced, %d never produced" % (
            f["ticks"], f["rgb_generated"], f["rgb_never_generated"]),
        "- what became of them: %s" % json.dumps(f["status"]),
        "- results received: %d; joined to a sent frame: %d; of frames sent "
        "before the run began: %d; naming an unknown frame: %d; repeated: %d; "
        "wrong run: %d" % (
            i["received"], i["matched"], i["before_run"], i["unknown_frame"],
            i["repeated"], i["wrong_run"]),
        "- delivery: %.1f %% of produced frames came back with a result" % (
            100 * i["delivery_rate"]),
        "- wall rate %s ticks/s (simulated %s), results %s/s, JPEG %s Mbit/s; "
        "real time at 30 FPS: %s" % (
            fmt(r["wall_fps"], 1), fmt(r["simulated_fps"], 0),
            fmt(r["results_per_wall_second"], 1), fmt(r["jpeg_mbit_per_s"], 1),
            "yes" if r["real_time_30_fps"] else "NO"),
        "", "## Latency (this PC's clock)", "",
        "| | count | P50 ms | P95 ms | max ms |", "|---|---|---|---|---|"]
    for name, key in (("tick to result", "tick_to_result_ms"),
                      ("JPEG sent to result", "jpeg_sent_to_result_ms"),
                      ("tick to control applied", "tick_to_control_applied_ms"),
                      ("perception age when applied (simulation)",
                       "perception_age_when_applied_sim_ms")):
        s = m["latency"][key]
        lines.append("| %s | %d | %s | %s | %s |" % (
            name, s["count"], fmt(s.get("p50"), 1), fmt(s.get("p95"), 1),
            fmt(s.get("max"), 1)))
    d = m["detection"]
    lines += ["", "## Detection (source %dx%d, FOV %s, model %s)" % (
        m["source"]["width"], m["source"]["height"], m["source"]["fov_deg"],
        m["detector"]["model"]), "",
        "| Distance m | objects in received frames | found | recall, received | "
        "objects in all frames | recall, end to end | under 10 px high |",
        "|---|---|---|---|---|---|---|"]
    for name, b in d["bins"].items():
        lines.append("| %s | %d | %d | %s | %d | %s | %d |" % (
            name, b["eligible_received"], b["tp"], fmt(b["recall_received"]),
            b["eligible_generated"], fmt(b["recall_end_to_end"]), b["tiny"]))
    lines += ["", "False positives: %d of %d predictions (%s)." % (
        d["false_positives"], d["predictions"], d["note"]), "",
        "## Metric tracking", "",
        "| Distance m | matched | range valid | range RMSE m | range max m | "
        "velocity valid | speed RMSE m/s | accel valid | accel RMSE m/s2 |",
        "|---|---|---|---|---|---|---|---|---|"]
    for name, b in m["metric"].items():
        lines.append("| %s | %d | %d | %s | %s | %d | %s | %d | %s |" % (
            name, b["matched_detections"], b["range_valid"],
            fmt(b["range_error_m"].get("rmse"), 3),
            fmt(b["range_error_m"].get("max_abs"), 3), b["velocity_valid"],
            fmt(b["velocity_error_mps"].get("rmse"), 3), b["acceleration_valid"],
            fmt(b["acceleration_error_mps2"].get("rmse"), 3)))
    lane, c = m["lane"], m["control"]
    lines += ["", "## Lane", "",
              "- valid in %d of %d frames (%s); invalid: %s" % (
                  lane["valid"], lane["eligible_frames"],
                  "-" if lane["coverage"] is None else "%.1f %%" % (100 * lane["coverage"]),
                  json.dumps(lane["invalid_reasons"])),
              "- lane centre against the map: RMSE %s m, max %s m; heading RMSE %s rad" % (
                  fmt(lane["centre_error_m"].get("rmse"), 3),
                  fmt(lane["centre_error_m"].get("max_abs"), 3),
                  fmt(lane["heading_error_rad"].get("rmse"), 4)),
              "", "## Control", "",
              "- applied per tick: %s" % json.dumps(c["applied_reasons"]),
              "- drove from tick %s to %s; stop: %s at tick %s" % (
                  c["first_driving_tick"], c["last_driving_tick"], c["stop_reason"],
                  c["stop_tick"]),
              "- max speed %s m/s, final %s m/s; throttle and brake together: %d ticks" % (
                  fmt(c["max_speed_mps"]), fmt(c["final_speed_mps"]),
                  c["throttle_and_brake_together"]),
              "- ego from the lane centre while driving: RMSE %s m, max %s m" % (
                  fmt(c["ego_lateral_deviation_m"].get("rmse"), 3),
                  fmt(c["ego_lateral_deviation_m"].get("max_abs"), 3)),
              "- smallest bumper gap: %s" % json.dumps(
                  {k: round(v, 2) for k, v in c["min_bumper_gap_m"].items()}),
              "- collisions: %d, lane markings crossed: %d" % (
                  len(c["collisions"]), len(c["lane_invasions"])),
              "- commands the simulator rejected: %s" % json.dumps(
                  m["commands_rejected_by_simulator"])]
    show = m.get("show") or {}
    if show.get("holds") or show.get("lane_change", {}).get("state") \
            or show.get("follow") or show.get("park") or show.get("box_overlap"):
        lines += ["", "## Show", ""]
        for hold in show["holds"]:
            lines.append("- braked and held, ticks %d to %d (%s); %s" % (
                hold["first_tick"], hold["last_tick"], ", ".join(hold["reasons"]),
                "drove on at tick %d" % hold["drove_on_tick"]
                if hold["drove_on_tick"] is not None else "did not drive on"))
        change = show["lane_change"]
        if change["state"]:
            lines.append("- lane change: %s; began at tick %s, done at tick %s%s" % (
                change["state"], change.get("active_tick"), change.get("done_tick"),
                ", skipped: %s" % change["skipped_reason"]
                if change.get("skipped_reason") else ""))
        for name, f in show["follow"].items():
            lines.append("- behind %s between the stopping distance and the follow gap "
                         "for %.1f s, %.1f s of them within 2 m/s of its speed; gap "
                         "median %s m" % (name, f["seconds_in_the_gap"],
                                          f["seconds_at_its_speed"],
                                          fmt(f["gap_m"].get("p50"), 1)))
        lines.append("- the vehicle's box over another's: %s" % (
            "; ".join("%s for %d ticks from tick %d" % (n, o["ticks"], o["first_tick"])
                      for n, o in show["box_overlap"].items()) or "never"))
        park = show.get("park")
        if park:
            lines.append("- park: %s; backwards for %d ticks, at most %s m/s" % (
                "parked from tick %s" % park["parked_tick"]
                if park["parked_ticks_at_the_end"] else
                "given up (%s) at tick %s" % (", ".join(park["abort_reasons"]),
                                              park["abort_tick"])
                if park["abort_reasons"] else "not parked",
                park["reverse_ticks"], fmt(park.get("max_reverse_speed_mps"))))
            if park.get("stall"):
                lines.append("- ended in the stall at (%.2f, %.2f), %s when the run began: "
                             "%s m along and %s m beside its axis, %s deg off its "
                             "direction" % (
                                 park["stall"]["x"], park["stall"]["y"],
                                 "occupied" if park["stall_was_occupied"] else "empty",
                                 fmt(park["final_u_m"]), fmt(park["final_v_m"]),
                                 fmt(park["final_heading_deg"], 1)))
            else:
                lines.append("- ended in no stall (of %d, %d occupied when the run began)"
                             % (park["stalls"], park["stalls_occupied"]))
            error = park.get("seen_stall_error")
            if error:
                lines.append("- that stall as the rear camera's model put it at the end: "
                             "centre %s m off, direction %s deg off, %s m wider, %s m "
                             "longer than painted" % (
                                 fmt(error["centre_m"]), fmt(error["direction_deg"], 1),
                                 fmt(error["width_m"]), fmt(error["length_m"])))
    for fault in m["faults"]:
        lines.append("- fault %s at tick %d: braked at tick %s (%s), %s ms of "
                     "simulation, %s ms of wall time" % (
                         fault["kind"], fault["from_tick"], fault.get("brake_tick"),
                         fault.get("reason"), fmt(fault.get("sim_ms_to_brake"), 0),
                         fmt(fault.get("wall_ms_to_brake"), 0)))
    if m.get("vehicle"):
        lines += ["", "## Vehicle response (open loop)", ""]
        for item in m["vehicle"]["steer"]:
            lines.append("- steer %+.2f%s: turns %s, rear axle on a radius of %.1f m, "
                         "wheel angle %.2f deg, gain %.2f of the nominal, sign %s" % (
                             item["steer"], " in reverse" if item.get("reverse") else "",
                             "right" if item["turns_right"] else "left",
                             item.get("rear_axle_radius_m") or float("inf"),
                             abs(math.degrees(item["wheel_angle_rad"])), item["gain"],
                             "ok" if item["sign_ok"] else "WRONG"))
        b = m["vehicle"]["brake"]
        if b:
            lines.append("- full brake from %s m/s: first deceleration after %s s, "
                         "%s m/s2 between 90 %% and 10 %% of the speed, stopped "
                         "after %s m and %s s" % (
                             fmt(b["speed_at_brake_mps"]),
                             fmt(b["delay_to_first_deceleration_s"], 3),
                             fmt(b["deceleration_90_to_10_mps2"]),
                             fmt(b["stopping_distance_m"]), fmt(b["stopping_time_s"])))
    if m.get("depth_fixture"):
        d = m["depth_fixture"]
        h = d["road_height_error_m"]
        lines += ["", "## Depth and calibration fixture", "",
                  "- %d frames, %d road pixels" % (d["frames"], d["road_pixels"]),
                  "- road height from depth, reading depth along the optical axis: "
                  "RMSE %s m, max %s m" % (
                      fmt(h["depth_along_optical_axis"].get("rmse"), 4),
                      fmt(h["depth_along_optical_axis"].get("max_abs"), 4)),
                  "- reading it along the ray: RMSE %s m, max %s m" % (
                      fmt(h["depth_along_ray"].get("rmse"), 4),
                      fmt(h["depth_along_ray"].get("max_abs"), 4)),
                  "- so depth is measured **%s**" % d["depth_is"].replace("_", " "),
                  "- road point from depth against the ray-plane intersection, "
                  "relative to range: P95 %s, max %s" % (
                      fmt(d["road_point_error_relative"].get("p95"), 4),
                      fmt(d["road_point_error_relative"].get("max"), 4)),
                  "- %d object silhouettes against the projected box: P50 %s px, "
                  "P95 %s px, max %s px (%s)" % (
                      d["silhouettes"], fmt(d["silhouette_edges_px"].get("p50"), 1),
                      fmt(d["silhouette_edges_px"].get("p95"), 1),
                      fmt(d["silhouette_edges_px"].get("max"), 1), d["note"])]
        r = m.get("reprojection")
        if r:
            lines.append("- %d markers at known points in %d images: reprojection "
                         "error P50 %s px, P95 %s px, max %s px; %d not found" % (
                             r["markers"], r["images"], fmt(r["error_px"].get("p50")),
                             fmt(r["error_px"].get("p95")), fmt(r["error_px"].get("max")),
                             r["not_found"]))
    if m.get("lane_parity"):
        p = m["lane_parity"]
        a, b = p["device_vs_cpu_fp32_px"], p["area_resize_vs_authors_resize_px"]
        lines += ["", "## Lane model: device against CPU FP32", "",
                  "- %d images, %d lane rows compared: median %s px, P95 %s px, "
                  "max %s px; %d rows found by one side only" % (
                      p["images"], a["count"], fmt(a.get("p50"), 1),
                      fmt(a.get("p95"), 1), fmt(a.get("max"), 1),
                      p["device_vs_cpu_rows_only_one_has"]),
                  "- this pre-processing against the authors' (PIL resize), same "
                  "CPU model: median %s px, P95 %s px, max %s px; %d rows by one "
                  "side only" % (fmt(b.get("p50"), 1), fmt(b.get("p95"), 1),
                                 fmt(b.get("max"), 1),
                                 p["area_resize_vs_authors_rows_only_one_has"])]
    if "replay" in m:
        p = m["replay"]
        lines += ["", "## Replay from artifacts", "",
                  "- %d of %d frames replayed, %d values compared, largest "
                  "difference %.2e: %s" % (
                      p["replayed"], p["frames_in_live_run"], p["compared_values"],
                      p["max_abs_difference"], "pass" if p["pass"] else "FAIL"),
                  ] + ["- %s" % e for e in p["errors"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--lane-model", help="ONNX lane model: compare the "
                        "device's lane pixels with it on this PC's CPU")
    args = parser.parse_args()
    try:
        metrics = evaluate(args.run_dir, args.replay, args.lane_model)
    except MissingArtifact as error:
        print("evaluation failed: %s is missing" % error, file=sys.stderr)
        return 2
    protocol.write_json(os.path.join(args.run_dir, "metrics.json"), metrics)
    text = summary_markdown(metrics)
    with open(os.path.join(args.run_dir, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    return 0 if metrics["verdict"] in ("pass", "no expectations") else 1


if __name__ == "__main__":
    sys.exit(main())
