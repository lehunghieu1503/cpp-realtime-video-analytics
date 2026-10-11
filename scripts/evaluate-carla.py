#!/usr/bin/env python3
"""Evaluate one CARLA run from the files in its run directory.

Usage: scripts/evaluate-carla.py RUN_DIR [--replay]

Reads what the simulator and the controller wrote (see
ros2/video_analytics_simulation/video_analytics_simulation/simulator.py) and
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "ros2", "video_analytics_simulation"))
from video_analytics_simulation import protocol  # noqa: E402
from video_analytics_simulation.calibration import Pose, Rig, decode_depth  # noqa: E402
from video_analytics_simulation.metric_tracker import MetricTracker  # noqa: E402

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
    if not values:
        return {"count": 0}
    values = np.asarray(values, dtype=float)
    return {"count": int(len(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max()),
            "mean": float(values.mean())}


def error_summary(errors):
    if not errors:
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
    run_ticks = {f for f, t in ticks.items() if "tick" in t}
    # Images of the ticks before the run (the vehicle settling) are not its
    # frames, even when one of them arrives after the run began
    generated = {f: e for f, e in frames.items() if e["registered"] and f in run_ticks}
    statuses = {}
    for entry in generated.values():
        key = entry["status"] if entry["status"] != "dropped" else \
            "dropped:" + entry.get("reason", "?")
        statuses[key] = statuses.get(key, 0) + 1
    return generated, {
        "ticks": len(run_ticks),
        "rgb_generated": len(generated),
        # The tick happened and the camera gave no image: a loss of its own
        "rgb_never_generated": len(run_ticks - set(generated)),
        "status": statuses,
        "sent": statuses.get("sent", 0),
        "jpeg_bytes": percentile_summary([e["jpeg_bytes"] for e in generated.values()
                                          if e["status"] == "sent"]),
    }


def identity_section(perception, generated, run_id):
    """Every result received must name a frame this run sent, once."""
    by_ns = {e["sim_ns"]: f for f, e in generated.items() if e["status"] == "sent"}
    seen = set()
    counts = {"received": len(perception), "matched": 0, "wrong_run": 0,
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
        entry = artifact.get(sim_ns, {})
        path = entry.get("depth_artifact")
        if not path:
            result["errors"].append("%d: no depth artifact (%s)" % (
                sim_ns, entry.get("depth_artifact_error", "beyond the run's budget")))
            break  # the filter's state depends on every frame before
        try:
            with open(os.path.join(run_dir, path), "rb") as f:
                raw = zlib.decompress(f.read())
        except (OSError, zlib.error) as error:
            result["errors"].append("%d: %s" % (sim_ns, error))
            break
        if zlib.crc32(raw) != entry["depth_crc32"]:
            result["errors"].append("%d: depth checksum differs" % sim_ns)
            break
        h, w = entry["depth_shape"]
        bgr = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        depth = decode_depth(np.concatenate(
            [bgr, np.full((h, w, 1), 255, np.uint8)], axis=2))
        ego = ego_by_ns.get(sim_ns)
        if ego is None:
            result["errors"].append("%d: no ego state" % sim_ns)
            break
        again = tracker.update(envelope, depth, dict(ego, pose=Pose.from_dict(ego["pose"])))
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
                    limit = max(value, 0.02 * float(name.split("-")[1]))
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
        elif key == "evaluator_must_fail":
            pass  # handled by the caller: a negative control
        else:
            add("unknown expectation %s" % key, False, "not implemented")
    return results


def evaluate(run_dir, do_replay=False):
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise MissingArtifact(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    rig = Rig.from_config(manifest["calibration"])
    frames, ticks = protocol.read_ledger(os.path.join(run_dir, "frames.jsonl"))
    truth_rows = read_jsonl(os.path.join(run_dir, "truth.jsonl"))
    controls = read_jsonl(os.path.join(run_dir, "controls.jsonl"))
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
    generated, frame_metrics = frames_section(frames, ticks)
    usable, identity = identity_section(perception, generated, manifest["run_id"])
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
        "faults": faults, "commands_rejected_by_simulator": rejected,
        "rates": {
            "simulated_fps": simulator.get("simulated_fps"),
            "wall_fps": simulator.get("wall_fps"),
            "results_per_wall_second": (identity["matched"] / simulator["wall_seconds"])
            if simulator.get("wall_seconds") else None,
            "jpeg_mbit_per_s": simulator.get("jpeg_mbit_per_s"),
            "real_time_30_fps": bool(simulator.get("wall_fps", 0) >= 29.5),
        },
        "simulator": simulator,
    }
    if do_replay:
        metrics["replay"] = replay(run_dir, manifest, perception, ego_rows, generated,
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
        "- results received: %d; joined to a sent frame: %d; naming an unknown "
        "frame: %d; repeated: %d; wrong run: %d" % (
            i["received"], i["matched"], i["unknown_frame"], i["repeated"],
            i["wrong_run"]),
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
    for fault in m["faults"]:
        lines.append("- fault %s at tick %d: braked at tick %s (%s), %s ms of "
                     "simulation, %s ms of wall time" % (
                         fault["kind"], fault["from_tick"], fault.get("brake_tick"),
                         fault.get("reason"), fmt(fault.get("sim_ms_to_brake"), 0),
                         fmt(fault.get("wall_ms_to_brake"), 0)))
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
    args = parser.parse_args()
    try:
        metrics = evaluate(args.run_dir, args.replay)
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
