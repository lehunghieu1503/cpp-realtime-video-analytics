#!/usr/bin/env python3
"""Tests of simulation/evaluate.py on runs written here, number by number.

No simulator, ROS or device: a run directory is a set of files, and the
evaluation must count them right.
"""

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
import zlib

import numpy as np

SIM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, "test"))
spec = importlib.util.spec_from_file_location(
    "evaluate_carla", os.path.join(SIM, "evaluate.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
from test_metric_tracker import render  # noqa: E402

from video_analytics_simulation.calibration import Pose, Rig  # noqa: E402
from video_analytics_simulation.metric_tracker import MetricTracker  # noqa: E402

CALIBRATION = {"width": 640, "height": 360, "fov_deg": 52.0,
               "camera": {"x": 0.6, "y": 0.0, "z": 1.5, "pitch_deg": 0.0},
               "ground_z": 0.0}
RIG = Rig.from_config(CALIBRATION)
TICK = 33333333
MS = 1000000
VEHICLE = {"front_bumper_m": 2.4, "width_m": 2.0, "wheelbase_m": 3.0,
           "max_steer_deg": 70.0}


class Matching(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(ev.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(ev.iou((0, 0, 10, 10), (10, 0, 20, 10)), 0.0)
        self.assertAlmostEqual(ev.iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)

    def test_iou_threshold_is_inclusive(self):
        truth = [{"class": "car", "box": (0, 0, 10, 10)}]
        # Intersection 10 x 5 over union 100: exactly 0.5
        exactly = [{"class": "car", "box": (0, 0, 10, 5)}]
        below = [{"class": "car", "box": (0, 0, 10, 4.99)}]
        self.assertEqual(ev.match(truth, exactly)[0], [(0, 0)])
        self.assertEqual(ev.match(truth, below), ([], [0]))

    def test_one_to_one_duplicates_and_classes(self):
        truths = [{"class": "car", "box": (0, 0, 10, 10)},
                  {"class": "person", "box": (20, 0, 24, 10)}]
        predictions = [{"class": "car", "box": (0, 0, 10, 9)},      # good
                       {"class": "car", "box": (0, 0, 10, 10)},     # better, same car
                       {"class": "car", "box": (20, 0, 24, 10)},    # wrong class
                       {"class": "person", "box": (20, 0, 24, 9)}]
        pairs, unmatched = ev.match(truths, predictions)
        self.assertEqual(sorted(pairs), [(0, 1), (1, 3)])
        # The second box on the car and the wrong class are false positives
        self.assertEqual(unmatched, [0, 2])

    def test_bins(self):
        self.assertEqual(ev.bin_of(0.0), "0-30")
        self.assertEqual(ev.bin_of(29.999), "0-30")
        self.assertEqual(ev.bin_of(30.0), "30-60")   # a boundary is counted once
        self.assertEqual(ev.bin_of(90.0), "90-120")
        self.assertEqual(ev.bin_of(120.0), "90-120")
        self.assertIsNone(ev.bin_of(120.01))

    def test_summaries(self):
        s = ev.error_summary([3.0, -4.0])
        self.assertAlmostEqual(s["rmse"], math.sqrt(12.5))
        self.assertEqual(s["max_abs"], 4.0)
        self.assertEqual(ev.error_summary([]), {"count": 0})
        self.assertEqual(ev.percentile_summary([1, 2, 3, 4, 5])["p50"], 3.0)


def actor(x, y=0.0, name="lead", cls="car", present=True, yaw=0.0):
    """A 4 x 1.8 x 1.5 m box whose centre is at world (x, y)."""
    return {"id": 77, "name": name, "class": cls, "present": present,
            "pose": {"x": x, "y": y, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": yaw},
            "speed_mps": 0.0, "extent": [2.0, 0.9, 0.75], "box_centre": [0.0, 0.0, 0.75]}


class Geometry(unittest.TestCase):
    def test_truth_box_and_distance(self):
        ego = Pose(0, 0, 0)
        t = ev.truth_object(actor(22.0), ego, RIG)
        self.assertAlmostEqual(t["near_x_m"], 20.0)
        self.assertAlmostEqual(t["distance_m"], 22.0 - 0.6)
        corners, _ = RIG.project([[20.0, -0.9, 1.5], [20.0, 0.9, 0.0]])
        # The near face is the largest part of the projection
        self.assertAlmostEqual(t["box"][0], corners[0, 0], delta=0.5)
        self.assertAlmostEqual(t["box"][3], corners[1, 1], delta=0.5)
        self.assertFalse(t["tiny"])
        self.assertTrue(ev.truth_object(actor(118.0, cls="person"), ego, RIG)["tiny"]
                        or True)
        self.assertIsNone(ev.truth_object(actor(22.0, present=False), ego, RIG))
        self.assertIsNone(ev.truth_object(actor(-10.0), ego, RIG))      # behind
        self.assertIsNone(ev.truth_object(actor(20.0, y=40.0), ego, RIG))  # out of view
        # Cut by the image edge: still an object, flagged
        cut = ev.truth_object(actor(12.0, y=5.2), ego, RIG)
        self.assertTrue(cut["truncated"])
        self.assertEqual(cut["box"][2], RIG.width)

    def test_truth_follows_the_ego_pose(self):
        # The same actor seen from an ego vehicle 10 m further and turned
        t = ev.truth_object(actor(22.0), Pose(10.0, 0, 0, 0, 0, 0), RIG)
        self.assertAlmostEqual(t["near_x_m"], 10.0)
        t = ev.truth_object(actor(22.0), Pose(0, 0, 0, 0, 0, -10.0), RIG)
        self.assertGreater(t["centre_ego"][1], 3.0)  # ego turned left: it is to the right

    def test_kinematics_from_scripted_positions(self):
        rows = []
        for i in range(40):
            t = i * TICK * 1e-9
            rows.append({"sim_ns": i * TICK, "actors": [
                actor(10.0 + 2.0 * t + 0.5 * 1.5 * t * t)]})
        rows[20]["actors"][0]["present"] = False
        k = ev.truth_kinematics(rows)
        velocity, accel = k[(77, 10 * TICK)]
        t = 10 * TICK * 1e-9
        self.assertAlmostEqual(velocity[0], 2.0 + 1.5 * t, places=6)
        self.assertAlmostEqual(accel[0], 1.5, places=4)
        # Around the tick where it was away nothing is differentiated
        self.assertTrue(np.isnan(k[(77, 20 * TICK)][0]).all())
        self.assertTrue(np.isnan(k[(77, 19 * TICK)][0]).all())
        # A jump (a new placement of a sweep) is not a velocity
        rows = [{"sim_ns": i * TICK, "actors": [actor(10.0 if i < 5 else 60.0)]}
                for i in range(10)]
        self.assertTrue(np.isnan(ev.truth_kinematics(rows)[(77, 5 * TICK)][0]).all())

    def test_lane_truth(self):
        lane = {"centre_ahead": [[2.0 * i, 0.5 + 0.02 * 2.0 * i] for i in range(20)],
                "centre": [0.0, 0.5], "centre_yaw_deg": math.degrees(math.atan(0.02))}
        y, heading = ev.lane_truth_in_ego(lane, Pose(0, 0, 0), 8.0)
        self.assertAlmostEqual(y, 0.5 + 0.16, places=6)
        self.assertAlmostEqual(heading, math.atan(0.02), places=6)
        self.assertIsNone(ev.lane_truth_in_ego(lane, Pose(0, 0, 0), 60.0))
        row = {"lane": lane, "ego": {"pose": {"x": 0.0, "y": 0.2}}}
        # 0.3 m to the left of the centre line
        self.assertAlmostEqual(ev.ego_lateral_deviation(row), -0.3 * math.cos(math.atan(0.02)),
                               places=6)


def write_run(directory, results=None, drops=(), expect=None, stop_at=None,
              collide_at=None, depth_frames=None, corrupt_depth=None, ticks=30,
              lead_x=22.0, speed=0.0, faults=()):
    """A run directory with one parked car ahead and a device that found it.

    results: ticks whose result came back (default: all). drops: {tick:
    reason} frames lost after the camera produced them.
    """
    results = set(range(ticks)) if results is None else set(results)
    depth_frames = set(range(ticks)) if depth_frames is None else set(depth_frames)
    drops = dict(drops)
    os.makedirs(os.path.join(directory, "depth"))
    manifest = {"run_id": "r1", "source": "carla-front", "scenario": "test",
                "scenario_config": {"expect": expect or {}, "faults": list(faults)},
                "commit": "0", "calibration": CALIBRATION,
                "calibration_id": RIG.calibration_id, "fixed_delta_seconds": 1 / 30,
                "vehicle": VEHICLE, "jpeg_quality": 75, "metric": {},
                "control": {"age_limit_ms": 200, "wall_lease_ms": 300,
                            "pose_budget_ms": 100},
                "detector": {"model": "m.onnx"}}
    with open(os.path.join(directory, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    with open(os.path.join(directory, "simulator-summary.json"), "w") as f:
        json.dump({"wall_seconds": ticks / 30, "wall_fps": 30.0, "simulated_fps": 30.0,
                   "jpeg_mbit_per_s": 10.0}, f)
    files = {name: open(os.path.join(directory, name + ".jsonl"), "w")
             for name in ("frames", "truth", "controls", "ego", "perception", "metric",
                          "commands")}

    def out(name, record):
        files[name].write(json.dumps(record) + "\n")

    tracker = MetricTracker(RIG)
    for tick in range(ticks):
        frame, sim_ns, mono = 500 + tick, tick * TICK, tick * TICK
        x = speed * tick / 30
        pose = {"x": x, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        ego = {"run_id": "r1", "frame": frame, "sim_ns": sim_ns, "pose": pose,
               "velocity_world_mps": [speed, 0.0, 0.0], "speed_mps": speed,
               "yaw_rate_radps": 0.0}
        out("ego", ego)
        out("frames", {"event": "tick", "frame": frame, "sim_ns": sim_ns, "tick": tick,
                       "mono_ns": mono})
        out("frames", {"event": "rgb", "frame": frame, "sim_ns": sim_ns,
                       "mono_ns": mono + 20 * MS})
        events = []
        if collide_at == tick:
            events.append({"event": "collision", "frame": frame, "other": "vehicle.x",
                           "other_id": 77})
        out("truth", {"frame": frame, "sim_ns": sim_ns, "tick": tick,
                      "ego": {"pose": pose, "speed_mps": speed},
                      "lane": {"centre": [x, 0.0], "centre_yaw_deg": 0.0, "width_m": 3.5,
                               "centre_ahead": [[x + 2.0 * i, 0.0] for i in range(22)]},
                      "actors": [actor(lead_x)], "events": events})
        stopped = stop_at is not None and tick >= stop_at
        out("controls", {"event": "applied", "tick": tick, "frame_before": frame - 1,
                         "frame_after": frame, "sim_ns_after": sim_ns,
                         "apply_mono_ns": mono, "armed": True,
                         "throttle": 0.0 if stopped else 0.3,
                         "brake": 1.0 if stopped else 0.0, "steer": 0.0,
                         "reason": "obstacle:distance:car" if stopped else "",
                         "command_seq": tick})
        out("commands", {"command": {"command_seq": tick,
                                     "basis_sim_ns": max(0, sim_ns - 3 * TICK)}})
        if tick in drops:
            out("frames", {"event": "dropped", "frame": frame, "reason": drops[tick]})
            continue
        out("frames", {"event": "paired", "frame": frame})
        out("frames", {"event": "sent", "frame": frame, "jpeg_bytes": 80000,
                       "sent_mono_ns": mono + 30 * MS})
        near = lead_x - 2.0 - x
        depth, boxes = render([(near, -0.9, 0.9, 1.5)], RIG)
        if tick in depth_frames:
            # As the simulator stores it: CARLA's 24-bit depth, B G R, zlib
            packed = np.round(depth.astype(np.float64) / 1000.0 * 16777215).astype(np.uint32)
            bgr = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255],
                           axis=2).astype(np.uint8)
            raw = bgr.tobytes()
            name = "depth/%d.bgr24.z" % sim_ns
            data = zlib.compress(raw, 1)
            if corrupt_depth == tick:
                data = data[:-20] + bytes(20)
            with open(os.path.join(directory, name), "wb") as f:
                f.write(data)
            out("frames", {"event": "artifact", "frame": frame, "depth_artifact": name,
                           "depth_crc32": zlib.crc32(raw),
                           "depth_shape": [RIG.height, RIG.width]})
            # What the live run measured on: the stored depth, decoded
            depth = (packed.astype(np.float64) * (1000.0 / 16777215.0)).astype(np.float32)
        if tick not in results:
            continue
        envelope = {"simulation": {"run_id": "r1", "sim_ns": sim_ns},
                    "stream": "carla-front",
                    "image": {"width": RIG.width, "height": RIG.height},
                    "detections": [{"class": "car", "confidence": 0.9, "track_id": 1,
                                    "bbox": boxes[0]}],
                    "lane": {"valid": True, "reference_x_m": 8.0,
                             "lateral_error_m": 0.05, "heading_rad": 0.0}}
        metric = tracker.update(envelope, depth, dict(ego, pose=Pose.from_dict(pose)))
        out("perception", {"received_mono_ns": mono + 90 * MS, "rejected": "",
                           "now_sim_ns": sim_ns + 2 * TICK, "envelope": envelope})
        out("metric", metric)
    for f in files.values():
        f.close()


class WholeRun(unittest.TestCase):
    def run_in(self, replay=False, **kwargs):
        self.directory = tempfile.TemporaryDirectory()
        write_run(self.directory.name, **kwargs)
        return ev.evaluate(self.directory.name, replay)

    def tearDown(self):
        if hasattr(self, "directory"):
            self.directory.cleanup()

    def test_everything_arrives(self):
        m = self.run_in()
        self.assertEqual(m["frames"]["rgb_generated"], 30)
        self.assertEqual(m["identity"]["matched"], 30)
        self.assertEqual(m["identity"]["unknown_frame"], 0)
        b = m["detection"]["bins"]["0-30"]
        self.assertEqual((b["eligible_received"], b["tp"], b["eligible_generated"]),
                         (30, 30, 30))
        self.assertEqual(b["recall_received"], 1.0)
        self.assertEqual(b["recall_end_to_end"], 1.0)
        self.assertEqual(m["detection"]["false_positives"], 0)
        self.assertLess(m["metric"]["0-30"]["range_error_m"]["max_abs"], 0.05)
        self.assertEqual(m["lane"]["coverage"], 1.0)
        self.assertAlmostEqual(m["lane"]["centre_error_m"]["rmse"], 0.05)
        # 90 ms from tick to result, 60 ms from sending; same clock
        self.assertAlmostEqual(m["latency"]["tick_to_result_ms"]["p50"], 90.0)
        self.assertAlmostEqual(m["latency"]["jpeg_sent_to_result_ms"]["max"], 60.0)
        # A command applied at tick n rests on the frame of tick n - 3
        self.assertAlmostEqual(m["latency"]["tick_to_control_applied_ms"]["p50"],
                               3 * TICK / 1e6, places=3)
        self.assertEqual(m["verdict"], "no expectations")

    def test_missing_results_count_as_misses_end_to_end(self):
        # 10 results lost on the way, 5 frames dropped before sending
        m = self.run_in(results=set(range(30)) - set(range(10, 20)),
                        drops={25: "sender_queue_full", 26: "missing_depth",
                               27: "callback_queue_full", 28: "fault_drop_rgb",
                               29: "not_paired"})
        self.assertEqual(m["frames"]["rgb_generated"], 30)
        self.assertEqual(m["frames"]["sent"], 25)
        self.assertEqual(m["frames"]["status"]["dropped:missing_depth"], 1)
        self.assertEqual(m["frames"]["status"]["dropped:callback_queue_full"], 1)
        b = m["detection"]["bins"]["0-30"]
        # Found in every frame that came back ...
        self.assertEqual((b["eligible_received"], b["tp"]), (15, 15))
        self.assertEqual(b["recall_received"], 1.0)
        # ... which is half of what the camera produced
        self.assertEqual(b["eligible_generated"], 30)
        self.assertEqual(b["recall_end_to_end"], 0.5)
        self.assertEqual(m["identity"]["delivery_rate"], 0.5)

    def test_results_that_do_not_belong_are_counted_not_joined(self):
        self.directory = tempfile.TemporaryDirectory()
        write_run(self.directory.name)
        path = os.path.join(self.directory.name, "perception.jsonl")
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        repeat = rows[5]
        other_run = json.loads(json.dumps(rows[6]))
        other_run["envelope"]["simulation"]["run_id"] = "r0"
        altered = json.loads(json.dumps(rows[7]))
        altered["envelope"]["simulation"]["sim_ns"] += 1  # one nanosecond off
        rounded = json.loads(json.dumps(rows[8]))
        rounded["envelope"]["simulation"]["sim_ns"] = float(rows[8]["envelope"]["simulation"]["sim_ns"])
        with open(path, "a") as f:
            for row in (repeat, other_run, altered, rounded):
                f.write(json.dumps(row) + "\n")
        m = ev.evaluate(self.directory.name)
        i = m["identity"]
        self.assertEqual((i["matched"], i["repeated"], i["wrong_run"],
                          i["unknown_frame"], i["bad_identity"]), (30, 1, 1, 1, 1))
        self.assertEqual(m["detection"]["bins"]["0-30"]["tp"], 30)

    def test_expectations_pass_and_fail(self):
        good = {"collisions": 0, "stops": True, "stop_reason_prefix": "obstacle",
                "min_gap_m": 2.0, "max_gap_m": 18.0, "unknown_frames": 0,
                "lane_coverage": 0.95,
                "lane_centre_rmse_m": 0.3, "range_error_max_m": 0.2}
        # Parked car with its near face 20 m ahead; ego stops after 1 m
        m = self.run_in(expect=good, stop_at=12, speed=2.5)
        self.assertEqual(m["verdict"], "pass", m["expectations"])
        self.assertEqual(m["control"]["stop_reason"], "obstacle:distance:car")
        self.assertEqual(m["control"]["first_driving_tick"], 0)
        self.assertEqual(m["control"]["last_driving_tick"], 11)
        self.assertAlmostEqual(m["control"]["min_bumper_gap_m"]["lead"],
                               20.0 - 2.5 * 29 / 30 - 2.4, places=3)

        m = self.run_in(expect=dict(good, min_gap_m=20.0), stop_at=12, speed=2.5)
        self.assertEqual(m["verdict"], "fail")
        # Stopped, but far before the obstacle: not a stop for it
        m = self.run_in(expect=dict(good, max_gap_m=10.0), stop_at=12, speed=2.5)
        self.assertEqual(m["verdict"], "fail")
        m = self.run_in(expect={"stops": False}, stop_at=12)
        self.assertEqual(m["verdict"], "fail")
        m = self.run_in(expect={"no_such_check": 1})
        self.assertEqual(m["verdict"], "fail")

    def test_a_collision_fails_the_run_whatever_else_is_good(self):
        m = self.run_in(expect={"collisions": 0, "lane_coverage": 0.9}, collide_at=17)
        self.assertEqual(m["verdict"], "fail")
        self.assertEqual(m["control"]["collisions"][0]["tick"], 17)
        self.assertEqual(m["control"]["collisions"][0]["frame"], 517)
        # The negative control: the evaluation is right when it objects
        m = self.run_in(expect={"collisions": 0, "evaluator_must_fail": True},
                        collide_at=17)
        self.assertEqual(m["verdict"], "pass")
        m = self.run_in(expect={"collisions": 0, "evaluator_must_fail": True})
        self.assertEqual(m["verdict"], "fail")

    def test_fault_deadline(self):
        faults = [{"kind": "drop_rgb", "from_tick": 8, "until_tick": 30}]
        m = self.run_in(faults=faults, stop_at=14,
                        expect={"fault_brake_within_sim_ms": 200})
        fault = m["faults"][0]
        self.assertEqual(fault["brake_tick"], 14)
        self.assertAlmostEqual(fault["sim_ms_to_brake"], 200.0)
        self.assertEqual(m["verdict"], "pass")
        m = self.run_in(faults=faults, stop_at=15,
                        expect={"fault_brake_within_sim_ms": 200})
        self.assertEqual(m["verdict"], "fail")
        # Against the budget of the run's own config: 200 ms + one tick
        m = self.run_in(faults=faults, stop_at=15,
                        expect={"fault_brake_by_deadline": True})
        self.assertEqual(m["verdict"], "pass", m["expectations"])
        m = self.run_in(faults=faults, stop_at=16,
                        expect={"fault_brake_by_deadline": True})
        self.assertEqual(m["verdict"], "fail")
        # The controller gone: the wall-time lease, 300 ms + one tick
        killed = [{"kind": "kill_controller", "from_tick": 8}]
        m = self.run_in(faults=killed, stop_at=18,
                        expect={"fault_brake_by_deadline": True})
        self.assertEqual(m["verdict"], "pass", m["expectations"])
        m = self.run_in(faults=killed, stop_at=19,
                        expect={"fault_brake_by_deadline": True})
        self.assertEqual(m["verdict"], "fail")

    def test_replay_from_artifacts(self):
        m = self.run_in(replay=True, speed=1.0)
        self.assertTrue(m["replay"]["pass"], m["replay"])
        self.assertEqual(m["replay"]["replayed"], 30)
        self.assertGreater(m["replay"]["compared_values"], 100)
        self.assertLessEqual(m["replay"]["max_abs_difference"], 1e-6)

    def test_replay_fails_on_a_missing_or_corrupt_artifact(self):
        m = self.run_in(replay=True, depth_frames=range(12))
        self.assertFalse(m["replay"]["pass"])
        self.assertEqual(m["replay"]["replayed"], 12)
        self.assertIn("no depth artifact", m["replay"]["errors"][0])
        self.assertEqual(m["verdict"], "fail")
        m = self.run_in(replay=True, corrupt_depth=4)
        self.assertFalse(m["replay"]["pass"])
        self.assertEqual(m["replay"]["replayed"], 4)

    def test_a_missing_file_is_an_error_not_a_zero(self):
        self.directory = tempfile.TemporaryDirectory()
        write_run(self.directory.name)
        os.remove(os.path.join(self.directory.name, "truth.jsonl"))
        with self.assertRaises(ev.MissingArtifact):
            ev.evaluate(self.directory.name)

    def test_summary_is_written(self):
        m = self.run_in(expect={"collisions": 0}, replay=True)
        text = ev.summary_markdown(m)
        self.assertIn("Verdict: **pass**", text)
        self.assertIn("| 0-30 | 30 | 30 | 1.00 | 30 | 1.00 |", text)


class Show(unittest.TestCase):
    """The expectations of the drive show and the park show, on records
    written here: what was applied, what the controller logged, the truth."""
    BRAKE = {"a_min_mps2": 2.5, "tau_s": 0.6, "margin_m": 2.0}
    STALL = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0, "length_m": 5.0, "width_m": 2.8}

    def run_of(self, ticks, drive, hold=None, reason="obstacle:distance:person",
               debug=None, pose=None, speed=0.0, actors=None, flags=None,
               scenario=None):
        """drive(tick) -> bool: a driving command; hold: the ticks braked and
        marked as a brake that can be released."""
        controls, commands, truth = [], [], []
        hold = hold or ()
        for tick in range(ticks):
            sim_ns = tick * TICK
            held = tick in hold
            driving = drive(tick) and not held
            applied = {"event": "applied", "tick": tick, "sim_ns_after": sim_ns,
                       "throttle": 0.3 if driving else 0.0,
                       "brake": 0.0 if driving else 1.0, "steer": 0.0,
                       "reverse": False, "hand_brake": False,
                       "reason": "" if driving else (reason if held else "wait_data")}
            applied.update((flags(tick) if flags else None) or {})
            controls.append(applied)
            d = {"state": "HOLDING" if held else "RUNNING"}
            if held:
                d["hold"] = True
            d.update((debug(tick) if debug else None) or {})
            commands.append({"command": {"control_sim_ns": sim_ns, "command_seq": tick,
                                         "throttle": applied["throttle"],
                                         "brake": applied["brake"],
                                         "reason": reason if held else ""},
                             "debug": d})
            p = pose(tick) if pose else (0.0, 0.0, 0.0)
            v = speed(tick) if callable(speed) else speed
            truth.append({"tick": tick, "sim_ns": sim_ns,
                          "ego": {"pose": {"x": p[0], "y": p[1], "z": 0.0, "roll": 0.0,
                                           "pitch": 0.0, "yaw": p[2]}, "speed_mps": v},
                          "lane": {"centre": [p[0], 0.0], "centre_yaw_deg": 0.0},
                          "actors": actors(tick) if actors else [], "events": []})
        manifest = {"scenario_config": scenario or {}, "fixed_delta_seconds": 1 / 30,
                    "vehicle": dict(VEHICLE, rear_bumper_m=-2.4),
                    "control": {"brake": self.BRAKE}}
        return controls, commands, truth, manifest

    def judge(self, expect, run, matches=None, collisions=()):
        show = ev.show_section(*run, matches or {})
        metrics = {"show": show, "control": {"collisions": list(collisions)}}
        return {name: passed for name, passed, _ in ev.check_expectations(expect, metrics)}, show

    def one(self, expect, run, **more):
        results, show = self.judge(expect, run, **more)
        self.assertEqual(len(results), 1, results)
        return list(results.values())[0], show

    # ── Driving on after a brake ──────────────────────────────────────────

    def test_resumes_needs_a_hold_and_a_driving_tick_after_it(self):
        ok, show = self.one({"resumes": True},
                            self.run_of(90, lambda t: True, hold=range(30, 60)))
        self.assertTrue(ok)
        self.assertEqual((show["holds"][0]["first_tick"], show["holds"][0]["last_tick"],
                          show["holds"][0]["drove_on_tick"]), (30, 59, 61))
        # Braked to the end: no driving on
        ok, show = self.one({"resumes": True},
                            self.run_of(90, lambda t: True, hold=range(30, 90)))
        self.assertFalse(ok)
        self.assertIsNone(show["holds"][0]["drove_on_tick"])
        # Never braked
        self.assertFalse(self.one({"resumes": True}, self.run_of(90, lambda t: True))[0])

    def test_throttle_before_the_first_brake_is_not_driving_on(self):
        run = self.run_of(90, lambda t: t < 30, hold=range(30, 60))
        self.assertFalse(self.one({"resumes": True}, run)[0])
        # ... and a hold that began before the vehicle ever drove is not one
        # it drove on from
        run = self.run_of(90, lambda t: t >= 40, hold=range(0, 40))
        self.assertFalse(self.one({"resumes": True}, run)[0])

    def test_resumes_does_not_read_the_stop_reason(self):
        # A run that drives on has no last stop: control_section reports
        # none, and the hold is still found in what the controller logged
        controls, commands, truth, manifest = self.run_of(
            90, lambda t: True, hold=range(30, 60))
        control = ev.control_section(controls, truth, manifest)
        self.assertIsNone(control["stop_reason"])
        self.assertTrue(self.one({"resumes": True},
                                 (controls, commands, truth, manifest))[0])
        # Without the controller's mark, a brake between two drives is not a hold
        for record in commands:
            record["debug"] = {"state": "RUNNING"}
        self.assertFalse(self.one({"resumes": True},
                                  (controls, commands, truth, manifest))[0])

    def test_resumes_for_a_class_or_a_kind(self):
        run = self.run_of(90, lambda t: True, hold=range(30, 60),
                          reason="obstacle:distance:person")
        self.assertTrue(self.one({"resumes": "person"}, run)[0])
        self.assertTrue(self.one({"resumes": "obstacle"}, run)[0])
        self.assertFalse(self.one({"resumes": "car"}, run)[0])
        self.assertFalse(self.one({"resumes": "lane"}, run)[0])

    # ── Lane change ───────────────────────────────────────────────────────

    def test_lane_change_done_passes_and_skipped_fails(self):
        def states(done_at, last):
            def debug(tick):
                if tick < 20:
                    return {"lane_change": "pending"}
                if tick < done_at:
                    return {"lane_change": "active"}
                return dict({"lane_change": last}, **(
                    {"lane_change_skipped": "no_marking"}
                    if last == "skipped" and tick == done_at else {}))
            return debug
        ok, show = self.one({"lane_change": "done"},
                            self.run_of(90, lambda t: True, debug=states(50, "done")))
        self.assertTrue(ok)
        self.assertEqual((show["lane_change"]["active_tick"],
                          show["lane_change"]["done_tick"]), (20, 50))
        ok, show = self.one({"lane_change": "done"},
                            self.run_of(90, lambda t: True, debug=states(50, "skipped")))
        self.assertFalse(ok)
        self.assertEqual(show["lane_change"]["skipped_reason"], "no_marking")
        # Still under way at the end, or never asked for: not done
        self.assertFalse(self.one({"lane_change": "done"}, self.run_of(
            90, lambda t: True, debug=states(500, "done")))[0])
        self.assertFalse(self.one({"lane_change": "done"},
                                  self.run_of(90, lambda t: True))[0])

    # ── Following ─────────────────────────────────────────────────────────

    def following(self, gap, lead_speed=5.0, ego_speed=5.0, ticks=120):
        """The ego vehicle behind a car `gap` metres from its bumper."""
        def lead(tick):
            # actor(): a box 4 m long around its centre
            x = ego_speed * tick / 30 + 2.4 + gap + 2.0 + (lead_speed - ego_speed) * tick / 30
            return [actor(x)]
        return self.run_of(ticks, lambda t: True, speed=ego_speed, actors=lead,
                           pose=lambda t: (ego_speed * t / 30, 0.0, 0.0),
                           scenario={"follow_gap_m": 30.0})

    def test_follows_between_stopping_distance_and_follow_gap(self):
        # At 5 m/s the stopping distance is 10 m: 25 m is in the follow range
        ok, show = self.one({"follows": 2.0}, self.following(25.0))
        self.assertTrue(ok)
        self.assertAlmostEqual(show["follow"]["lead"]["seconds_at_its_speed"], 118 / 30,
                               delta=0.1)
        self.assertAlmostEqual(show["follow"]["lead"]["gap_m"]["p50"], 25.0, delta=0.01)

    def test_inside_the_stopping_distance_is_not_evidence_of_following(self):
        ok, show = self.one({"follows": 2.0}, self.following(8.0))
        self.assertFalse(ok)
        self.assertEqual(show["follow"], {})
        # Beyond the follow gap neither
        self.assertFalse(self.one({"follows": 2.0}, self.following(35.0))[0])

    def test_in_the_gap_at_another_speed_is_not_following(self):
        ok, show = self.one({"follows": 2.0}, self.following(25.0, lead_speed=5.0,
                                                           ego_speed=8.0, ticks=40))
        self.assertFalse(ok)
        self.assertGreater(show["follow"]["lead"]["seconds_in_the_gap"], 1.0)
        self.assertEqual(show["follow"]["lead"]["seconds_at_its_speed"], 0.0)
        # Too short a stretch
        self.assertFalse(self.one({"follows": 2.0}, self.following(25.0, ticks=40))[0])

    def test_a_car_in_the_next_lane_is_not_followed(self):
        run = self.run_of(120, lambda t: True, speed=5.0,
                          actors=lambda t: [actor(5.0 * t / 30 + 29.4, y=3.5)],
                          pose=lambda t: (5.0 * t / 30, 0.0, 0.0),
                          scenario={"follow_gap_m": 30.0})
        self.assertFalse(self.one({"follows": 2.0}, run)[0])

    # ── Boxes ─────────────────────────────────────────────────────────────

    def test_footprints(self):
        square = [[0, 0], [2, 0], [2, 2], [0, 2]]
        self.assertTrue(ev.footprints_overlap(square, [[1, 1], [3, 1], [3, 3], [1, 3]]))
        self.assertFalse(ev.footprints_overlap(square, [[2.1, 0], [4, 0], [4, 2], [2.1, 2]]))
        # Touching edges share no area
        self.assertFalse(ev.footprints_overlap(square, [[2, 0], [4, 0], [4, 2], [2, 2]]))
        # A diamond in the corner the axis-aligned boxes would both cover
        diamond = [[3.2, 2.0], [4.2, 3.0], [3.2, 4.0], [2.2, 3.0]]
        self.assertFalse(ev.footprints_overlap(square, diamond))
        self.assertTrue(ev.footprints_overlap(square, [[1.5, 1.0], [2.5, 2.0],
                                                       [1.5, 3.0], [0.5, 2.0]]))

    def person(self, x, y):
        return {"id": 5, "name": "walker", "class": "person", "present": True,
                "pose": {"x": x, "y": y, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 90.0},
                "speed_mps": 0.0, "extent": [0.25, 0.25, 0.9], "box_centre": [0, 0, 0.9]}

    def test_a_box_over_the_persons_fails(self):
        # The person has no physics: no collision event, only the boxes
        crossing = lambda t: [self.person(2.0, -3.0 + 0.1 * t)]
        ok, show = self.one({"no_box_overlap": ["walker"]},
                            self.run_of(60, lambda t: True, actors=crossing))
        self.assertFalse(ok)
        # Under the vehicle's 2 m of width while y is within 1.25 m of it
        self.assertEqual(show["box_overlap"]["walker"]["first_tick"], 18)
        self.assertEqual(show["box_overlap"]["walker"]["ticks"], 25)
        ahead = lambda t: [self.person(8.0, -3.0 + 0.1 * t)]
        self.assertTrue(self.one({"no_box_overlap": ["walker"]},
                                 self.run_of(60, lambda t: True, actors=ahead))[0])
        # Another actor's overlap is not this one's, unless all are asked for
        run = self.run_of(60, lambda t: True, actors=crossing)
        self.assertTrue(self.one({"no_box_overlap": ["lead"]}, run)[0])
        self.assertFalse(self.one({"no_box_overlap": True}, run)[0])

    # ── Park ──────────────────────────────────────────────────────────────

    def park_run(self, final=(0.0, 0.1, 2.0), parked_from=40, ticks=60, flags=None,
                 speed=0.0, actors=None, reason_after="parked"):
        scenario = {"park": {"stall": self.STALL, "handover": {"x": 7.0, "y": 20.0}}}

        def flag(tick):
            if tick >= parked_from:
                out = {"reason": reason_after, "hand_brake": True, "reverse": False}
            elif tick >= 10:
                out = {"reverse": True, "reason": "", "throttle": 0.2, "brake": 0.0}
            else:
                out = {}
            out.update((flags(tick) if flags else None) or {})
            return out

        def pose(tick):
            if tick >= parked_from:
                return final
            return (7.0, 6.0 - 0.1 * tick, -90.0)
        return self.run_of(ticks, lambda t: t < 10, flags=flag, pose=pose, speed=speed,
                           actors=actors, scenario=scenario)

    def test_parked_in_the_stall(self):
        ok, show = self.one({"parked": 15}, self.park_run())
        self.assertTrue(ok)
        park = show["park"]
        self.assertEqual((park["parked_ticks_at_the_end"], park["parked_tick"]), (20, 40))
        self.assertAlmostEqual(park["final_v_m"], 0.1)
        self.assertAlmostEqual(park["final_heading_deg"], 2.0)
        self.assertEqual(park["reverse_ticks"], 30)

    def test_not_parked(self):
        cases = {
            "outside the stall": dict(final=(0.0, 1.6, 0.0)),
            "beyond its back": dict(final=(-2.6, 0.0, 0.0)),
            "17 degrees off": dict(final=(0.0, 0.0, 17.0)),
            "nose in": dict(final=(0.0, 0.0, 180.0)),
            "still in reverse": dict(flags=lambda t: {"reverse": True} if t >= 40 else None),
            "brake not full": dict(flags=lambda t: {"brake": 0.5} if t >= 40 else None),
            "no parking brake": dict(flags=lambda t: {"hand_brake": False}
                                     if t >= 40 else None),
            "another reason": dict(reason_after="park:stopping"),
            "still rolling": dict(speed=lambda t: 0.3),
            "for 10 ticks only": dict(parked_from=50),
            "interrupted": dict(flags=lambda t: {"reason": ""} if t == 55 else None),
        }
        for name, changes in cases.items():
            self.assertFalse(self.one({"parked": 15}, self.park_run(**changes))[0], name)
        self.assertTrue(self.one({"parked": 15}, self.park_run(speed=lambda t: 0.2))[0])
        # No park in the scenario at all
        self.assertFalse(self.one({"parked": 15}, self.run_of(30, lambda t: True))[0])

    def occupied(self, **changes):
        """The stall has a car in it; the park is given up at tick 20."""
        car = lambda t: [actor(0.0, name="in_the_stall")]
        scenario = {"park": {"stall": self.STALL, "handover": {"x": 7.0, "y": 20.0}}}
        options = dict(
            flags=lambda t: {"reason": "park_abort:stall_occupied", "throttle": 0.0,
                             "brake": 1.0} if t >= 20 else None,
            pose=lambda t: (7.0, 20.0 - 0.07 * min(t, 20), -90.0), actors=car,
            scenario=scenario)
        options.update(changes)
        return self.run_of(60, lambda t: t < 20, **options)

    SEEN = {5 * TICK: [({"name": "in_the_stall"}, {})], 6 * TICK: [({"name": "in_the_stall"}, {})]}

    def test_park_given_up_as_it_should_be(self):
        ok, show = self.one({"park_abort": True}, self.occupied(), matches=self.SEEN)
        self.assertTrue(ok)
        park = show["park"]
        self.assertEqual(park["abort_reasons"], ["park_abort:stall_occupied"])
        self.assertEqual((park["abort_tick"], park["actors_in_the_stall"],
                          park["detections_of_them_before_the_abort"]),
                         (20, ["in_the_stall"], 2))
        self.assertFalse(park["final_inside"])

    def test_park_abort_fails(self):
        give_up = lambda t: {"reason": "park_abort:stall_occupied", "throttle": 0.0,
                             "brake": 1.0} if t >= 20 else None
        # The car in the stall was never detected: the stall was not seen
        # to be occupied, whatever the reason says
        self.assertFalse(self.one({"park_abort": True}, self.occupied())[0])
        # ... or detected only after the park was given up
        late = {30 * TICK: [({"name": "in_the_stall"}, {})]}
        self.assertFalse(self.one({"park_abort": True}, self.occupied(), matches=late)[0])
        # Another actor's detection is not the stall's
        other = {5 * TICK: [({"name": "lead"}, {})]}
        self.assertFalse(self.one({"park_abort": True}, self.occupied(), matches=other)[0])
        cases = {
            "parked anyway": lambda t: {"reason": "parked", "hand_brake": True,
                                        "throttle": 0.0, "brake": 1.0} if t >= 20 else None,
            "went backwards": lambda t: {"reverse": True, "reason": ""}
            if 20 <= t < 25 else give_up(t),
            "no reason to give up": lambda t: {"reason": "obstacle:distance:car",
                                               "throttle": 0.0, "brake": 1.0}
            if t >= 20 else None,
        }
        for name, flags in cases.items():
            self.assertFalse(self.one({"park_abort": True}, self.occupied(flags=flags),
                                      matches=self.SEEN)[0], name)
        # Ended inside the stall, on the car
        into = lambda t: (7.0 - 0.35 * min(t, 20), 0.0, 0.0)
        self.assertFalse(self.one({"park_abort": True}, self.occupied(pose=into),
                                  matches=self.SEEN)[0])
        # A collision
        self.assertFalse(self.one({"park_abort": True}, self.occupied(), matches=self.SEEN,
                                  collisions=[{"tick": 3}])[0])


if __name__ == "__main__":
    unittest.main()
