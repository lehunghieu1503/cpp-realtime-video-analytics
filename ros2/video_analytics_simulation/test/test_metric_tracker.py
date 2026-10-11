import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from video_analytics_simulation.calibration import Pose, Rig  # noqa: E402
from video_analytics_simulation.metric_tracker import MetricTracker, measure_box  # noqa: E402

RIG = Rig(640, 360, 52.0, 1.5, 0.0, 1.6)
DT_NS = 33333333


def render(objects, rig=RIG):
    """Depth image of a flat road with upright rectangles on it.

    objects: (x, y_left, y_right, height) in the ego frame, facing the camera.
    Returns the image and the pixel box of each object.
    """
    v, u = np.mgrid[0:rig.height, 0:rig.width]
    rays = rig.rays_ego(u + 0.5, v + 0.5)
    with np.errstate(divide="ignore"):
        t = (rig.ground_z - rig.z) / rays[..., 2]
    depth = np.where((rays[..., 2] < 0) & (t < 1000), t, 1000.0)
    boxes = []
    for x, left, right, height in objects:
        t = (x - rig.x) / rays[..., 0]
        hit = rays * t[..., None] + rig.pose.translation
        inside = ((hit[..., 1] >= left) & (hit[..., 1] <= right)
                  & (hit[..., 2] >= 0) & (hit[..., 2] <= height) & (t < depth))
        depth = np.where(inside, t, depth)
        corners, _ = rig.project([[x, left, height], [x, right, 0.0]])
        boxes.append({"x": float(corners[0, 0]), "y": float(corners[0, 1]),
                      "w": float(corners[1, 0] - corners[0, 0]),
                      "h": float(corners[1, 1] - corners[0, 1])})
    return depth.astype(np.float32), boxes


def ego_state(sim_ns, pose, velocity=(0.0, 0.0), yaw_rate=0.0, run="r1"):
    return {"run_id": run, "sim_ns": sim_ns, "pose": pose,
            "velocity_world_mps": [velocity[0], velocity[1], 0.0],
            "yaw_rate_radps": yaw_rate}


def perception(sim_ns, boxes, ids, run="r1", name="car"):
    detections = []
    for box, track_id in zip(boxes, ids):
        d = {"class": name, "bbox": box}
        if track_id is not None:
            d["track_id"] = track_id
        detections.append(d)
    return {"simulation": {"run_id": run, "sim_ns": sim_ns}, "stream": "front",
            "image": {"width": RIG.width, "height": RIG.height},
            "detections": detections}


def drive(tracker, steps, ego_at, object_at, skip=()):
    """Runs the tracker over a scripted scene; returns the object's observations.

    ego_at(t) -> (Pose, world velocity, yaw rate); object_at(t) -> world x, y.
    """
    out = []
    for i in range(steps):
        if i in skip:
            continue
        t = i * DT_NS * 1e-9
        pose, velocity, yaw_rate = ego_at(t)
        centre = pose.from_parent([*object_at(t), 0.0])
        depth, boxes = render([(centre[0], centre[1] - 0.9, centre[1] + 0.9, 1.5)])
        result = tracker.update(perception(i * DT_NS, boxes, [7]), depth,
                                ego_state(i * DT_NS, pose, velocity, yaw_rate))
        out.append(result["observations"][0])
    return out


class Measurement(unittest.TestCase):
    def test_range_of_a_car(self):
        depth, boxes = render([(21.5, -0.9, 0.9, 1.5)])
        # The detector's box is looser than the object: road and background in it
        box = {"x": boxes[0]["x"] - 8, "y": boxes[0]["y"] - 6,
               "w": boxes[0]["w"] + 16, "h": boxes[0]["h"] + 12}
        m = measure_box(box, depth, RIG)
        self.assertTrue(m["valid"])
        self.assertAlmostEqual(m["near_x_m"], 21.5, delta=0.05)
        self.assertAlmostEqual(m["position_ego_m"][0], 21.5, delta=0.05)
        self.assertAlmostEqual(m["position_ego_m"][1], 0.0, delta=0.1)
        self.assertLess(m["lateral_extent_m"][0], -0.5)
        self.assertGreater(m["lateral_extent_m"][1], 0.5)

    def test_off_axis_and_behind_a_thin_pole(self):
        # A pole 6 m ahead covers a few pixels of the car's box at 15 m
        depth, boxes = render([(16.5, 2.0, 3.8, 1.5), (7.5, 2.85, 2.9, 1.5)])
        m = measure_box(boxes[0], depth, RIG)
        self.assertTrue(m["valid"])
        self.assertAlmostEqual(m["near_x_m"], 16.5, delta=0.1)
        self.assertAlmostEqual(m["position_ego_m"][1], 2.9, delta=0.3)

    def test_nearer_object_with_support_wins(self):
        depth, boxes = render([(31.5, -1.0, 1.0, 1.6), (11.5, -0.3, 1.0, 1.2)])
        m = measure_box(boxes[0], depth, RIG)
        self.assertAlmostEqual(m["near_x_m"], 11.5, delta=0.1)

    def test_invalid_measurements(self):
        depth, boxes = render([(21.5, -0.9, 0.9, 1.5)])
        self.assertEqual(measure_box({"x": 250, "y": 300, "w": 100, "h": 50},
                                     depth, RIG)["reason"], "only_road")
        self.assertEqual(measure_box({"x": 250, "y": 10, "w": 100, "h": 50},
                                     depth, RIG)["reason"], "no_depth")
        self.assertEqual(measure_box({"x": 700, "y": 10, "w": 50, "h": 50},
                                     depth, RIG)["reason"], "box_outside_image")
        self.assertEqual(measure_box(boxes[0], depth[:100], RIG)["reason"], "depth_size")


class Tracking(unittest.TestCase):
    def test_parked_car_while_ego_drives_and_turns(self):
        speed, yaw_rate = 2.5, math.radians(6.0)  # 9 km/h, gentle turn right

        def ego_at(t):
            yaw = yaw_rate * t
            # Arc of radius speed / yaw_rate
            radius = speed / yaw_rate
            x, y = radius * math.sin(yaw), radius * (1 - math.cos(yaw))
            return (Pose(x, y, 0, 0, 0, math.degrees(yaw)),
                    (speed * math.cos(yaw), speed * math.sin(yaw)), yaw_rate)

        obs = drive(MetricTracker(RIG), 60, ego_at, lambda t: (25.0, 1.0))
        self.assertFalse(obs[0]["kinematics_valid"])
        self.assertIsNone(obs[0]["velocity_world_mps"])
        self.assertTrue(obs[-1]["kinematics_valid"])
        late = obs[30:]
        world = max(math.hypot(*o["velocity_world_mps"]) for o in late)
        self.assertLess(world, 0.3, "a parked car has no world velocity")
        # Seen from the ego vehicle it closes at about the ego speed
        self.assertAlmostEqual(late[-1]["velocity_relative_mps"][0], -speed, delta=0.5)
        # The naive derivative of the ego-frame position is far from zero
        naive = (obs[-1]["position_ego_m"][0] - obs[-2]["position_ego_m"][0]) / (DT_NS * 1e-9)
        self.assertGreater(abs(naive), 1.5)
        for o in obs:
            self.assertTrue(all(math.isfinite(v) and v > 0 for v in o["covariance"]))

    def test_constant_speed_and_first_frames(self):
        still = lambda t: (Pose(0, 0, 0), (0.0, 0.0), 0.0)
        obs = drive(MetricTracker(RIG), 60, still, lambda t: (15.0 + 3.0 * t, 0.5))
        self.assertEqual([o["kinematics_valid"] for o in obs[:3]], [False, False, False])
        for o in obs[30:]:
            self.assertAlmostEqual(o["velocity_world_mps"][0], 3.0, delta=0.4)
            self.assertAlmostEqual(o["velocity_world_mps"][1], 0.0, delta=0.3)
        self.assertAlmostEqual(obs[-1]["velocity_relative_mps"][0], 3.0, delta=0.4)

    def test_acceleration(self):
        still = lambda t: (Pose(0, 0, 0), (0.0, 0.0), 0.0)
        obs = drive(MetricTracker(RIG), 90, still,
                    lambda t: (12.0 + 0.5 * 2.0 * t * t, 0.0))
        self.assertFalse(obs[2]["acceleration_valid"])
        late = [o["acceleration_world_mps2"][0] for o in obs[45:]]
        self.assertTrue(all(o["acceleration_valid"] for o in obs[45:]))
        self.assertAlmostEqual(float(np.mean(late)), 2.0, delta=0.6)

    def test_dropped_frames_use_the_real_gap(self):
        still = lambda t: (Pose(0, 0, 0), (0.0, 0.0), 0.0)
        skip = {10, 11, 12, 20, 33, 34, 35, 36, 37}
        obs = drive(MetricTracker(RIG), 60, still, lambda t: (15.0 + 3.0 * t, 0.0),
                    skip=skip)
        self.assertAlmostEqual(obs[-1]["velocity_world_mps"][0], 3.0, delta=0.4)

    def test_repeat_and_out_of_order_frames_are_not_used(self):
        tracker = MetricTracker(RIG)
        depth, boxes = render([(21.5, -0.9, 0.9, 1.5)])
        pose = Pose(0, 0, 0)
        self.assertIsNotNone(tracker.update(perception(5 * DT_NS, boxes, [1]), depth,
                                            ego_state(5 * DT_NS, pose)))
        before = tracker.tracks[("r1", "front", 1)].updates
        for ns in (5 * DT_NS, 4 * DT_NS):
            self.assertIsNone(tracker.update(perception(ns, boxes, [1]), depth,
                                             ego_state(ns, pose)))
        self.assertEqual(tracker.tracks[("r1", "front", 1)].updates, before)
        with self.assertRaises(ValueError):
            tracker.update(perception(6 * DT_NS, boxes, [1]), depth,
                           ego_state(7 * DT_NS, pose))
        with self.assertRaises(ValueError):
            tracker.update(perception(8 * DT_NS, boxes, [1]), depth,
                           ego_state(8 * DT_NS, pose, run="r2"))
        # Another run keeps its own tracks
        tracker.update(perception(0, boxes, [1], run="r2"), depth,
                       ego_state(0, pose, run="r2"))
        self.assertEqual(len(tracker.tracks), 2)

    def test_no_track_id_gives_a_position_only(self):
        tracker = MetricTracker(RIG)
        depth, boxes = render([(9.5, -0.3, 0.3, 1.7)])
        result = tracker.update(perception(0, boxes, [None], name="person"), depth,
                                ego_state(0, Pose(0, 0, 0)))
        o = result["observations"][0]
        self.assertTrue(o["valid"])
        self.assertAlmostEqual(o["near_x_m"], 9.5, delta=0.1)
        self.assertFalse(o["kinematics_valid"])
        self.assertIsNone(o["velocity_world_mps"])
        self.assertEqual(len(tracker.tracks), 0)

    def test_id_reused_after_a_gap_is_a_new_object(self):
        tracker = MetricTracker(RIG)
        pose = Pose(0, 0, 0)
        for i in range(10):
            depth, boxes = render([(21.5 + 0.1 * i, -0.9, 0.9, 1.5)])
            tracker.update(perception(i * DT_NS, boxes, [3]), depth,
                           ego_state(i * DT_NS, pose))
        late = 10 * DT_NS + int(2e9)
        depth, boxes = render([(41.5, 3.0, 4.8, 1.5)])
        o = tracker.update(perception(late, boxes, [3]), depth,
                           ego_state(late, pose))["observations"][0]
        self.assertFalse(o["kinematics_valid"])
        self.assertAlmostEqual(o["filtered_world_m"][0], 41.5, delta=0.2)

    def test_jump_is_gated_then_restarts(self):
        tracker = MetricTracker(RIG)
        pose = Pose(0, 0, 0)
        for i in range(20):
            depth, boxes = render([(21.5, -0.9, 0.9, 1.5)])
            tracker.update(perception(i * DT_NS, boxes, [3]), depth,
                           ego_state(i * DT_NS, pose))
        depth, boxes = render([(31.5, -0.9, 0.9, 1.5)])
        reasons = []
        for i in range(20, 24):
            o = tracker.update(perception(i * DT_NS, boxes, [3]), depth,
                               ego_state(i * DT_NS, pose))["observations"][0]
            reasons.append(o["reason"])
            # The raw range is reported whatever the filter thinks of it
            self.assertAlmostEqual(o["near_x_m"], 31.5, delta=0.1)
        self.assertEqual(reasons[:3], ["gated", "gated", "gated"])
        self.assertAlmostEqual(o["filtered_world_m"][0], 31.5, delta=0.3)

    def test_missing_track_is_predicted_then_retired(self):
        tracker = MetricTracker(RIG)
        pose = Pose(0, 0, 0)
        depth, boxes = render([(21.5, -0.9, 0.9, 1.5)])
        for i in range(5):
            tracker.update(perception(i * DT_NS, boxes, [3]), depth,
                           ego_state(i * DT_NS, pose))
        o = tracker.update(perception(5 * DT_NS, [], []), depth,
                           ego_state(5 * DT_NS, pose))["observations"]
        self.assertEqual(len(o), 1)
        self.assertTrue(o[0]["predicted_only"])
        self.assertEqual(o[0]["measurement_sim_ns"], 4 * DT_NS)
        late = 5 * DT_NS + int(1e9)
        o = tracker.update(perception(late, [], []), depth,
                           ego_state(late, pose))["observations"]
        self.assertEqual(o, [])
        self.assertEqual(len(tracker.tracks), 0)


if __name__ == "__main__":
    unittest.main()
