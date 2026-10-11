import copy
import math
import os
import sys
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import fixtures  # noqa: E402
from test_metric_tracker import render  # noqa: E402
from video_analytics_simulation import lane, protocol  # noqa: E402
from video_analytics_simulation.calibration import Pose, Rig  # noqa: E402
from video_analytics_simulation.controller import (  # noqa: E402
    RUNNING, STOP_LATCHED, WAIT_DATA, ControlCore, lane_config_from, pure_pursuit,
    stopping_distance)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
with open(os.path.join(ROOT, "deploy", "config", "carla.yaml")) as f:
    CONFIG = yaml.safe_load(f)
# Half the pixels of the real rig: the same geometry, tests four times faster
CONFIG["calibration"]["width"] = 640
CONFIG["calibration"]["height"] = 360
RIG = Rig.from_config(CONFIG["calibration"])
LANE_CONFIG = lane_config_from(CONFIG["lane"])
VEHICLE = {"wheelbase_m": 3.0, "max_steer_deg": 70.0, "width_m": 2.16,
           "front_bumper_m": 2.43}
TICK = 33333333
MS = 1000000


def lane_envelope(offset=0.0, curvature=0.0, heading=0.0, markings=(-1.75, 1.75)):
    """What the device would send for a lane whose centre is `offset` metres
    to the right of the vehicle."""
    pixels = [fixtures.lane_pixels(RIG, m + offset, curvature, heading,
                                   xs=np.arange(5.5, 32.0, 1.0)) for m in markings]
    return lane.estimate_lane(pixels, RIG, LANE_CONFIG)


class World:
    """A scripted scene feeding a ControlCore as the ROS node would."""

    def __init__(self, config=CONFIG, run="r1"):
        self.core = ControlCore(config, VEHICLE, run)
        self.run = run
        self.tick = 0
        self.pose = Pose(0, 0, 0)
        self.speed = 0.0
        self.objects = []  # (class, x, y_left, y_right, height, track id) in the ego frame
        self.lane = lane_envelope()
        self.last = None

    def ego(self, tick=None, pose=None, run=None):
        tick = self.tick if tick is None else tick
        pose = pose or self.pose
        yaw = math.radians(pose.yaw)
        return {"run_id": run or self.run, "frame": 1000 + tick, "sim_ns": tick * TICK,
                "pose": dict(x=pose.x, y=pose.y, z=pose.z, roll=0.0, pitch=0.0,
                             yaw=pose.yaw),
                "velocity_world_mps": [self.speed * math.cos(yaw),
                                       self.speed * math.sin(yaw), 0.0],
                "speed_mps": self.speed, "yaw_rate_radps": 0.0}

    def envelope(self, tick=None, **changes):
        tick = self.tick if tick is None else tick
        depth, boxes = render([o[1:5] for o in self.objects], RIG)
        detections = []
        for (name, *_rest, track_id), box in zip(self.objects, boxes):
            d = {"class": name, "confidence": 0.9, "bbox": box}
            if track_id is not None:
                d["track_id"] = track_id
            detections.append(d)
        envelope = {"simulation": {"run_id": self.run, "sim_ns": tick * TICK},
                    "stream": "carla-front",
                    "image": {"width": RIG.width, "height": RIG.height},
                    "detections": detections, "lane": copy.deepcopy(self.lane)}
        envelope.update(changes)
        return envelope, depth

    def frame(self, deliver=True, with_depth=True, **changes):
        """One tick: ego state, and (unless withheld) depth and the result."""
        ego = self.ego()
        self.core.on_ego(ego)
        envelope, depth = self.envelope(**changes)
        if with_depth:
            self.core.on_depth(self.tick * TICK, depth)
        result = None
        if deliver:
            result = self.core.on_perception(envelope, ego["sim_ns"])
        self.last = self.core.step(ego)
        self.tick += 1
        return result

    def arm(self, run=None):
        return self.core.on_arm({"run_id": run or self.run, "enable": True},
                                self.ego(self.tick - 1))

    def running(self):
        for _ in range(3):
            self.frame()
        assert self.arm() == "", self.core.gates(self.ego(self.tick - 1))
        return self


class Maths(unittest.TestCase):
    def test_stopping_distance(self):
        brake = {"a_min_mps2": 2.5, "tau_s": 0.3, "margin_m": 2.0}
        v = 10 / 3.6
        self.assertAlmostEqual(stopping_distance(v, brake),
                               v * 0.3 + v * v / 5.0 + 2.0)
        self.assertAlmostEqual(stopping_distance(v, brake), 4.377, places=3)
        self.assertEqual(stopping_distance(0.0, brake), 2.0)

    def test_pure_pursuit_signs(self):
        steer = lambda y: pure_pursuit(8.0, y, 3.0, -1.5, math.radians(70))
        self.assertEqual(steer(0.0), 0.0)
        self.assertGreater(steer(1.0), 0.0)   # target to the right: steer right
        self.assertLess(steer(-1.0), 0.0)
        self.assertAlmostEqual(steer(1.0), -steer(-1.0))
        self.assertGreater(steer(2.0), steer(1.0))
        # Wheel angle for a point 1 m right, 9.5 m ahead of the rear axle
        alpha = math.atan2(1.0, 9.5)
        wheel = math.atan2(2 * 3.0 * math.sin(alpha), math.hypot(9.5, 1.0))
        self.assertAlmostEqual(steer(1.0), wheel / math.radians(70))
        # More than the wheels can turn: clipped
        self.assertEqual(pure_pursuit(0.0, 5.0, 3.0, -1.5, 0.01), 1.0)
        self.assertEqual(pure_pursuit(0.0, -5.0, 3.0, -1.5, 0.01), -1.0)


class Arming(unittest.TestCase):
    def test_brakes_until_armed_and_never_arms_itself(self):
        world = World()
        self.assertEqual(world.arm(), "gates:no_perception;fresh_frames_0")
        for _ in range(30):
            world.frame()
            self.assertEqual((world.last["throttle"], world.last["brake"]), (0.0, 1.0))
            self.assertEqual(world.last["reason"], "wait_data")
        # Every gate passes, and it still waits
        self.assertEqual(world.core.state, WAIT_DATA)
        self.assertTrue(world.core.status(world.ego(world.tick - 1))["ready"])
        self.assertEqual(world.arm(run="r0"), "wrong_run")
        self.assertEqual(world.core.on_arm({"run_id": "r1"}, world.ego()),
                         "not_an_arm_message")
        self.assertEqual(world.core.state, WAIT_DATA)
        self.assertEqual(world.arm(), "")
        self.assertEqual(world.core.state, RUNNING)
        world.frame()
        self.assertGreater(world.last["throttle"], 0.0)
        self.assertEqual(world.last["brake"], 0.0)

    def test_gates(self):
        world = World()
        world.frame()
        world.frame()
        self.assertEqual(world.arm(), "gates:fresh_frames_2")
        world.lane = lane_envelope(markings=(-1.75,))
        world.frame()
        self.assertIn("lane_invalid", world.arm())
        world.lane = lane_envelope()
        for _ in range(3):
            world.frame()
        world.speed = 1.0
        world.core.on_ego(world.ego(world.tick - 1))
        self.assertEqual(world.arm(), "gates:moving")

    def test_class_filter_that_drops_a_hazard_class_blocks_arming(self):
        only_car = dict(CONFIG, classes=["car"])
        world = World(only_car)
        for _ in range(5):
            world.frame()
        refused = world.arm()
        self.assertIn("classes_filter_drops:bicycle,bus,motorcycle,person,truck", refused)
        self.assertEqual(world.core.state, WAIT_DATA)
        all_six = dict(CONFIG, classes=list(protocol.HAZARD_CLASSES) + ["dog"])
        world = World(all_six)
        for _ in range(3):
            world.frame()
        self.assertEqual(world.arm(), "")
        # The shipped config leaves the key out: nothing is filtered
        self.assertFalse(CONFIG.get("classes"))


class Driving(unittest.TestCase):
    def test_straight_lane(self):
        world = World().running()
        for _ in range(20):
            world.frame()
            c = world.last
            self.assertEqual(c["reason"], "")
            self.assertAlmostEqual(c["steer"], 0.0, delta=0.01)
            self.assertFalse(c["throttle"] > 0 and c["brake"] > 0)
            self.assertLessEqual(c["throttle"], CONFIG["control"]["speed"]["max_throttle"])
        self.assertEqual(world.core.state, RUNNING)

    def test_steers_towards_the_lane_centre(self):
        for offset, sign in ((0.6, 1), (-0.6, -1)):
            world = World()
            world.lane = lane_envelope(offset=offset)
            world.running()
            steers = []
            for _ in range(10):
                world.frame()
                steers.append(world.last["steer"])
            self.assertGreater(sign * steers[-1], 0.01)
            # Rate limited: no step larger than rate * dt
            limit = CONFIG["control"]["steer"]["rate_per_s"] / 30 + 1e-9
            self.assertLessEqual(max(abs(np.diff([0.0] + steers))), limit)
        world = World()
        world.lane = lane_envelope(curvature=0.004)
        world.running()
        world.frame()
        self.assertGreater(world.last["steer"], 0.0)  # a curve to the right

    def test_speed_loop(self):
        world = World().running()
        world.speed = 4.0  # faster than 10 km/h
        world.frame()
        self.assertEqual(world.last["throttle"], 0.0)
        self.assertGreater(world.last["brake"], 0.0)
        world = World().running()
        for _ in range(300):  # held back at standstill: the integral is bounded
            world.frame()
        self.assertLessEqual(world.last["throttle"],
                             CONFIG["control"]["speed"]["max_throttle"])
        self.assertLessEqual(abs(world.core.integral), 2.0)

    def test_commands_pass_the_simulators_check(self):
        world = World().running()
        world.frame()
        c = world.last
        now = c["control_sim_ns"]
        limits = {"age_limit_ns": 200 * MS, "pose_budget_ns": 100 * MS}
        self.assertEqual(protocol.check_command(
            c, "r1", "carla-front", c["command_seq"] - 1, now, limits,
            lambda ns: True, lambda ns, frame: True), "")
        self.assertEqual(c["valid_until_sim_ns"], c["basis_sim_ns"] + 200 * MS)


class Stopping(unittest.TestCase):
    def test_stale_perception_latches_until_reset(self):
        world = World().running()
        world.frame()
        ticks = 0
        while world.core.state == RUNNING and ticks < 20:
            world.frame(deliver=False)
            ticks += 1
        # 200 ms at 30 Hz: the 7th tick without a result is too old
        self.assertEqual(ticks, 7)
        self.assertEqual(world.core.stop_reason, "stale_perception")
        self.assertEqual((world.last["throttle"], world.last["brake"]), (0.0, 1.0))
        for _ in range(5):  # fresh data again: still stopped
            world.frame()
        self.assertEqual(world.core.state, STOP_LATCHED)
        self.assertEqual(world.last["brake"], 1.0)
        self.assertEqual(world.arm(), "state_STOP_LATCHED")

    def test_lane_lost(self):
        world = World().running()
        world.lane = lane_envelope(markings=(-1.75,))
        world.frame()
        self.assertEqual(world.core.stop_reason, "lane:missing_boundary")
        self.assertEqual(world.last["brake"], 1.0)

    def test_frames_that_cannot_be_joined(self):
        world = World().running()
        self.assertEqual(world.frame(with_depth=False)[1], "no_depth_for_frame")
        bad = [
            (dict(simulation={"run_id": "r0", "sim_ns": world.tick * TICK}), "wrong_run"),
            (dict(stream="rear"), "wrong_source"),
            (dict(image={"width": 640, "height": 480}), "image_size"),
            (dict(simulation={"run_id": "r1", "sim_ns": float(world.tick * TICK)}),
             "bad_sim_ns"),
            (dict(lane=dict(lane_envelope(), calibration_id="rig_other")),
             "calibration_mismatch"),
            (dict(simulation={"run_id": "r1", "sim_ns": 10 ** 15}), "no_ego_for_frame"),
        ]
        for changes, reason in bad:
            ego = world.ego()
            world.core.on_ego(ego)
            envelope, depth = world.envelope(**changes)
            world.core.on_depth(world.tick * TICK, depth)
            self.assertEqual(world.core.on_perception(envelope, ego["sim_ns"])[1], reason)
        # A repeat of the frame already used, and an older one
        world = World().running()
        envelope, _ = world.envelope(tick=world.tick - 1)
        self.assertEqual(world.core.on_perception(envelope)[1], "not_newer")
        envelope, _ = world.envelope(tick=world.tick - 3)
        self.assertEqual(world.core.on_perception(envelope)[1], "not_newer")
        self.assertEqual(world.core.on_ego(world.ego(run="r0")), "wrong_run")

    def test_missing_depth_stops_by_the_deadline(self):
        world = World().running()
        ticks = 0
        while world.core.state == RUNNING and ticks < 20:
            world.frame(with_depth=False)
            ticks += 1
        self.assertEqual(world.core.stop_reason, "stale_perception")
        self.assertLessEqual(ticks, 7)


class Obstacles(unittest.TestCase):
    def test_car_ahead_inside_the_stopping_distance(self):
        world = World().running()
        world.speed = 2.7
        # Rear of a car 6.5 m ahead of the origin: gap 4.07 m < 4.3 m
        world.objects = [("car", 6.5, -0.9, 0.9, 1.5, None)]
        world.frame()
        self.assertEqual(world.core.stop_reason, "obstacle:distance:car")
        self.assertEqual((world.last["throttle"], world.last["brake"]), (0.0, 1.0))
        decision = world.core.debug["obstacles"][0]
        self.assertAlmostEqual(decision["gap_m"], 6.5 - VEHICLE["front_bumper_m"],
                               delta=0.1)
        # It had no track id: the first raw detection is enough to brake

    def test_car_ahead_but_far(self):
        world = World().running()
        world.speed = 2.7
        world.objects = [("car", 20.0, -0.9, 0.9, 1.5, 4)]
        for _ in range(10):
            world.frame()
        self.assertEqual(world.core.state, RUNNING)
        self.assertEqual(world.core.debug["obstacles"][0]["decision"], "in_path_far")

    def test_closing_car_brakes_on_time_to_collision(self):
        world = World().running()
        world.speed = 0.0
        for i in range(90):
            # Coming towards the vehicle at 6 m/s from 25 m
            x = 25.0 - 6.0 * i / 30
            world.objects = [("car", x, -0.9, 0.9, 1.5, 9)]
            world.frame()
            if world.core.state != RUNNING:
                break
        self.assertEqual(world.core.stop_reason, "obstacle:ttc:car")
        gap = world.core.debug["obstacles"][0]["gap_m"]
        self.assertGreater(gap, stopping_distance(0.0, CONFIG["control"]["brake"]))
        self.assertLess(gap / 6.0, CONFIG["control"]["ttc_min_s"] + 0.2)

    def test_adjacent_lane_and_roadside_do_not_brake(self):
        world = World().running()
        world.speed = 2.7
        world.objects = [("car", 12.0, 2.6, 4.4, 1.5, 1),      # lane to the right
                         ("truck", 13.0, -4.6, -2.5, 3.0, 2),  # lane to the left
                         ("person", 14.0, 5.0, 5.5, 1.7, 3)]   # on the verge
        for _ in range(15):
            world.frame()
        self.assertEqual(world.core.state, RUNNING)
        self.assertEqual({d["decision"] for d in world.core.debug["obstacles"]},
                         {"outside_path"})

    def test_person_in_the_path(self):
        world = World().running()
        world.speed = 2.0
        world.objects = [("person", 5.5, -0.2, 0.3, 1.7, None)]
        world.frame()
        self.assertEqual(world.core.stop_reason, "obstacle:distance:person")

    def test_other_classes_are_logged_not_braked_for(self):
        world = World().running()
        world.speed = 2.7
        world.objects = [("dog", 5.0, -0.3, 0.3, 0.6, None),
                         ("bench", 5.5, -0.8, 0.8, 0.9, 5)]
        for _ in range(5):
            world.frame()
        self.assertEqual(world.core.state, RUNNING)
        self.assertEqual([d["decision"] for d in world.core.debug["obstacles"]],
                         ["non_hazard_class", "non_hazard_class"])

    def test_curve_follows_the_lane_not_the_image_centre(self):
        # A lane curving right: a car straight ahead at 22 m is outside it,
        # a car to the right at 22 m is in it
        world = World()
        world.lane = lane_envelope(curvature=0.008)
        world.running()
        y = 0.008 * 22 * 22  # 3.9 m right
        world.objects = [("car", 22.0, -0.9, 0.9, 1.5, 1),
                         ("car", 22.0, y - 0.9, y + 0.9, 1.5, 2)]
        world.frame()
        decisions = [d["decision"] for d in world.core.debug["obstacles"]]
        self.assertEqual(decisions, ["outside_path", "in_path_far"])

    def test_hazard_without_usable_depth(self):
        world = World().running()
        world.speed = 2.7
        ego = world.ego()
        world.core.on_ego(ego)
        envelope, depth = world.envelope()
        # A person 6 m ahead whose box holds no depth (sensor fault)
        corners, _ = RIG.project([[6.0, -0.3, 1.7], [6.0, 0.3, 0.0]])
        box = {"x": float(corners[0, 0]), "y": float(corners[0, 1]),
               "w": float(corners[1, 0] - corners[0, 0]),
               "h": float(corners[1, 1] - corners[0, 1])}
        envelope["detections"] = [{"class": "person", "confidence": 0.8, "bbox": box}]
        depth = depth.copy()
        depth[int(box["y"]):int(box["y"] + box["h"]) + 1,
              int(box["x"]):int(box["x"] + box["w"]) + 1] = 1000.0
        world.core.on_depth(world.tick * TICK, depth)
        self.assertEqual(world.core.on_perception(envelope, ego["sim_ns"])[1], "")
        command = world.core.step(ego)
        self.assertEqual(command["reason"], "obstacle:unreadable:person")
        self.assertEqual(command["brake"], 1.0)


class DelayedSource(unittest.TestCase):
    """The result of a frame arrives after the vehicle has moved on."""

    def setUp(self):
        self.world = World().running()
        world = self.world
        world.speed = 2.7
        world.lane = lane_envelope(offset=0.3)
        # Outside the stopping distance when seen: gap 9 - 2.43 = 6.57 m
        world.objects = [("car", 9.0, -0.6, 1.2, 1.5, None)]
        ego = world.ego()
        world.core.on_ego(ego)
        envelope, depth = world.envelope()
        world.core.on_depth(world.tick * TICK, depth)
        self.assertEqual(world.core.on_perception(envelope, ego["sim_ns"])[1], "")
        self.basis_ns = world.tick * TICK
        self.capture = world.pose

    def test_lane_and_gap_are_in_the_current_ego_frame(self):
        world = self.world
        # 4 ticks later: 1.0 m forward, 0.2 m to the right, turned 3 degrees
        moved = Pose(1.0, 0.2, 0, 0, 0, 3.0)
        world.tick += 4
        ego = world.ego(pose=moved)
        world.core.on_ego(ego)
        command = world.core.step(ego)
        self.assertEqual(command["reason"], "")

        reference = lane.rebase_lane(world.lane, self.capture, moved, 0.0, LANE_CONFIG)
        debug = world.core.debug
        self.assertAlmostEqual(debug["lane"]["lateral_error_m"],
                               reference["lateral_error_m"], places=9)
        self.assertAlmostEqual(debug["lane"]["heading_rad"], reference["heading_rad"],
                               places=9)
        # The centre was 0.3 m right; now it is about 0.1 m right and the lane
        # points 3 degrees to the left of the vehicle
        self.assertAlmostEqual(debug["lane"]["heading_rad"], -math.radians(3), delta=2e-3)
        # The support moved with the vehicle and got shorter
        self.assertLess(debug["lane"]["x_max_m"], world.lane["centerline"]["x_max_m"] - 0.9)

        # The obstacle, by the same transform, with the unknown-velocity bound
        near = moved.from_parent(self.capture.to_parent(
            [[9.0, -0.6, 0.0], [9.0, 1.2, 0.0]]))
        age = 4 * TICK * 1e-9
        expected = near[:, 0].min() - CONFIG["control"]["unknown_closing_mps"] * age \
            - VEHICLE["front_bumper_m"]
        self.assertAlmostEqual(debug["obstacles"][0]["gap_m"], expected, delta=0.06)
        # The command still names the frame it rests on and when that expires
        self.assertEqual(command["basis_sim_ns"], self.basis_ns)
        self.assertEqual(command["control_sim_ns"], world.tick * TICK)
        self.assertEqual(command["valid_until_sim_ns"], self.basis_ns + 200 * MS)

    def test_brakes_when_the_vehicle_has_closed_the_gap(self):
        world = self.world
        moved = Pose(2.6, 0.0, 0, 0, 0, 0.0)  # gap now 9 - 2.6 - 2.43 = 3.97 m
        world.tick += 5
        ego = world.ego(pose=moved)
        world.core.on_ego(ego)
        command = world.core.step(ego)
        self.assertEqual(command["reason"], "obstacle:distance:car")
        self.assertEqual(command["brake"], 1.0)

    def test_no_lane_left_to_steer_on(self):
        world = self.world
        world.objects = []
        moved = Pose(29.0, 0.0, 0, 0, 0, 0.0)
        world.tick += 2
        ego = world.ego(pose=moved)
        world.core.on_ego(ego)
        world.core.basis["metric"]["observations"] = []
        command = world.core.step(ego)
        self.assertEqual(command["reason"], "lane:insufficient_support")
        self.assertEqual(world.core.state, STOP_LATCHED)

    def test_a_pose_that_is_not_usable(self):
        world = self.world
        world.tick += 1
        ego = world.ego()
        ego["pose"]["yaw"] = float("nan")
        world.core.on_ego(ego)
        self.assertEqual(world.core.step(ego)["reason"], "bad_ego_state")


if __name__ == "__main__":
    unittest.main()
