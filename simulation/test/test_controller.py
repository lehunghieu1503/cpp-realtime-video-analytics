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
    HOLDING, LANE_LATCH_FRAMES, RUNNING, STOP_LATCHED, WAIT_DATA, ControlCore,
    lane_config_from, pure_pursuit, stopping_distance)

with open(os.path.join(os.path.dirname(__file__), "..", "config", "carla.yaml")) as f:
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
AGE_MS = CONFIG["control"]["age_limit_ms"]
# Ticks without a result until the last one is older than the age limit
STALE_TICKS = int(AGE_MS * MS // TICK) + 1


def lane_envelope(offset=0.0, curvature=0.0, heading=0.0, markings=(-1.75, 1.75)):
    """What the device would send for a lane whose centre is `offset` metres
    to the right of the vehicle."""
    pixels = [fixtures.lane_pixels(RIG, m + offset, curvature, heading,
                                   xs=np.arange(5.5, 32.0, 1.0)) for m in markings]
    # As the device sends it: the fitted lane and the pixels it came from
    return dict(lane.estimate_lane(pixels, RIG, LANE_CONFIG), pixels=pixels)


class World:
    """A scripted scene feeding a ControlCore as the ROS node would."""

    def __init__(self, config=CONFIG, run="r1", scenario=None):
        self.core = ControlCore(config, VEHICLE, run, scenario=scenario)
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
                "tick": tick,
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

    def test_perception_close_to_its_limit_does_not_arm(self):
        world = World()
        for _ in range(3):
            world.frame()
        # No newer result for a while: older than 60 % of the limit, still
        # inside it. Armed now, the next tick could already be too late.
        ticks = int(0.8 * AGE_MS * MS // TICK)
        world.tick += ticks - 1
        ego = world.ego()
        world.core.on_ego(ego)
        age_ms = (ego["sim_ns"] - world.core.basis["sim_ns"]) / MS
        self.assertTrue(0.6 * AGE_MS < age_ms <= AGE_MS)
        refused = world.core.on_arm({"run_id": "r1", "enable": True}, ego)
        self.assertIn("stale_perception", refused)
        self.assertEqual(world.core.state, WAIT_DATA)
        self.assertFalse(world.core.status(ego)["ready"])

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
        limits = {"age_limit_ns": AGE_MS * MS, "pose_budget_ns": 100 * MS}
        self.assertEqual(protocol.check_command(
            c, "r1", "carla-front", c["command_seq"] - 1, now, limits,
            lambda ns: True, lambda ns, frame: True), "")
        self.assertEqual(c["valid_until_sim_ns"], c["basis_sim_ns"] + AGE_MS * MS)


class Stopping(unittest.TestCase):
    def test_stale_perception_latches_until_reset(self):
        world = World().running()
        world.frame()
        ticks = 0
        while world.core.state == RUNNING and ticks < 40:
            world.frame(deliver=False)
            ticks += 1
        # The first tick at which the last result is older than the limit
        self.assertEqual(ticks, STALE_TICKS)
        self.assertEqual(world.core.stop_reason, "stale_perception")
        self.assertEqual((world.last["throttle"], world.last["brake"]), (0.0, 1.0))
        for _ in range(5):  # fresh data again: still stopped
            world.frame()
        self.assertEqual(world.core.state, STOP_LATCHED)
        self.assertEqual(world.last["brake"], 1.0)
        self.assertEqual(world.arm(), "state_STOP_LATCHED")

    def test_lane_lost_for_a_frame_holds_then_drives_on(self):
        world = World().running()
        world.frame()
        good = world.lane
        world.lane = lane_envelope(markings=(-1.75,))
        world.frame()
        self.assertEqual(world.core.state, HOLDING)
        self.assertEqual(world.core.stop_reason, "lane:missing_boundary")
        self.assertEqual(world.last["reason"], "lane:missing_boundary")
        self.assertEqual((world.last["throttle"], world.last["brake"]), (0.0, 1.0))
        self.assertTrue(world.core.debug["hold"])
        world.lane = good
        world.frame()  # the next valid lane: no second of waiting
        self.assertEqual(world.core.state, RUNNING)
        self.assertGreater(world.last["throttle"], 0.0)
        self.assertEqual(world.last["reason"], "")

    def test_lane_lost_for_five_frames_latches(self):
        world = World().running()
        good = world.lane
        world.lane = lane_envelope(markings=(-1.75,))
        for _ in range(LANE_LATCH_FRAMES - 1):
            world.frame()
            self.assertEqual(world.core.state, HOLDING)
        world.frame()
        self.assertEqual(world.core.state, STOP_LATCHED)
        self.assertEqual(world.core.stop_reason, "lane:missing_boundary")
        world.lane = good
        for _ in range(40):
            world.frame()
            self.assertEqual((world.last["throttle"], world.last["brake"]), (0.0, 1.0))
        self.assertEqual(world.core.state, STOP_LATCHED)

    def test_one_bad_lane_result_used_for_many_ticks_counts_once(self):
        world = World().running()
        world.lane = lane_envelope(markings=(-1.75,))
        world.frame()
        for _ in range(LANE_LATCH_FRAMES + 2):  # no newer result arrives
            world.frame(deliver=False)
        self.assertEqual(world.core.state, HOLDING)
        self.assertEqual(world.core.lane_bad, 1)

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
        while world.core.state == RUNNING and ticks < 40:
            world.frame(with_depth=False)
            ticks += 1
        self.assertEqual(world.core.stop_reason, "stale_perception")
        self.assertLessEqual(ticks, STALE_TICKS)


class Obstacles(unittest.TestCase):
    def test_car_ahead_inside_the_stopping_distance(self):
        world = World().running()
        world.speed = 2.7
        # Rear of a car 6.5 m ahead of the origin: gap 4.07 m < 4.3 m
        world.objects = [("car", 6.5, -0.9, 0.9, 1.5, None)]
        world.frame()
        self.assertEqual(world.core.stop_reason, "obstacle:distance:car")
        self.assertEqual((world.last["throttle"], world.last["brake"]), (0.0, 1.0))
        self.assertEqual(world.core.state, HOLDING)
        self.assertTrue(world.core.debug["hold"])
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

    def test_a_box_with_nothing_above_the_road_is_not_braked_for(self):
        # A shadow in the corner of the image taken for a person: the depth
        # inside the box is the road's
        world = World().running()
        world.speed = 2.7
        ego = world.ego()
        world.core.on_ego(ego)
        envelope, depth = world.envelope()
        height = RIG.height
        envelope["detections"] = [{"class": "person", "confidence": 0.4, "bbox": {
            "x": 0.0, "y": height - 110.0, "w": 100.0, "h": 109.0}}]
        world.core.on_depth(world.tick * TICK, depth)
        self.assertEqual(world.core.on_perception(envelope, ego["sim_ns"])[1], "")
        command = world.core.step(ego)
        self.assertEqual(world.core.debug["obstacles"][0]["measurement"], "only_road")
        self.assertEqual(world.core.debug["obstacles"][0]["decision"], "flat_on_road")
        self.assertEqual(command["reason"], "")
        self.assertEqual(world.core.state, RUNNING)


class HoldAndResume(unittest.TestCase):
    CLEAR = CONFIG["control"]["clear_frames"]

    def held(self):
        """Stopped behind a car that then drives away."""
        world = World().running()
        world.speed = 2.5
        world.objects = [("car", 6.5, -0.9, 0.9, 1.5, 7)]
        world.frame()
        self.assertEqual(world.core.state, HOLDING)
        world.speed = 0.0
        return world

    def test_obstacle_holds_and_the_vehicle_drives_on_when_it_is_gone(self):
        world = self.held()
        for _ in range(20):
            world.frame()
            self.assertEqual(world.last["reason"], "obstacle:distance:car")
            self.assertEqual((world.last["throttle"], world.last["brake"]), (0.0, 1.0))
            self.assertTrue(world.core.debug["hold"])
        world.objects = []
        # Predicted for max_gap_s, then clear_frames results really clear
        ticks = 0
        while world.core.state == HOLDING and ticks < 200:
            world.frame()
            ticks += 1
        gap_ticks = int(CONFIG["metric"]["max_gap_s"] * 30)
        self.assertGreaterEqual(ticks, gap_ticks + self.CLEAR)
        self.assertLessEqual(ticks, gap_ticks + self.CLEAR + 2)
        self.assertEqual(world.core.state, RUNNING)
        self.assertEqual(world.core.stop_reason, "")
        self.assertGreater(world.last["throttle"], 0.0)
        self.assertNotIn("hold", world.core.debug)

    def test_an_obstacle_that_stays_is_not_crept_up_to(self):
        # Standing still its gap is larger than the stopping distance from
        # 0 m/s; it is not larger than the one from the cruising speed
        world = self.held()
        for _ in range(4 * self.CLEAR):
            world.frame()
        self.assertEqual(world.core.state, HOLDING)
        self.assertEqual(world.last["brake"], 1.0)

    def test_a_missed_detection_is_still_the_obstacle(self):
        world = self.held()
        for _ in range(3):
            world.frame()
        world.objects = []  # the detector misses it for a while
        world.frame()
        predicted = world.core.basis["metric"]["observations"]
        # What the tracker really gives for a track it did not measure
        self.assertEqual([(o["predicted_only"], o["class"]) for o in predicted],
                         [(True, "car")])
        self.assertAlmostEqual(predicted[0]["near_x_m"], 6.5, delta=0.1)
        self.assertEqual(world.last["reason"], "obstacle:predicted:car")
        self.assertEqual(world.last["brake"], 1.0)
        for _ in range(8):
            world.frame()
            self.assertEqual(world.core.clear_count, 0)  # a prediction is not "clear"
        self.assertEqual(world.core.state, HOLDING)
        # Seen again before the prediction ran out: the wait starts over
        world.objects = [("car", 6.5, -0.9, 0.9, 1.5, 7)]
        world.frame()
        self.assertEqual(world.last["reason"], "obstacle:distance:car")

    def test_a_prediction_ends_with_max_gap(self):
        world = self.held()
        world.objects = []
        gap_ticks = int(CONFIG["metric"]["max_gap_s"] * 30)
        for _ in range(gap_ticks):
            world.frame()
            self.assertEqual(world.last["reason"], "obstacle:predicted:car")
        world.frame()
        world.frame()
        self.assertEqual(world.core.basis["metric"]["observations"], [])
        self.assertGreater(world.core.clear_count, 0)
        self.assertEqual(world.last["brake"], 1.0)  # still waiting for clear_frames

    def test_clear_frames_outlast_a_prediction(self):
        self.assertGreater(self.CLEAR, CONFIG["metric"]["max_gap_s"] * 30)

    def test_a_result_used_twice_is_one_clear_frame(self):
        world = self.held()
        world.objects = []
        for _ in range(int(CONFIG["metric"]["max_gap_s"] * 30) + 2):
            world.frame()
        count = world.core.clear_count
        self.assertGreater(count, 0)
        for _ in range(5):
            world.frame(deliver=False)
        self.assertEqual(world.core.clear_count, count)

    def test_stale_perception_still_latches_while_holding(self):
        world = self.held()
        for _ in range(STALE_TICKS + 1):
            world.frame(deliver=False)
        self.assertEqual(world.core.state, STOP_LATCHED)
        self.assertEqual(world.core.stop_reason, "stale_perception")
        world.objects = []
        for _ in range(3 * self.CLEAR):
            world.frame()
        self.assertEqual(world.core.state, STOP_LATCHED)


class BeyondTheLaneSeen(unittest.TestCase):
    """An object farther than the lane was observed."""

    def scene(self, curvature):
        world = World()
        world.lane = lane_envelope(curvature=curvature)
        world.running()
        world.speed = 13.0  # stopping distance 43.6 m: 40 m ahead is inside it
        return world

    def test_straight_lane_is_continued_and_the_next_lane_does_not_brake(self):
        world = self.scene(0.0)
        world.objects = [("car", 40.0, 2.6, 4.4, 1.5, 1)]
        world.frame()
        decision = world.core.debug["obstacles"][0]
        self.assertGreater(40.0, world.core.debug["lane"]["x_max_m"])
        self.assertEqual(decision["decision"], "outside_path")
        self.assertEqual(world.core.state, RUNNING)
        # In the ego lane at the same distance it does
        world.objects = [("car", 40.0, -0.9, 0.9, 1.5, 2)]
        world.frame()
        self.assertEqual(world.last["reason"], "obstacle:distance:car")
        # The lane reported is the lane seen, not the lane continued
        self.assertLess(world.core.debug["lane"]["x_max_m"], 31.0)

    def test_curved_lane_is_not_continued(self):
        world = self.scene(0.002)
        world.objects = [("car", 40.0, 2.6, 4.4, 1.5, 1)]
        world.frame()
        self.assertEqual(world.core.debug["obstacles"][0]["decision"],
                         "brake_uncertain_path")
        self.assertEqual(world.last["reason"], "obstacle:uncertain_path:car")

    def test_beyond_the_continuation_the_old_rule_holds(self):
        world = self.scene(0.0)
        world.speed = 20.0  # stopping distance 94 m
        world.objects = [("car", 60.0, 2.6, 4.4, 1.5, 1)]
        world.frame()
        self.assertEqual(world.core.debug["obstacles"][0]["decision"],
                         "brake_uncertain_path")


class SpeedLimit(unittest.TestCase):
    def test_cap_and_shared_target(self):
        self.assertEqual(CONFIG["control"]["target_speed_kmh"], 9.0)
        self.assertEqual(CONFIG["control"]["speed_cap_kmh"], 30.0)
        fast = copy.deepcopy(CONFIG)
        fast["control"]["target_speed_kmh"] = 80.0
        world = World(fast).running()
        world.speed = 30.0 / 3.6 + 1.0  # above the cap, far below 80 km/h
        world.frame()
        self.assertEqual(world.last["throttle"], 0.0)
        self.assertGreater(world.last["brake"], 0.0)
        world.speed = 30.0 / 3.6 - 1.0
        world.frame()
        self.assertGreater(world.last["throttle"], 0.0)


class Following(unittest.TestCase):
    """A scenario that sets its own speed and a follow gap. 30 km/h is
    8.33 m/s; the stopping distance from it is 20.9 m."""
    CRUISE = 30.0 / 3.6
    SHOW = {"target_speed_kmh": 30.0, "follow_gap_m": 30.0}

    def scene(self, scenario, lead_x, lead_speed=5.0, name="car", frames=8):
        """Ego at 30 km/h, a road user ahead driving at lead_speed."""
        world = World(scenario=scenario).running()
        world.speed = self.CRUISE
        for i in range(frames):
            world.pose = Pose(self.CRUISE * i / 30.0, 0, 0)
            x = lead_x + lead_speed * i / 30.0 - world.pose.x
            world.objects = [(name, x, -0.9, 0.9, 1.5, 4)]
            world.frame()
        return world

    def test_follows_a_slower_car_between_stopping_distance_and_follow_gap(self):
        world = self.scene(self.SHOW, 25.5)
        follow = world.core.debug["follow"]
        self.assertGreater(follow["gap_m"], world.core.debug["stopping_distance_m"])
        self.assertLess(follow["gap_m"], 30.0)
        self.assertAlmostEqual(follow["lead_speed_mps"], 5.0, delta=0.6)
        self.assertAlmostEqual(follow["target_speed_mps"], follow["lead_speed_mps"])
        self.assertEqual(world.core.state, RUNNING)
        self.assertEqual(world.last["reason"], "")
        # Slower than the vehicle goes: no throttle, a brake, no full stop
        self.assertEqual(world.last["throttle"], 0.0)
        self.assertGreater(world.last["brake"], 0.0)

    def test_blends_in_from_the_follow_gap(self):
        world = self.scene(self.SHOW, 30.5)  # gap about 27 m: between 24 and 30
        follow = world.core.debug["follow"]
        self.assertGreater(follow["target_speed_mps"], follow["lead_speed_mps"] + 0.5)
        self.assertLess(follow["target_speed_mps"], self.CRUISE - 0.5)

    def test_without_follow_gap_the_car_ahead_is_not_followed(self):
        world = self.scene({"target_speed_kmh": 30.0}, 25.5)
        self.assertNotIn("follow", world.core.debug)
        self.assertEqual(world.last["brake"], 0.0)
        self.assertEqual(world.last["reason"], "")

    def test_without_a_scenario_the_shared_speed_holds(self):
        world = World().running()
        self.assertEqual(world.core.target_speed_kmh, 9.0)
        self.assertIsNone(world.core.follow_gap_m)
        self.assertIsNone(world.core.lane_change)
        world.speed = 3.5  # above 9 km/h
        world.frame()
        self.assertEqual(world.last["throttle"], 0.0)

    def test_inside_the_stopping_distance_it_brakes_instead(self):
        world = self.scene(self.SHOW, 20.0)  # gap 17.6 m < 20.9 m
        self.assertEqual(world.core.state, HOLDING)
        self.assertEqual(world.last["reason"], "obstacle:distance:car")
        self.assertEqual(world.last["brake"], 1.0)
        world = self.scene(self.SHOW, 15.0, lead_speed=0.0, name="person", frames=2)
        self.assertEqual(world.last["reason"], "obstacle:distance:person")
        self.assertNotIn("follow", world.core.debug)

    def test_velocity_not_known_yet_keeps_the_target_speed(self):
        world = self.scene(self.SHOW, 25.5, frames=1)
        self.assertNotIn("follow", world.core.debug)
        self.assertEqual(world.last["reason"], "")
        self.assertEqual(world.last["brake"], 0.0)


class LaneChange(unittest.TestCase):
    """lane_change: {at_tick, side} of a scenario. Lane markings 3.5 m apart;
    `offset` moves them all to the right, as when the vehicle moves left."""
    LINES = (-5.25, -1.75, 1.75)

    def world(self, markings, at_tick=6, side="left", **more):
        world = World(scenario={"lane_change": dict(at_tick=at_tick, side=side, **more)})
        world.lane = lane_envelope(markings=markings)
        return world.running()

    def moved(self, world, start, stop, markings=None, step=0.05):
        """The vehicle moves sideways: every result shows the markings a
        little further to the right."""
        for offset in np.arange(start, stop + 1e-9, step):
            world.lane = lane_envelope(offset=float(offset), markings=markings or self.LINES)
            world.frame()

    def test_moves_over_when_the_next_marking_is_seen_and_holds_until_the_pair_changes(self):
        world = self.world(self.LINES)
        while world.tick < 6:
            world.frame()
            self.assertEqual(world.core.debug["lane_change"], "pending")
            self.assertAlmostEqual(world.last["steer"], 0.0, delta=0.01)
        for _ in range(20):
            world.frame()
        self.assertEqual(world.core.debug["lane_change"], "active")
        self.assertAlmostEqual(world.core.lane_change["width_m"], 3.5, delta=0.1)
        self.assertLess(world.last["steer"], -0.01)  # to the left
        # 1 m to the left: the old lane's centre is now to the right. Without
        # the shift the steering would go back to it.
        self.moved(world, 0.0, 1.0)
        for _ in range(90):
            world.frame()
        self.assertEqual(world.core.debug["lane_change"], "active")
        self.assertAlmostEqual(world.core.lane_change["shift_m"], 3.5, delta=0.1)
        self.assertLess(world.last["steer"], -0.01)
        plain = World()
        plain.lane = lane_envelope(offset=1.0)
        plain.running().frame()
        self.assertGreater(plain.last["steer"], 0.0)
        # Over the marking: the markings either side are now the next lane's
        self.moved(world, 1.0, 1.7)
        world.lane = lane_envelope(offset=1.8, markings=self.LINES)
        world.frame()
        self.assertEqual(world.core.debug["lane_change"], "done")
        self.assertEqual(world.core.state, RUNNING)
        # From here on it is ordinary lane keeping, on the new lane
        self.moved(world, 1.8, 3.5)
        for _ in range(60):
            world.frame()
        self.assertAlmostEqual(world.last["steer"], 0.0, delta=0.01)
        self.assertEqual(world.core.lane_change["state"], "done")
        self.assertEqual(world.core.state, RUNNING)

    def test_on_the_marking_the_device_has_no_lane_and_the_vehicle_keeps_going(self):
        # As the lane model behaves: while the vehicle crosses, the marking
        # beyond is gone, and the one being crossed is on neither side
        world = self.world(self.LINES)
        for _ in range(20):
            world.frame()
        self.moved(world, 0.0, 1.5)
        self.moved(world, 1.5, 1.7, markings=(-1.75, 1.75))
        # Over it. The vehicle stands still in this test, so the marking
        # stays where the last result before had it, whatever comes now
        world.lane = lane_envelope(offset=2.0, markings=(-1.75, 1.75))
        for _ in range(10):
            world.frame()
        self.assertFalse(world.lane["valid"])
        self.assertEqual(world.lane["reason"], "missing_boundary")
        self.assertEqual(world.core.state, RUNNING)
        self.assertEqual(world.last["reason"], "")
        self.assertEqual(world.core.debug["lane_change"], "active")
        self.assertEqual(world.core.lane_bad, 0)
        self.assertLess(world.last["steer"], -0.01)
        # The device reports a lane again, with that marking on its right
        world.lane = lane_envelope(offset=1.8, markings=self.LINES)
        world.frame()
        self.assertEqual(world.core.debug["lane_change"], "done")

    def test_losing_the_marking_being_crossed_is_losing_the_lane(self):
        world = self.world(self.LINES)
        for _ in range(20):
            world.frame()
        self.moved(world, 0.0, 1.5)
        world.lane = dict(lane_envelope(markings=(1.75,)), pixels=[])
        # Where the last result that had it as the lane's boundary puts it
        # is good for coast_s, 2 s
        for _ in range(60):
            world.frame()
            self.assertEqual(world.core.state, RUNNING)
            self.assertEqual(world.last["reason"], "")
        world.frame()
        self.assertEqual(world.core.state, HOLDING)
        self.assertEqual(world.last["reason"], "lane:marking_lost")
        self.assertEqual(world.last["brake"], 1.0)
        for _ in range(LANE_LATCH_FRAMES):
            world.frame()
        self.assertEqual(world.core.state, STOP_LATCHED)

    def test_a_mixed_up_marking_does_not_take_over(self):
        # The lane model, changing the slot of the marking being crossed,
        # reports a line between it and its neighbour for a few results
        world = self.world(self.LINES)
        for _ in range(20):
            world.frame()
        self.moved(world, 0.0, 1.6)
        for _ in range(60):
            world.frame()
        steer = world.last["steer"]
        # A "lane" 3.5 m wide whose boundaries are neither where the marking is
        world.lane = lane_envelope(markings=(-1.75 + 1.6 - 1.1, 1.75 + 1.6 - 1.1))
        self.assertTrue(world.lane["valid"])
        for _ in range(6):
            world.frame()
        self.assertEqual(world.core.debug["lane_change"], "active")
        self.assertEqual(world.last["reason"], "")
        self.assertAlmostEqual(world.last["steer"], steer, delta=0.01)
        world.lane = lane_envelope(offset=1.8, markings=self.LINES)
        world.frame()
        self.assertEqual(world.core.debug["lane_change"], "done")

    def test_to_the_right(self):
        world = self.world((-1.75, 1.75, 5.25), side="right")
        for _ in range(30):
            world.frame()
        self.assertEqual(world.core.debug["lane_change"], "active")
        self.assertGreater(world.last["steer"], 0.01)

    def test_without_the_next_marking_it_is_skipped_and_the_lane_is_kept(self):
        for markings, pixels in (((-1.75, 1.75), None), (self.LINES, [])):
            world = self.world(markings)
            if pixels is not None:
                world.lane["pixels"] = pixels  # the device sent no pixels
            for _ in range(80):
                world.frame()
                self.assertAlmostEqual(world.last["steer"], 0.0, delta=0.01)
            self.assertEqual(world.core.debug["lane_change"], "skipped")
            self.assertEqual(world.core.lane_change["reason"], "no_marking")
            self.assertEqual(world.core.state, RUNNING)
        # The tick it gave up at carries the flag
        world = self.world((-1.75, 1.75), at_tick=6)
        flagged = []
        for _ in range(60):
            world.frame()
            flagged.append(world.core.debug.get("lane_change_skipped"))
        self.assertEqual([f for f in flagged if f], ["no_marking"])

    def test_a_road_user_in_the_next_lane_keeps_it_in_its_own(self):
        world = self.world(self.LINES)
        world.objects = [("car", 14.0, -4.4, -2.6, 1.5, 3)]
        for _ in range(80):
            world.frame()
        self.assertEqual(world.core.lane_change["reason"], "lane_occupied")
        self.assertAlmostEqual(world.last["steer"], 0.0, delta=0.01)

    def test_a_marking_that_is_too_far_is_not_the_next_lane(self):
        world = self.world((-7.5, -1.75, 1.75))  # 5.75 m: not a lane's width
        for _ in range(80):
            world.frame()
        self.assertEqual(world.core.lane_change["reason"], "no_marking")

    def test_gives_up_when_the_pair_never_changes(self):
        world = self.world(self.LINES, timeout_ticks=40)
        for _ in range(80):
            world.frame()
        self.assertEqual(world.core.lane_change["reason"], "timeout")
        self.assertEqual(world.core.debug["lane_change"], "skipped")
        self.assertEqual(world.core.state, RUNNING)
        for _ in range(30):  # back to its own lane, which is where it still is
            world.frame()
        self.assertAlmostEqual(world.last["steer"], 0.0, delta=0.01)


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
        self.assertEqual(command["valid_until_sim_ns"], self.basis_ns + AGE_MS * MS)

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
        self.assertEqual(command["brake"], 1.0)
        self.assertEqual(world.core.state, HOLDING)

    def test_a_pose_that_is_not_usable(self):
        world = self.world
        world.tick += 1
        ego = world.ego()
        ego["pose"]["yaw"] = float("nan")
        world.core.on_ego(ego)
        self.assertEqual(world.core.step(ego)["reason"], "bad_ego_state")


if __name__ == "__main__":
    unittest.main()
