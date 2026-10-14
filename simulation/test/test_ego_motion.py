"""Dead reckoning: degrees at the boundary, radians inside."""

import math
import os
import unittest

import yaml

from video_analytics_simulation.calibration import Pose
from video_analytics_simulation.ego_motion import EgoMotion

DT = 1.0 / 30.0
NS = int(DT * 1e9)
ZERO = {"speed_scale": 1.0, "speed_noise_mps": 0.0,
        "yaw_rate_bias_radps": 0.0, "yaw_rate_noise_radps": 0.0, "seed": 1}


def step(motion, i, speed, yaw_rate=0.0, reverse=False):
    return motion.accept({"sim_ns": i * NS, "speed_mps": speed,
                          "yaw_rate_radps": yaw_rate}, reverse)


class EgoMotionTest(unittest.TestCase):
    def test_straight_ten_seconds_at_five_metres_per_second(self):
        motion = EgoMotion(ZERO, speed_cap_mps=20.0)
        motion.seed(Pose(0, 0, 0), 0)
        for i in range(1, 301):
            self.assertEqual(step(motion, i, 5.0), "")
        self.assertAlmostEqual(motion.pose.x, 50.0, delta=0.01)
        self.assertAlmostEqual(motion.pose.y, 0.0, delta=0.01)
        self.assertAlmostEqual(motion.pose.yaw, 0.0, delta=0.01)

    def test_seeded_at_minus_ninety_degrees_goes_west(self):
        motion = EgoMotion(ZERO, speed_cap_mps=20.0)
        motion.seed(Pose(0, 0, 0.157, yaw=-90.0), 0)
        for i in range(1, 301):
            self.assertEqual(step(motion, i, 2.0), "")
        self.assertAlmostEqual(motion.pose.x, 0.0, delta=0.02)
        self.assertAlmostEqual(motion.pose.y, -20.0, delta=0.02)
        self.assertAlmostEqual(motion.pose.yaw, -90.0, delta=0.05)
        self.assertAlmostEqual(motion.pose.z, 0.157)

    def test_one_percent_scale_over_one_hundred_metres(self):
        motion = EgoMotion(dict(ZERO, speed_scale=1.01), speed_cap_mps=20.0)
        motion.seed(Pose(0, 0, 0), 0)
        for i in range(1, 601):
            step(motion, i, 5.0)
        self.assertAlmostEqual(motion.pose.x, 101.0, delta=0.05)

    def test_reverse_is_negative(self):
        motion = EgoMotion(ZERO, speed_cap_mps=20.0)
        motion.seed(Pose(0, 0, 0), 0)
        for i in range(1, 31):
            step(motion, i, 1.0, reverse=True)
        self.assertAlmostEqual(motion.pose.x, -1.0, delta=0.01)

    def test_rejects_a_bad_sample_without_moving(self):
        motion = EgoMotion(ZERO, speed_cap_mps=10.0)
        motion.seed(Pose(0, 0, 0), 0)
        self.assertEqual(step(motion, 1, 1.0), "")
        x = motion.pose.x
        self.assertEqual(step(motion, 1, 1.0), "ego_not_newer")
        self.assertEqual(motion.accept(
            {"sim_ns": NS + int(0.2e9), "speed_mps": 1.0, "yaw_rate_radps": 0.0},
            False), "ego_dt")
        self.assertEqual(motion.accept(
            {"sim_ns": 2 * NS, "speed_mps": -1.0, "yaw_rate_radps": 0.0},
            False), "ego_speed")
        self.assertEqual(motion.accept(
            {"sim_ns": 2 * NS, "speed_mps": 1.0, "yaw_rate_radps": 10.0},
            False), "ego_yaw_rate")
        self.assertAlmostEqual(motion.pose.x, x)

    def test_yaml_locks_the_five_odometry_numbers(self):
        path = os.path.join(os.path.dirname(__file__), "..", "config", "carla.yaml")
        with open(path) as f:
            odom = yaml.safe_load(f)["control"]["odometry"]
        self.assertEqual(odom, {
            "speed_scale": 1.015,
            "speed_noise_mps": 0.03,
            "yaw_rate_bias_radps": 0.001745,
            "yaw_rate_noise_radps": 0.003491,
            "seed": 1,
        })
        self.assertAlmostEqual(
            math.degrees(odom["yaw_rate_bias_radps"]), 0.1, delta=0.001)


if __name__ == "__main__":
    unittest.main()
