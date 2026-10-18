import json
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import fixtures  # noqa: E402
from video_analytics_simulation import lane  # noqa: E402
from video_analytics_simulation.calibration import (  # noqa: E402
    Pose, Rig, decode_depth, right_to_left, rotation)


def rig(**changes):
    config = json.loads(json.dumps(fixtures.CALIBRATION))
    config.update(changes)
    return Rig.from_config(config)


class Calibration(unittest.TestCase):
    def test_intrinsics(self):
        r = rig()
        self.assertAlmostEqual(r.focal, 1280 / (2 * math.tan(math.radians(26))))
        np.testing.assert_allclose(r.K[:2, 2], [640, 360])
        self.assertEqual(r.K[0, 0], r.K[1, 1])

    def test_axes_and_signs(self):
        r = rig(camera={"x": 1.5, "y": 0.0, "z": 1.6, "pitch_deg": 0.0})
        # Ahead on the optical axis: image centre
        uv, depth = r.project([[21.5, 0.0, 1.6]])
        np.testing.assert_allclose(uv, [[640, 360]])
        np.testing.assert_allclose(depth, [20.0])
        # To the right is a larger u, lower is a larger v
        self.assertGreater(r.project([[21.5, 2.0, 1.6]])[0][0, 0], 640)
        self.assertGreater(r.project([[21.5, 0.0, 0.0]])[0][0, 1], 360)
        # Behind the camera: no pixel
        uv, depth = r.project([[0.0, 0.0, 1.6]])
        self.assertTrue(depth[0] <= 0 and np.isnan(uv).all())
        # A camera pitched up sees the same point lower in the image
        up = rig(camera={"x": 1.5, "y": 0.0, "z": 1.6, "pitch_deg": 5.0})
        self.assertGreater(up.project([[21.5, 0.0, 1.6]])[0][0, 1], 360)
        # A positive yaw turns right: forward becomes +y
        np.testing.assert_allclose(rotation(0, 0, 90) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(right_to_left([1.0, 2.0, 3.0]), [1, -2, 3])

    def test_ground_round_trip(self):
        r = rig()
        points = np.array([[10.0, 0.0, 0.0], [20.0, -3.5, 0.0], [60.0, 5.0, 0.0],
                           [110.0, 1.0, 0.0]])
        uv, _ = r.project(points)
        xy, valid = r.uv_to_ground(uv[:, 0], uv[:, 1])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(xy, points[:, :2], atol=1e-6)

    def test_ground_rejects(self):
        r = rig()
        horizon_v = 360 + r.focal * math.tan(math.radians(3.0))
        _, valid = r.uv_to_ground([640, 640, 640, -1, 1280, 640, float("nan")],
                                  [horizon_v - 1, horizon_v + 0.2, 0, 700, 700, 720, 700])
        # above the horizon, just below it (beyond the range), sky, outside x4
        self.assertFalse(valid.any())
        xy, valid = r.uv_to_ground(640, 719)
        self.assertTrue(valid)
        self.assertGreater(xy[0], 0)

    def test_depth_decoding(self):
        raw = np.zeros((1, 4, 4), np.uint8)
        raw[0, 0] = [0, 0, 0, 255]          # B, G, R, A: nearest
        raw[0, 1] = [255, 255, 255, 255]    # farthest
        raw[0, 2] = [0, 1, 0, 255]          # G = 1: 256 units
        raw[0, 3] = [2, 0, 200, 255]        # needs more than 16 bits
        depth = decode_depth(raw)
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_allclose(
            depth[0], [0, 1000, 256 / 16777215 * 1000,
                       (200 + 2 * 65536) / 16777215 * 1000], rtol=1e-6)
        with self.assertRaises(ValueError):
            decode_depth(np.zeros((2, 2), np.uint8))
        with self.assertRaises(ValueError):
            decode_depth(np.zeros((2, 2, 4), np.float32))

    def test_depth_along_axis_or_along_ray(self):
        r = rig(camera={"x": 1.5, "y": 0.0, "z": 1.6, "pitch_deg": 0.0})
        # A wall 10 m ahead of the camera: off axis the two readings differ
        point = np.array([11.5, 4.0, 1.6])
        uv, depth = r.project(point[None])
        np.testing.assert_allclose(r.depth_to_ego(uv[:, 0], uv[:, 1], depth), point[None],
                                   atol=1e-9)
        as_ray = r.depth_to_ego(uv[:, 0], uv[:, 1], depth, radial=True)
        self.assertLess(as_ray[0, 0], 11.3)

    def test_rig_validation(self):
        for bad in ({"fov_deg": float("nan")}, {"width": 0}, {"fov_deg": 0.5},
                    {"ground_z": 2.0}, {"distortion": [0.1, 0, 0, 0, 0]}):
            with self.assertRaises(ValueError):
                rig(**bad)
        self.assertEqual(rig(distortion=[0, 0, 0, 0, 0]).calibration_id,
                         rig().calibration_id)
        self.assertNotEqual(rig(fov_deg=52.001).calibration_id, rig().calibration_id)
        with self.assertRaises(ValueError):
            Pose.from_dict({"x": 0, "y": 0, "z": float("inf"), "roll": 0, "pitch": 0,
                            "yaw": 0})

    def test_pose_round_trip(self):
        pose = Pose(10, -4, 0.5, 1.0, -2.0, 37.0)
        points = np.array([[1.0, 2.0, 3.0], [-5.0, 0.5, 0.0]])
        np.testing.assert_allclose(pose.from_parent(pose.to_parent(points)), points,
                                   atol=1e-12)
        R = pose.rotation
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


class Lane(unittest.TestCase):
    def test_fixture_file_matches_the_reference(self):
        with open(fixtures.PATH) as f:
            stored = json.load(f)
        built = json.loads(json.dumps(fixtures.build()))

        def compare(a, b, path=""):
            if isinstance(a, dict):
                self.assertEqual(set(a), set(b), path)
                for k in a:
                    compare(a[k], b[k], path + "/" + k)
            elif isinstance(a, list):
                self.assertEqual(len(a), len(b), path)
                for i, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, "%s[%d]" % (path, i))
            elif isinstance(a, float):
                self.assertAlmostEqual(a, b, places=6, msg=path)
            else:
                self.assertEqual(a, b, path)
        compare(stored, built)

    def test_decoder_details(self):
        case = fixtures.DECODER_CASES[0]
        points = lane.decode_ufld(fixtures.synthetic_logits(case), 1280, 720)
        # Rows 0..5 have no lane; rows are reversed, so the first point is the
        # bottom row anchor (284 of 288)
        self.assertEqual(len(points[0]), 50)
        self.assertEqual(points[0][0][1], int(720 * 284 / 288) - 1)
        # Column counted from 1: a peak at bin 20 + 0.45 * 55 is 45.75 + 1
        self.assertEqual(points[0][0][0], int((20 + 0.45 * 55 + 1) * (799 / 99) * 1280 / 800) - 1)
        self.assertEqual(points[3], [])
        with self.assertRaises(ValueError):
            lane.decode_ufld(np.zeros((101, 56, 3)), 1280, 720)
        bad = fixtures.synthetic_logits(case)
        bad[3, 3, 0] = np.nan
        with self.assertRaises(ValueError):
            lane.decode_ufld(bad, 1280, 720)

    def test_lane_signs(self):
        r = rig()
        px = [fixtures.lane_pixels(r, o) for o in (-1.35, 2.15)]
        result = lane.estimate_lane(px, r)
        # The lane centre is 0.4 m to the right of the vehicle
        self.assertAlmostEqual(result["lateral_error_m"], 0.4, delta=0.02)
        self.assertAlmostEqual(result["heading_rad"], 0.0, delta=0.005)
        px = [fixtures.lane_pixels(r, o, 0.004) for o in (-1.75, 1.75)]
        self.assertGreater(lane.estimate_lane(px, r)["curvature_inv_m"], 0.006)

    def test_rebase_after_moving(self):
        r = rig()
        px = [fixtures.lane_pixels(r, o) for o in (-1.75, 1.75)]
        seen = lane.estimate_lane(px, r)
        capture = Pose(100, 50, 0, 0, 0, 30)
        same = lane.rebase_lane(seen, capture, capture)
        self.assertTrue(same["valid"])
        self.assertAlmostEqual(same["lateral_error_m"], 0.0, delta=1e-6)

        # 3 m forward, 0.5 m to the right and turned 4 degrees right since
        moved = Pose(*capture.to_parent([3.0, 0.5, 0.0]), 0, 0, 34)
        now = lane.rebase_lane(seen, capture, moved)
        self.assertTrue(now["valid"])
        # The lane centre is now to the left and heads to the left
        x = now["reference_x_m"]
        expected = -0.5 * math.cos(math.radians(4)) - (x) * math.tan(math.radians(4))
        self.assertAlmostEqual(now["heading_rad"], -math.radians(4), delta=1e-3)
        self.assertAlmostEqual(now["lateral_error_m"], expected, delta=0.05)
        self.assertLess(now["centerline"]["x_max_m"], seen["centerline"]["x_max_m"] - 2.5)

        # Far past what was seen: nothing left to steer on
        gone = Pose(*capture.to_parent([40.0, 0.0, 0.0]), 0, 0, 30)
        self.assertFalse(lane.rebase_lane(seen, capture, gone)["valid"])
        self.assertFalse(lane.rebase_lane({"valid": False}, capture, moved)["valid"])
        self.assertFalse(lane.rebase_lane(None, capture, moved)["valid"])


class VisibleBox(unittest.TestCase):
    """The box of what the camera sees of a polygon on the ground."""

    def test_same_box_as_cutting_the_projected_polygon_with_opencv(self):
        import cv2
        rig = Rig(1280, 720, 120.0, -2.45, 0.0, 1.0, pitch_deg=-30.0, yaw_deg=180.0)
        rng = np.random.default_rng(3)
        image = np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], np.float32)
        seen = 0
        for _ in range(300):
            # A 5 x 2.8 m rectangle on the ground, anywhere behind the vehicle
            centre, turn = rng.uniform([-16.0, -9.0], [-2.6, 9.0]), rng.uniform(0, np.pi)
            along = np.array([np.cos(turn), np.sin(turn)])
            across = np.array([-along[1], along[0]])
            ring = np.array([np.append(centre + a * 2.5 * along + b * 1.4 * across, 0.0)
                             for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
            box = rig.visible_box(ring)
            camera = rig.pose.from_parent(ring)
            if (camera[:, 0] < 0.2).any():
                continue  # the reference below has no near plane
            uv, _ = rig.project(ring)
            area, inside = cv2.intersectConvexConvex(uv.astype(np.float32), image)
            if area <= 1.0:
                self.assertTrue(box is None or (box[2] - box[0]) * (box[3] - box[1]) < 30.0)
                continue
            inside = inside.reshape(-1, 2)
            np.testing.assert_allclose(box, [*inside.min(axis=0), *inside.max(axis=0)],
                                       atol=0.05)
            seen += 1
        self.assertGreater(seen, 100)

    def test_a_polygon_through_the_camera_plane_and_one_behind_it(self):
        rig = Rig(1280, 720, 120.0, -2.45, 0.0, 1.0, pitch_deg=-30.0, yaw_deg=180.0)
        # From 3 m behind the vehicle to 3 m ahead of it: what is behind the
        # camera is cut off, the rest reaches the lower edge of the image
        through = np.array([[-5.45, 1.4, 0.0], [-5.45, -1.4, 0.0], [0.55, -1.4, 0.0],
                            [0.55, 1.4, 0.0]])
        box = rig.visible_box(through)
        self.assertIsNotNone(box)
        self.assertEqual(box[3], 720.0)
        self.assertTrue(0.0 <= box[0] < box[2] <= 1280.0)
        ahead = through + [8.0, 0.0, 0.0]
        self.assertIsNone(rig.visible_box(ahead))


if __name__ == "__main__":
    unittest.main()
