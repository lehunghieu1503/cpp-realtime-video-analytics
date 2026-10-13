import os
import sys
import unittest

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import fixtures  # noqa: E402
from video_analytics_simulation import viewer  # noqa: E402
from video_analytics_simulation.calibration import Pose, Rig  # noqa: E402
from video_analytics_simulation.parking import Stall  # noqa: E402

with open(os.path.join(os.path.dirname(__file__), "..", "config", "carla.yaml")) as f:
    CONFIG = yaml.safe_load(f)
FRONT = Rig.from_config(CONFIG["calibration"])
REAR = Rig.from_config(CONFIG["rear_calibration"])


class Window(unittest.TestCase):
    """The window's picture, without a screen: what cv2 would show is kept."""

    def setUp(self):
        self.shown = []
        self.key = -1
        self.real = {name: getattr(cv2, name) for name in (
            "namedWindow", "imshow", "waitKey", "getWindowProperty", "destroyWindow")}
        cv2.namedWindow = lambda *a: None
        cv2.imshow = lambda name, image: self.shown.append(image.copy())
        cv2.waitKey = lambda ms: self.key
        cv2.getWindowProperty = lambda *a: 1.0
        cv2.destroyWindow = lambda *a: None
        self.pose = Pose(7.2, 12.0, 0.0, 0.0, 0.0, 90.0)
        self.view = viewer.Viewer({"front": FRONT, "rear": REAR}, lambda ns: self.pose)

    def tearDown(self):
        for name, function in self.real.items():
            setattr(cv2, name, function)

    def test_cameras_side_by_side_with_what_was_found_in_them(self):
        view = self.view
        grey = np.full((720, 1280, 4), 90, np.uint8)
        view.frame("front", 100, grey)
        view.frame("rear", 100, grey)
        self.assertTrue(view.show(0, 0.0, {}))
        raw = self.shown[-1]
        self.assertEqual(raw.shape, (360 + 28, 1280, 3))
        self.assertTrue((raw[:360] == 90).all())  # no result yet: the images as they are

        stall = Stall(0.0, 2.8, 0.0, 5.0, 2.8)
        found = {"simulation": {"sim_ns": 100}, "detections": [],
                 "stalls": fixtures.stall_detections([stall], self.pose, REAR),
                 "stall_points": fixtures.stall_points([stall], self.pose, REAR)}
        view.result("front", {
            "simulation": {"sim_ns": 100},
            "detections": [{"class": "car", "confidence": 0.9, "track_id": 3,
                            "bbox": {"x": 600, "y": 300, "w": 80, "h": 60}}],
            "lane": {"pixels": [[[500, 700], [600, 400]], [[800, 700], [700, 400]]]}})
        view.result("rear", found)
        view.show(6, 0.0, {})
        front, rear = self.shown[-1][:360, :640], self.shown[-1][:360, 640:]
        self.assertGreater((front != 90).any(axis=2).sum(), 200)  # the box, two lane lines
        # The stall the model found, green for an empty one, and over its
        # corners the ends of its dividers: crosses
        self.assertGreater((rear == (0, 255, 0)).all(axis=2).sum(), 5)
        self.assertGreater((rear == (255, 255, 0)).all(axis=2).sum(), 5)
        self.assertEqual((rear == (0, 255, 255)).all(axis=2).sum(), 0)

        # The controller follows it and takes it: its outline, where the
        # controller puts it, in yellow
        view.status({"state": "RUNNING", "stop_reason": "", "park": {
            "phase": "back", "target": 4,
            "stalls": [dict(stall.as_dict(), id=4,
                            corners=[[float(x), float(y)] for x, y, _ in stall.corners()])]}})
        view.result("rear", found)
        self.assertTrue(view.show(7, 0.4, {"reverse": True, "throttle": 0.2, "brake": 0.0,
                                           "steer": 0.45, "reason": ""}))
        drawn = self.shown[-1]
        self.assertGreater((drawn[:360, 640:] == (0, 255, 255)).all(axis=2).sum(), 200)
        self.assertGreater((drawn[360:] > 0).sum(), 500)  # the line of text

    def test_the_view_from_above_is_a_third_panel(self):
        from video_analytics_simulation.bev import BevRig
        above = BevRig.from_config(CONFIG["bev"])
        view = viewer.Viewer({"front": FRONT, "rear": REAR, "bev": above},
                             lambda ns: self.pose)
        view.frame("front", 100, np.full((720, 1280, 4), 90, np.uint8))
        view.frame("rear", 100, np.full((720, 1280, 4), 90, np.uint8))
        view.frame("bev", 100, np.full((800, 800, 3), 90, np.uint8))
        # A stall 3 m to the left of the vehicle, as the device finds it in the view
        stall = Stall(self.pose.x + 5.5, self.pose.y, 0.0, 5.0, 2.8)
        view.status({"state": "RUNNING", "stop_reason": "", "park": {
            "phase": "search", "target": 4,
            "stalls": [dict(stall.as_dict(), id=4,
                            corners=[[float(x), float(y)] for x, y, _ in stall.corners()])]}})
        view.result("bev", {"simulation": {"sim_ns": 100}, "detections": [],
                            "stalls": fixtures.stall_detections([stall], self.pose, above),
                            "stall_points": fixtures.stall_points([stall], self.pose, above)})
        self.assertTrue(view.show(3, 0.0, {}))
        shown = self.shown[-1]
        # Two cameras 360 high and the view 400 high, side by side, and the text
        self.assertEqual(shown.shape, (400 + 28, 640 + 640 + 400, 3))
        self.assertEqual(int(shown[360:400, :1280].sum()), 0)
        panel = shown[:400, 1280:]
        # Its ends as crosses, and the stall the controller takes in yellow,
        # left of the middle: the stall is on the vehicle's left
        self.assertGreater((panel == (255, 255, 0)).all(axis=2).sum(), 5)
        yellow = np.argwhere((panel == (0, 255, 255)).all(axis=2))
        self.assertGreater(len(yellow), 100)
        self.assertLess(yellow[:, 1].max(), 200)

    def test_a_result_of_a_frame_no_longer_kept_changes_nothing(self):
        view = self.view
        view.frame("front", 100, np.zeros((720, 1280, 4), np.uint8))
        view.result("front", {"simulation": {"sim_ns": 999}, "detections": [
            {"class": "car", "confidence": 0.9, "bbox": {"x": 0, "y": 0, "w": 500, "h": 300}}]})
        view.show(0, 0.0, {})
        self.assertEqual(int(self.shown[-1][:360].sum()), 0)
        for sim_ns in range(viewer.KEPT + 5):
            view.frame("front", sim_ns, np.zeros((720, 1280, 4), np.uint8))
        self.assertEqual(len(view.images["front"]), viewer.KEPT)

    def test_q_closes_the_window_and_the_run_goes_on_without_it(self):
        self.view.frame("front", 1, np.zeros((720, 1280, 4), np.uint8))
        self.key = ord("q")
        self.assertFalse(self.view.show(0, 0.0, {}))
        shown = len(self.shown)
        self.assertFalse(self.view.show(1, 0.0, {}))
        self.assertEqual(len(self.shown), shown)
        self.view.pump()
        self.view.close()


if __name__ == "__main__":
    unittest.main()
