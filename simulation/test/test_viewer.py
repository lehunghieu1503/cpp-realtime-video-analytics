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
        self.assertGreater((front != 90).any(axis=2).sum(), 200)  # the box and its label
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

    def test_lane_lines_are_drawn_on_the_road_and_not_in_a_car_park(self):
        result = {"simulation": {"sim_ns": 1}, "detections": [],
                  "lane": {"pixels": [[[500, 700], [600, 400]], [[800, 700], [700, 400]]]}}
        grey = np.full((720, 1280, 4), 90, np.uint8)
        # This viewer has a rear camera: the scenario parks
        self.view.frame("front", 1, grey)
        self.view.result("front", result)
        self.view.show(0, 0.0, {})
        self.assertTrue((self.shown[-1][:360, :640] == 90).all())
        road = viewer.Viewer({"front": FRONT}, lambda ns: self.pose)
        road.frame("front", 1, grey)
        road.result("front", result)
        road.show(0, 0.0, {})
        self.assertGreater((self.shown[-1][:360] == (0, 255, 255)).all(axis=2).sum(), 100)

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
