import math
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
            "namedWindow", "imshow", "waitKey", "getWindowProperty", "destroyWindow",
            "resizeWindow")}
        cv2.namedWindow = lambda *a: None
        cv2.resizeWindow = lambda *a: None
        cv2.imshow = lambda name, image: self.shown.append(image.copy())
        cv2.waitKey = lambda ms: self.key
        cv2.getWindowProperty = lambda *a: 1.0
        cv2.destroyWindow = lambda *a: None
        self.pose = Pose(7.2, 12.0, 0.0, 0.0, 0.0, 90.0)
        self.view = viewer.Viewer({"front": FRONT, "rear": REAR}, lambda ns: self.pose)

    def tearDown(self):
        for name, function in self.real.items():
            setattr(cv2, name, function)

    def test_front_camera_big_the_others_under_it_with_what_was_found_in_them(self):
        view = self.view
        grey = np.full((720, 1280, 4), 90, np.uint8)
        view.frame("front", 100, grey)
        view.frame("rear", 100, grey)
        self.assertTrue(view.show(0, 0.0, {}))
        raw = self.shown[-1]
        # The front camera 640 wide, the rear a quarter of that under it, the
        # map right of them, the text
        self.assertEqual(raw.shape, (360 + 90 + 46, 640 + 450, 3))
        self.assertTrue((raw[:450, :160] == 90).all())  # no result yet: the images as they are
        self.assertEqual(int(raw[360:450, 160:640].sum()), 0)

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
        front, rear = self.shown[-1][:360], view.panels["rear"]
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
        self.assertGreater((view.panels["rear"] == (0, 255, 255)).all(axis=2).sum(), 200)
        self.assertGreater((self.shown[-1][450:] > 0).sum(), 500)  # the lines of text

    def test_the_view_from_above_is_right_of_the_cameras(self):
        from video_analytics_simulation.bev import BevRig
        above = BevRig.from_config(CONFIG["bev"])
        view = viewer.Viewer({"front": FRONT, "rear": REAR, "bev": above},
                             lambda ns: None, vehicle={
                                 "width_m": 2.16, "front_bumper_m": 2.4, "rear_bumper_m": -2.4})
        view.frame("front", 100, np.full((720, 1280, 4), 90, np.uint8))
        view.frame("rear", 100, np.full((720, 1280, 4), 90, np.uint8))
        view.frame("bev", 100, np.full((800, 800, 3), 90, np.uint8))
        # A stall 3 m to the left of the vehicle, as the device finds it in the view
        stall = Stall(self.pose.x + 5.5, self.pose.y, 0.0, 5.0, 2.8)
        # The pose is the controller's own: the simulator's is not asked for
        view.status({"state": "RUNNING", "stop_reason": "", "park": {
            "phase": "search", "target": 4,
            "pose": {"x": self.pose.x, "y": self.pose.y, "z": 0.0, "roll": 0.0,
                     "pitch": 0.0, "yaw": self.pose.yaw},
            "stalls": [dict(stall.as_dict(), id=4,
                            corners=[[float(x), float(y)] for x, y, _ in stall.corners()])]}})
        view.result("bev", {"simulation": {"sim_ns": 100}, "detections": [],
                            "stalls": fixtures.stall_detections([stall], self.pose, above),
                            "stall_points": fixtures.stall_points([stall], self.pose, above)})
        self.assertTrue(view.show(3, 0.0, {}))
        shown = self.shown[-1]
        # The cameras 640 wide, the view and the map under it as high as they
        # are right of them, the text
        self.assertEqual(shown.shape, (360 + 90 + 46, 640 + 225, 3))
        self.assertGreater((shown[:450, 640:] == (120, 70, 40)).all(axis=2).sum(), 100)
        panel = view.with_vehicle("bev", view.panels["bev"])
        # Its ends as crosses, and the stall the controller takes in yellow,
        # left of the middle: the stall is on the vehicle's left
        self.assertGreater((panel == (255, 255, 0)).all(axis=2).sum(), 5)
        yellow = np.argwhere((panel == (0, 255, 255)).all(axis=2))
        self.assertGreater(len(yellow), 100)
        self.assertLess(yellow[:, 1].max(), 200)
        # The vehicle itself, in the middle, nose up
        self.assertTrue((panel[200, 200] == (120, 70, 40)).all())
        self.assertTrue((panel[165, 200] == (255, 255, 255)).all())
        # The map: the stall it parks in blue, left of the vehicle in its middle
        drawn = view.map()
        blue = np.argwhere((np.abs(drawn.astype(int) - (144, 93, 36)) < 4).all(axis=2))
        self.assertGreater(len(blue), 1000)
        self.assertLess(blue[:, 1].max(), 300)
        self.assertTrue((drawn[300, 300] == (120, 70, 40)).all())
        # A stall never seen whole is not on it
        park = dict(view.park, target=None)
        view.status({"park": dict(park, stalls=[dict(park["stalls"][0], id=5, hits=0)])})
        self.assertEqual(len(np.argwhere((np.abs(view.map().astype(int) - (144, 93, 36)) < 4)
                                         .all(axis=2))), 0)
        self.assertEqual(int((view.map() == 235).all(axis=2).sum()), 0)
        # Of one stall followed twice, the one seen more often only
        shifted = Stall(stall.centre[0] + 0.5, stall.centre[1], 0.0, 5.0, 2.8)
        rows = [dict(s.as_dict(), id=key, hits=hits, corners=[
            [float(x), float(y)] for x, y, _ in s.corners()])
            for s, key, hits in ((stall, 6, 20), (shifted, 7, 50))]
        view.status({"park": dict(park, stalls=rows)})
        self.assertEqual(view.drawn, {7})
        rows[1]["corners"] = [[x + 10.0, y] for x, y in rows[1]["corners"]]
        view.status({"park": dict(park, stalls=rows)})
        self.assertEqual(view.drawn, {6, 7})
        # Seen again after the pose drifted: the one seen last, though seen less
        drifted = Stall(stall.centre[0] + 1.7, stall.centre[1] + 1.1, 0.0, 5.0, 2.8)
        rows = [dict(s.as_dict(), id=key, hits=hits, seen_ns=seen, corners=[
            [float(x), float(y)] for x, y, _ in s.corners()])
            for s, key, hits, seen in ((stall, 8, 900, 10**9), (drifted, 9, 40, 9 * 10**9))]
        view.status({"park": dict(park, stalls=rows)})
        self.assertEqual(view.drawn, {9})

    def test_with_the_image_from_behind_it_is_big_and_every_camera_under_it(self):
        from video_analytics_simulation.bev import BevRig
        rigs = {"front": FRONT, "rear": REAR, "left": REAR, "right": REAR,
                "bev": BevRig.from_config(CONFIG["bev"])}
        view = viewer.Viewer(rigs, lambda ns: None)
        view.chase_frame(np.full((720, 1280, 4), 200, np.uint8))
        view.show(1, 0.0, {})
        shown = self.shown[-1]
        # 1280x720 behind, four cameras of 320x180 under it, the view and the
        # map 450 wide: a 1920x1080 screen holds it
        self.assertEqual(shown.shape, (720 + 180 + 46, 1280 + 450, 3))
        self.assertTrue((shown[:720, :1280] == 200).all())

    def test_a_result_of_a_frame_no_longer_kept_changes_nothing(self):
        view = self.view
        view.frame("front", 100, np.zeros((720, 1280, 4), np.uint8))
        view.result("front", {"simulation": {"sim_ns": 999}, "detections": [
            {"class": "car", "confidence": 0.9, "bbox": {"x": 0, "y": 0, "w": 500, "h": 300}}]})
        view.show(0, 0.0, {})
        self.assertEqual(int(self.shown[-1][:360, :640].sum()), 0)
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

    def test_a_window_qt_has_lost_is_closed_without_an_error(self):
        def lost(*a):
            raise cv2.error("NULL guiReceiver")
        cv2.getWindowProperty = cv2.destroyWindow = lost
        self.view.frame("front", 1, np.zeros((720, 1280, 4), np.uint8))
        self.assertFalse(self.view.show(0, 0.0, {}))
        self.view.close()

    def test_the_vehicle_from_above_in_place_of_its_outline(self):
        from video_analytics_simulation.bev import BevRig
        vehicle = {"width_m": 2.0, "front_bumper_m": 2.4, "rear_bumper_m": -2.4}
        # A camera 29.25 m above, 12 degrees: 0.012 m a pixel; a red vehicle
        # tagged 14 under it, nose up, and a road around it
        metres = 29.25 * 2.0 * math.tan(math.radians(6.0)) / 512
        picture = np.full((512, 512, 4), 128, np.uint8)
        tags = np.full((512, 512), 1, np.uint8)
        rows = slice(int(256 - 2.4 / metres) + 4, int(256 + 2.4 / metres) - 4)
        cols = slice(int(256 - 1.0 / metres) + 4, int(256 + 1.0 / metres) - 4)
        picture[rows, cols] = (0, 0, 255, 255)
        tags[rows, cols] = 14
        sprite = viewer.sprite_from(picture, tags, [14], 12.0, 29.25, vehicle)
        self.assertAlmostEqual(sprite.shape[0], 4.8 / metres, delta=2)
        self.assertAlmostEqual(sprite.shape[1], 2.0 / metres, delta=2)
        self.assertEqual(sprite[0, 0, 3], 0)  # the road is not part of it
        self.assertEqual(tuple(sprite[sprite.shape[0] // 2, sprite.shape[1] // 2]),
                         (0, 0, 255, 255))

        view = viewer.Viewer({"front": FRONT, "rear": REAR,
                              "bev": BevRig.from_config(CONFIG["bev"])},
                             lambda ns: None, vehicle=vehicle)
        view.set_sprite(sprite)
        panel = view.with_vehicle("bev", np.zeros((400, 400, 3), np.uint8))
        self.assertEqual(tuple(panel[200, 200]), (0, 0, 255))
        self.assertEqual(tuple(panel[200, 150]), (0, 0, 0))  # beside it, nothing drawn
        # In the map, an occupied stall holds the same car, grey
        stall = Stall(self.pose.x - 5.0, self.pose.y, 0.0, 5.0, 2.8)
        view.status({"park": {"phase": "search", "target": None,
                              "pose": {"x": self.pose.x, "y": self.pose.y, "z": 0.0,
                                       "roll": 0.0, "pitch": 0.0, "yaw": self.pose.yaw},
                              "stalls": [dict(stall.as_dict(), id=1, free=False, corners=[
                                  [float(x), float(y)] for x, y, _ in stall.corners()])]}})
        drawn = view.map()
        grey = (drawn[:, :, 0] == drawn[:, :, 2]) & (drawn[:, :, 0] > 80) \
            & (drawn[:, :, 0] < 200)
        self.assertGreater(int(grey.sum()), 3000)
        self.assertEqual(tuple(drawn[300, 300]), (0, 0, 255))

    def test_ground_no_camera_sees_now_is_the_ground_seen_before(self):
        from video_analytics_simulation.bev import BevRig
        rig = BevRig.from_config(CONFIG["bev"])
        poses = {1: Pose(0.0, 0.0, 0.0), 2: Pose(1.0, 0.0, 0.0)}  # 1 m forward
        view = viewer.Viewer({"front": FRONT, "bev": rig}, poses.get)
        np.testing.assert_allclose(viewer.carried(rig, poses[1], poses[1]),
                                   [[1, 0, 0], [0, 1, 0]], atol=1e-6)
        first = np.zeros((800, 800, 3), np.uint8)
        first[300:310, 395:405] = 255  # a patch 2.5 m ahead of the vehicle
        seen_all = np.zeros((800, 800), bool)
        view.fill(first, seen_all, 1)
        filled = view.fill(np.zeros_like(first), ~seen_all, 2)
        # 1 m on, the patch is 1.5 m ahead: 40 pixels further down
        self.assertTrue((filled[341:349, 396:404] == 255).all())
        self.assertEqual(int(filled[300:338].sum()), 0)
        # What a camera sees now is kept as it is
        now = np.full_like(first, 7)
        self.assertTrue((view.fill(now, seen_all, 2) == 7).all())


if __name__ == "__main__":
    unittest.main()
