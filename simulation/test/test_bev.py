import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from video_analytics_simulation.bev import BevRig, Joiner, Stitcher  # noqa: E402
from video_analytics_simulation.calibration import Pose, Rig  # noqa: E402
from video_analytics_simulation.parking import Stall  # noqa: E402
from video_analytics_simulation.stalls import StallTracker  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures", "bev.json")
VIEW = {"width": 800, "height": 800, "metres_per_pixel": 0.025}
CAMERAS = [  # rear, left, right: those of simulation/config/carla.yaml
    {"width": 1280, "height": 720, "fov_deg": 120.0,
     "camera": {"x": -2.45, "y": 0.0, "z": 1.0, "pitch_deg": -30.0, "yaw_deg": 180.0}},
    {"width": 1280, "height": 720, "fov_deg": 120.0,
     "camera": {"x": 0.55, "y": -1.05, "z": 1.0, "pitch_deg": -45.0, "yaw_deg": -105.0}},
    {"width": 1280, "height": 720, "fov_deg": 120.0,
     "camera": {"x": 0.55, "y": 1.05, "z": 1.0, "pitch_deg": -45.0, "yaw_deg": 105.0}},
]
# Pixels of the view (column, row): each camera's part, the seams left out,
# the ground under the vehicle and a corner nobody sees
PIXELS = [[400, 700], [400, 520], [150, 600], [650, 610], [100, 400], [230, 380],
          [340, 330], [700, 400], [560, 390], [455, 345], [400, 100], [400, 400],
          [20, 780], [780, 20], [300, 250], [520, 470]]


def stitcher():
    return Stitcher(BevRig.from_config(VIEW), [Rig.from_config(c) for c in CAMERAS])


def fixture():
    """What tests/fixtures/bev.json holds: for each pixel the camera it is
    taken from and where in that camera's image, as remap is asked."""
    made = stitcher()
    pixels = []
    for column, row in PIXELS:
        source = int(made.source[row, column])
        place = [None, None] if source < 0 else [
            round(float(made.maps[source][0][row, column]), 3),
            round(float(made.maps[source][1][row, column]), 3)]
        pixels.append([column, row, source] + place)
    return {"view": VIEW, "cameras": CAMERAS, "pixels": pixels}


class View(unittest.TestCase):
    def test_a_pixel_is_the_ground_it_shows(self):
        view = BevRig.from_config(VIEW)
        xy, valid = view.uv_to_ground(np.array([400.0, 500.0, 400.0, 800.0]),
                                      np.array([400.0, 400.0, 200.0, 10.0]))
        np.testing.assert_allclose(xy[:3], [[0.0, 0.0], [0.0, 2.5], [5.0, 0.0]], atol=1e-9)
        self.assertEqual(valid.tolist(), [True, True, True, False])
        uv, depth = view.project(np.array([[5.0, 0.0, 0.0], [-2.0, -3.0, 0.0]]))
        np.testing.assert_allclose(uv, [[400.0, 200.0], [280.0, 480.0]])
        self.assertTrue((depth > 0.3).all())
        self.assertFalse(view.uv_to_ground(400.0, 100.0, max_range_m=5.0)[1])

    def test_the_box_of_a_stall_is_of_the_part_inside_the_image(self):
        view = BevRig.from_config(VIEW)
        stall = Stall(-9.0, 0.0, 0.0, 5.0, 2.6)  # its back is 1.5 m past the lower edge
        x0, y0, x1, y1 = view.visible_box(stall.corners(0.0))
        np.testing.assert_allclose([x0, x1], [400 - 52, 400 + 52])
        np.testing.assert_allclose([y0, y1], [400 + 6.5 / 0.025, 800.0])
        self.assertIsNone(view.visible_box(Stall(-20.0, 0.0, 0.0, 5.0, 2.6).corners(0.0)))

    def test_the_stall_tracker_takes_it_for_a_camera(self):
        view = BevRig.from_config(VIEW)
        tracker = StallTracker(view)
        pose = Pose(10.0, 20.0, 0.0, 0.0, 0.0, 30.0)
        # Three mouth ends 2.7 m apart, 3 m to the left of the vehicle
        ends = [(x, -3.0) for x in (-2.7, 0.0, 2.7)]
        points = [{"class": "mouth", "confidence": 0.9,
                   "x": float(view.project(np.array([x, y, 0.0]))[0][0]),
                   "y": float(view.project(np.array([x, y, 0.0]))[0][1])} for x, y in ends]
        for i in range(6):
            tracker.update([], points, pose, 1000 * (i + 1))
        self.assertEqual(len(tracker.tracks), 2)
        centre = pose.from_parent(np.array(list(tracker.tracks[0].fitted.centre) + [0.0]))
        # 2.5 m further from the vehicle than the mouth: half the default length
        self.assertAlmostEqual(float(centre[1]), -5.5, places=2)
        self.assertAlmostEqual(tracker.tracks[0].fitted.width, 2.7, places=2)


class Stitching(unittest.TestCase):
    def test_the_fixture_is_what_this_code_gives(self):
        with open(FIXTURE) as f:
            self.assertEqual(json.load(f), fixture())

    def test_every_camera_has_a_part_and_the_vehicle_hides_the_ground_under_it(self):
        made = stitcher()
        share = [float((made.source == i).mean()) for i in range(3)]
        for part in share:
            self.assertGreater(part, 0.15)
        # Behind the rear camera it is the rear camera's; abeam, a mirror's
        self.assertEqual(made.source[700, 400], 0)
        self.assertEqual(made.source[420, 150], 1)
        self.assertEqual(made.source[420, 650], 2)
        # Ahead of the mirrors nobody looks
        self.assertEqual(made.source[100, 400], -1)

    def test_paint_on_the_ground_comes_out_where_it_is(self):
        import cv2
        made = stitcher()
        images = []
        for rig in made.rigs:
            # A white square on the ground, 1 m wide, 4 m to the left and 2 m
            # behind the origin, drawn into each camera's image
            corners = np.array([[-2.5, -4.5, 0.0], [-1.5, -4.5, 0.0],
                                [-1.5, -3.5, 0.0], [-2.5, -3.5, 0.0]])
            uv, depth = rig.project(corners)
            image = np.zeros((rig.height, rig.width, 3), np.uint8)
            if (depth > 0.3).all():
                cv2.fillPoly(image, [np.round(uv - 0.5).astype(np.int32)], (255, 255, 255))
            images.append(image)
        view = made.compose(images)
        white = np.argwhere(view[:, :, 0] > 127)
        # Rows 460 to 500 (x from -1.5 to -2.5), columns 220 to 260
        np.testing.assert_allclose(white.min(axis=0), [460, 220], atol=2)
        np.testing.assert_allclose(white.max(axis=0), [499, 259], atol=2)

    def test_an_image_of_another_size_is_refused(self):
        made = stitcher()
        with self.assertRaises(ValueError):
            made.paste(made.blank(), 0, np.zeros((360, 640, 3), np.uint8))


class Control(unittest.TestCase):
    def test_the_controller_follows_the_stalls_of_the_view_it_is_given(self):
        import yaml
        from video_analytics_simulation.controller import ControlCore
        with open(os.path.join(os.path.dirname(__file__), "..", "config", "carla.yaml")) as f:
            config = yaml.safe_load(f)
        vehicle = {"wheelbase_m": 3.0, "max_steer_deg": 70.0, "width_m": 2.16,
                   "front_bumper_m": 2.43}
        view = BevRig.from_config(config["bev"])
        core = ControlCore(config, vehicle, "r_test", scenario={"park": True}, stall_rig=view)
        self.assertIs(core.stalls.rig, view)
        # Not given one, it looks for them in the rear camera's image
        core = ControlCore(config, vehicle, "r_test", scenario={"park": True})
        self.assertIs(core.stalls.rig, core.rear_rig)
        self.assertEqual(core.rear_rig.width, 1280)


class Joining(unittest.TestCase):
    def test_the_rear_result_gets_the_stalls_of_the_view_of_its_frame(self):
        def result(stream, ns, **more):
            return dict({"stream": stream, "simulation": {"run_id": "r", "sim_ns": ns}}, **more)
        joiner = Joiner(keep=2)
        self.assertIsNone(joiner.offer("rear", result("carla-rear", 1, detections=["car"])))
        self.assertIsNone(joiner.offer("bev", result("carla-bev", 2, stalls=["b"],
                                                     stall_points=["q"])))
        joined = joiner.offer("bev", result("carla-bev", 1, stalls=["a"], stall_points=["p"]))
        self.assertEqual(joined["stream"], "carla-rear")
        self.assertEqual((joined["detections"], joined["stalls"], joined["stall_points"]),
                         (["car"], ["a"], ["p"]))
        self.assertEqual(joiner.offer("rear", result("carla-rear", 2))["stalls"], ["b"])
        # Nothing joins twice, and a half that waited too long is gone
        self.assertIsNone(joiner.offer("bev", result("carla-bev", 1, stalls=[])))
        for ns in (3, 4, 5):
            joiner.offer("rear", result("carla-rear", ns))
        self.assertIsNone(joiner.offer("bev", result("carla-bev", 3, stalls=[])))
        self.assertIsNotNone(joiner.offer("bev", result("carla-bev", 5, stalls=[])))
        self.assertIsNone(joiner.offer("bev", {"stream": "carla-bev"}))


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        with open(FIXTURE, "w") as f:
            json.dump(fixture(), f, indent=1)
            f.write("\n")
    else:
        unittest.main()
