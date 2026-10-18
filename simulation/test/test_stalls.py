import math
import os
import sys
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import fixtures  # noqa: E402
from video_analytics_simulation.calibration import Pose, Rig  # noqa: E402
from video_analytics_simulation.parking import Aisle, Stall, reverse_steer  # noqa: E402
from video_analytics_simulation.stalls import DEFAULTS, StallTracker  # noqa: E402

with open(os.path.join(os.path.dirname(__file__), "..", "config", "carla.yaml")) as f:
    CONFIG = yaml.safe_load(f)
REAR = Rig.from_config(CONFIG["rear_calibration"])
VEHICLE = {"wheelbase_m": 3.0, "max_steer_deg": 70.0, "width_m": 2.16}
TICK = 33333333
# A row of six stalls along y, opening towards +x; the aisle is at x = 7.2
ROW = [Stall(0.0, 2.8 * i, 0.0, 5.0, 2.8) for i in range(6)]


def see(tracker, stalls, pose, tick, occupied=(), noise_px=0.0, rng=None, model=True):
    """One frame of the rear camera at `pose`: what the stall model finds."""
    return tracker.update(
        fixtures.stall_detections(stalls, pose, REAR, occupied, noise_px, rng) if model else [],
        fixtures.stall_points(stalls, pose, REAR, noise_px, rng), pose, tick * TICK)


def passing(tracker, stalls, x=7.2, y_from=-6.0, y_to=30.0, step=0.07, occupied=(),
            noise_px=0.0, seed=0):
    """The vehicle drives along +y past the row; the rear camera looks back."""
    rng = np.random.default_rng(seed)
    tick = 0
    for y in np.arange(y_from, y_to, step):
        see(tracker, stalls, Pose(x, float(y), 0.0, 0.0, 0.0, 90.0), tick, occupied,
            noise_px, rng)
        tick += 1
    return tick


def by_index(tracker):
    """Tracks by the number of the stall of ROW they are on."""
    return {round(t.stall().centre[1] / 2.8): t for t in tracker.tracks}


class Geometry(unittest.TestCase):
    def test_corners_go_around_the_stall_mouth_first(self):
        stall = Stall(1.0, 2.0, 0.0, 5.0, 2.8)
        np.testing.assert_allclose(stall.corners()[:, :2], [
            [3.5, 3.4], [3.5, 0.6], [-1.5, 0.6], [-1.5, 3.4]])
        turned = Stall(0.0, 0.0, 90.0, 5.0, 2.8)  # opens towards +y
        np.testing.assert_allclose(turned.corners()[0, :2], [-1.4, 2.5], atol=1e-9)

    def test_the_ends_of_a_row_are_each_seen_once(self):
        pose = Pose(7.2, 14.0, 0.0, 0.0, 0.0, 90.0)
        ends = fixtures.stall_points(ROW[:2], pose, REAR)
        # Two stalls share a divider: three ends at the mouth, three at the back
        self.assertEqual(sorted(e["class"] for e in ends), ["back"] * 3 + ["mouth"] * 3)


class Tracking(unittest.TestCase):
    def test_a_yaw_bias_since_the_last_sighting_is_all_the_carry_adds(self):
        tracker = StallTracker(REAR)
        passing(tracker, ROW)
        track = by_index(tracker)[0]
        pose = Pose(7.2, 20.0, 0.0, 0.0, 0.0, 90.0)
        biased = Pose(7.2, 6.0, 0.0, 0.0, 0.0, 94.0)
        honest = track.seen_from(pose).heading_error(0.0)
        shifted = track.seen_from(biased).heading_error(0.0)
        self.assertAlmostEqual(shifted - honest, math.radians(4.0), delta=0.03)
        see(tracker, ROW[:1], pose, 5000)
        again = by_index(tracker)[0].seen_from(pose).heading_error(0.0)
        self.assertAlmostEqual(again, honest, delta=0.03)

    def test_a_row_passed_once_is_found_where_it_is(self):
        tracker = StallTracker(REAR)
        passing(tracker, ROW)
        found = by_index(tracker)
        self.assertEqual(sorted(found), [0, 1, 2, 3, 4, 5])
        for index, track in found.items():
            stall = track.stall()
            np.testing.assert_allclose(stall.centre, ROW[index].centre, atol=0.02)
            self.assertAlmostEqual(stall.width, 2.8, delta=0.02)
            self.assertAlmostEqual(stall.length, 5.0, delta=0.02)
            self.assertAlmostEqual(math.degrees(stall.heading_error(0.0)), 0.0, delta=0.3)
            self.assertTrue(track.free(tracker.config))
            # It opens towards the aisle the camera was on, and +v is +v
            self.assertGreater(track.plus.xy[1], track.minus.xy[1])

    def test_passed_the_other_way_it_opens_the_same_way(self):
        tracker = StallTracker(REAR)
        tick = 0
        for y in np.arange(22.0, -14.0, -0.07):
            see(tracker, ROW, Pose(7.2, float(y), 0.0, 0.0, 0.0, -90.0), tick)
            tick += 1
        for index, track in by_index(tracker).items():
            self.assertAlmostEqual(math.degrees(track.stall().heading_error(0.0)), 0.0,
                                   delta=0.3)
            np.testing.assert_allclose(track.stall().centre, ROW[index].centre, atol=0.02)

    def test_two_pixels_of_noise_are_outweighed_by_what_was_seen_near(self):
        tracker = StallTracker(REAR)
        passing(tracker, ROW, noise_px=2.0)
        found = by_index(tracker)
        self.assertEqual(sorted(found), [0, 1, 2, 3, 4, 5])
        for index, track in found.items():
            stall = track.stall()
            # The mouth passes 5 m from the camera; the back is never under 10 m
            self.assertAlmostEqual(stall.centre[1], 2.8 * index, delta=0.15)
            self.assertAlmostEqual(stall.width, 2.8, delta=0.2)
            self.assertAlmostEqual(stall.frame(ROW[index].world(2.5, 0.0))[0],
                                   stall.length / 2.0, delta=0.15)
            # The direction comes from the whole row's mouth, not from 2.8 m of it
            self.assertAlmostEqual(math.degrees(stall.heading_error(0.0)), 0.0, delta=1.5)

    def test_the_length_is_five_metres_until_the_back_ends_are_seen(self):
        long = [Stall(0.3, 2.8 * i, 0.0, 5.6, 2.8) for i in range(3)]  # mouth at 3.1
        tracker = StallTracker(REAR)
        ticks = passing(tracker, long, y_to=16.0)
        track = by_index(tracker)[1]
        # From the aisle the back of the row is beyond the range ends are used at
        self.assertEqual(track.backs, ())
        stall = track.stall()
        self.assertEqual(stall.length, DEFAULTS["length_default_m"])
        # Its mouth is where it is all the same: only its back is not known
        self.assertAlmostEqual(stall.frame([3.1, 2.8])[0], stall.length / 2.0, delta=0.02)
        # Backing into it: the back ends of its dividers come near
        rng = np.random.default_rng(1)
        for i, x in enumerate(np.arange(8.6, 3.2, -0.05)):
            see(tracker, long, Pose(float(x), 2.8, 0.0, 0.0, 0.0, 0.0), ticks + i,
                noise_px=1.0, rng=rng)
        self.assertEqual(len(track.backs), 2)
        stall = track.stall()
        np.testing.assert_allclose(stall.centre, long[1].centre, atol=0.05)
        self.assertAlmostEqual(stall.length, 5.6, delta=0.05)
        self.assertAlmostEqual(math.degrees(stall.heading_error(0.0)), 0.0, delta=1.0)
        self.assertLess(track.plus.nearest_m, 2.0)

    def test_a_row_seen_from_behind_opens_away_from_the_camera(self):
        # The camera passes behind this row: its back line is at x = 2.5,
        # 1.5 m from the camera's path, its mouths beyond, at x = -2.5
        away = [Stall(0.0, 2.8 * i, 180.0, 5.0, 2.8) for i in range(4)]
        tracker = StallTracker(REAR)
        passing(tracker, away, x=4.0, y_to=22.0)
        self.assertGreaterEqual(len(tracker.tracks), 2)
        for track in tracker.tracks:
            stall = track.stall()
            self.assertAlmostEqual(abs(math.degrees(stall.heading_error(180.0))), 0.0, delta=1.0)
            self.assertAlmostEqual(stall.centre[0], 0.0, delta=0.1)
            self.assertGreater(len(track.backs), 0)
            # +v is to the right of the way it opens, here towards -y
            self.assertLess(track.plus.xy[1], track.minus.xy[1])

    def test_without_stall_detections_there_are_stalls_but_none_is_free(self):
        tracker = StallTracker(REAR)
        tick = 0
        for y in np.arange(-6.0, 20.0, 0.07):
            see(tracker, ROW[:3], Pose(7.2, float(y), 0.0, 0.0, 0.0, 90.0), tick, model=False)
            tick += 1
        found = by_index(tracker)
        self.assertEqual(sorted(found), [0, 1, 2])
        self.assertFalse(any(t.free(tracker.config) for t in found.values()))
        # No detection to say how long they are either
        self.assertEqual(found[1].stall().length, DEFAULTS["length_default_m"])

    def test_a_car_in_a_stall(self):
        tracker = StallTracker(REAR)
        passing(tracker, ROW[:3], occupied=(1,))
        found = by_index(tracker)
        self.assertEqual([found[i].free(tracker.config) for i in range(3)],
                         [True, False, True])
        self.assertEqual(found[1].occupied_share(), 1.0)
        # The neighbours of the middle one share its ends; none beyond the row's
        self.assertIs(tracker.beside(found[1], 1), found[2])
        self.assertIs(tracker.beside(found[1], -1), found[0])
        self.assertIsNone(tracker.beside(found[0], -1))
        self.assertIsNone(tracker.beside(found[2], 1))

    def test_a_detection_votes_for_the_stall_whose_box_it_has(self):
        tracker = StallTracker(REAR)
        tick = 0
        for y in np.arange(-6.0, 20.0, 0.07):
            pose = Pose(7.2, float(y), 0.0, 0.0, 0.0, 90.0)
            stalls = fixtures.stall_detections(ROW[:3], pose, REAR, (1,))
            for detection in stalls:
                # Its corners are rough, and here in another order: not used
                detection["corners"] = detection["corners"][2:] + detection["corners"][:2]
                # Its box a few pixels off is still its box
                detection["bbox"]["x"] += 3
                detection["bbox"]["h"] += 2
            tracker.update(stalls, fixtures.stall_points(ROW[:3], pose, REAR), pose, tick * TICK)
            tick += 1
        found = by_index(tracker)
        self.assertEqual([found[i].free(tracker.config) for i in range(3)],
                         [True, False, True])
        self.assertGreater(min(t.hits for t in found.values()), 30)
        # A box that is no stall's votes for none
        votes = tracker.update(
            [{"class": "occupied", "confidence": 0.9, "corners": [[0, 0, 0.9]] * 4,
              "bbox": {"x": 100, "y": 600, "w": 300, "h": 100}}], [], pose, tick * TICK)
        self.assertEqual(votes["votes"], [])

    def test_an_end_seen_a_few_times_marks_nothing_and_is_forgotten(self):
        tracker = StallTracker(REAR)
        pose = Pose(7.2, 9.0, 0.0, 0.0, 0.0, 90.0)
        for tick in range(DEFAULTS["end_hits"] - 1):
            see(tracker, ROW[:1], pose, tick)
        self.assertEqual(len(tracker.ends), 2)   # its two mouth ends; the back is too far
        self.assertEqual(tracker.tracks, [])
        tracker.update([], [], pose, int(3e9))
        self.assertEqual(tracker.ends, [])
        # Seen often enough, they stay when they are no longer seen
        for tick in range(DEFAULTS["end_hits"]):
            see(tracker, ROW[:1], pose, 100 + tick)
        tracker.update([], [], pose, int(60e9))
        self.assertEqual(len(tracker.tracks), 1)

    def test_what_is_not_a_stall(self):
        pose = Pose(7.2, 12.0, 0.0, 0.0, 0.0, 90.0)
        # Two mouth ends 5.6 m apart
        tracker = StallTracker(REAR)
        for tick in range(10):
            see(tracker, [Stall(0.0, 0.0, 0.0, 5.0, 5.6)], pose, tick)
        self.assertEqual(tracker.tracks, [])
        # Ends the model is not sure of, ends that are far, and the sky
        tracker = StallTracker(REAR)
        unsure = [dict(p, confidence=0.3) for p in fixtures.stall_points(ROW[:1], pose, REAR)]
        far = fixtures.stall_points(ROW[:1], Pose(7.2, 30.0, 0, 0, 0, 90.0), REAR)
        self.assertTrue(far)
        for tick in range(10):
            tracker.update([], unsure, pose, tick * TICK)
            tracker.update([], far, Pose(7.2, 30.0, 0, 0, 0, 90.0), tick * TICK)
            tracker.update([], [{"class": "mouth", "confidence": 0.9, "x": 640.0, "y": 5.0}],
                           pose, tick * TICK)
        self.assertEqual(tracker.ends, [])


class TheAisle(unittest.TestCase):
    def setUp(self):
        self.aisle = Aisle(Pose(7.2, -6.0, 0.0, 0.0, 0.0, 90.0), VEHICLE,
                           {"turn_gain": 1.0})

    def test_past_a_stall_and_the_circle_into_it(self):
        aisle, stall = self.aisle, ROW[1]
        self.assertEqual(aisle.direction(stall), 1)
        # Rear axle 1.5 m behind the origin: 4 m past the axis at y = 8.3
        pose = Pose(7.2, 8.3, 0.0, 0.0, 0.0, 90.0)
        self.assertAlmostEqual(aisle.past(stall, pose), 4.0)
        self.assertAlmostEqual(aisle.travelled(pose), 14.3)
        # From 7.2 m out to 1 m inside the mouth, which is 2.5 m out
        self.assertAlmostEqual(aisle.radius(stall, pose), 7.2 - 1.5)
        self.assertAlmostEqual(aisle.steer_share(3.0 / math.tan(math.radians(31.5))), 0.45)
        self.assertEqual(aisle.steer_share(0.0), math.inf)
        # The other way along the same aisle
        back = Aisle(Pose(7.2, 30.0, 0.0, 0.0, 0.0, -90.0), VEHICLE, {})
        self.assertEqual(back.direction(stall), -1)
        self.assertAlmostEqual(back.past(stall, Pose(7.2, -2.7, 0, 0, 0, -90.0)), 4.0)

    def test_stalls_that_are_not_on_this_aisle(self):
        aisle = self.aisle
        self.assertEqual(aisle.direction(Stall(0.0, 0.0, 180.0, 5.0, 2.8)), 0)   # opens away
        self.assertEqual(aisle.direction(Stall(0.0, 0.0, 45.0, 5.0, 2.8)), 0)    # askew
        self.assertEqual(aisle.direction(Stall(-9.0, 0.0, 0.0, 5.0, 2.8)), 0)    # a row beyond
        self.assertEqual(aisle.direction(Stall(4.0, 0.0, 0.0, 5.0, 2.8)), 0)     # under the wheels

    def test_the_line_in_the_ego_frame(self):
        track = self.aisle.track(Pose(7.7, 3.0, 0.0, 0.0, 0.0, 90.0))
        # Heading +y, its right is -x: at x = 7.7 it is 0.5 m to the left of
        # the line, and the line is to its right
        np.testing.assert_allclose(track["coefficients"], [0.0, 0.0, 0.5], atol=1e-9)
        self.assertAlmostEqual(self.aisle.offset([7.7, 3.0]), -0.5)
        self.assertAlmostEqual(self.aisle.heading_error(100.0), math.radians(10.0))

    def test_reverse_steer_points_the_vehicle_at_the_side_it_is_on(self):
        self.assertEqual(reverse_steer(0.0, 0.0, {}), 0.0)
        self.assertGreater(reverse_steer(0.0, -0.2, {}), 0.0)   # left of the line
        self.assertLess(reverse_steer(0.0, 0.2, {}), 0.0)
        self.assertGreater(reverse_steer(0.05, 0.0, {}), 0.0)   # turned right
        self.assertEqual(reverse_steer(1.0, 0.0, {}), 0.3)      # never more than this


if __name__ == "__main__":
    unittest.main()
