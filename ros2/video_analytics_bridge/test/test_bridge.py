import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "video_analytics_bridge"))
import envelope  # noqa: E402  (the module alone: it does not import ROS)

FRAME = {
    "frame_id": 42, "stream": "carla-front",
    "simulation": {"run_id": "r_20261011_ab12", "sim_ns": 1533333333},
    "image": {"width": 1280, "height": 720},
    "detections": [{"class": "car", "confidence": 0.9, "track_id": 3,
                    "bbox": {"x": 600, "y": 300, "w": 80, "h": 60}}],
    "lane": {"valid": True, "confidence": 0.9, "reason": "",
             "left": {"coefficients": [0.0, 0.0, -1.75], "x_min_m": 8, "x_max_m": 30},
             "right": {"coefficients": [0.0, 0.0, 1.75], "x_min_m": 8, "x_max_m": 30},
             "centerline": {"coefficients": [0.0, 0.0, 0.0], "x_min_m": 8, "x_max_m": 30},
             "reference_x_m": 8.0, "lateral_error_m": 0.0, "heading_rad": 0.0,
             "curvature_inv_m": 0.0},
    "stalls": [{"class": "empty", "confidence": 0.93,
                "bbox": {"x": 150, "y": 207, "w": 460, "h": 103},
                "corners": [[150.2, 302.1, 0.98], [508.4, 310.0, 0.97],
                            [606.3, 207.2, 0.91], [466.0, 205.4, 0.9]]}],
    "latency_ms": 12.3, "fps": 30.0,
}


def changed(path, value):
    frame = copy.deepcopy(FRAME)
    target = frame
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return frame


class Envelope(unittest.TestCase):
    def test_a_complete_frame_passes(self):
        self.assertEqual(envelope.check(FRAME), "")
        self.assertEqual(envelope.check(changed(["lane"], KeyError)), "")
        self.assertEqual(envelope.check(changed(["detections"], [])), "")
        invalid_lane = {"valid": False, "reason": "missing_boundary",
                        "centerline": None, "lateral_error_m": None}
        self.assertEqual(envelope.check(changed(["lane"], invalid_lane)), "")
        self.assertEqual(envelope.check(changed(["simulation", "sim_ns"], 0)), "")
        # A camera without the stall model, and one that found no stall
        self.assertEqual(envelope.check(changed(["stalls"], KeyError)), "")
        self.assertEqual(envelope.check(changed(["stalls"], [])), "")

    def test_malformed_frames_are_named(self):
        cases = [
            (["simulation"], KeyError, "no simulation block"),
            (["simulation", "run_id"], "bad/id", "bad run_id"),
            (["simulation", "run_id"], "", "bad run_id"),
            (["simulation", "run_id"], 7, "bad run_id"),
            (["simulation", "sim_ns"], 1533333333.0, "bad sim_ns"),
            (["simulation", "sim_ns"], True, "bad sim_ns"),
            (["simulation", "sim_ns"], -1, "bad sim_ns"),
            (["simulation", "sim_ns"], KeyError, "bad sim_ns"),
            (["stream"], KeyError, "no stream"),
            (["image"], KeyError, "bad image size"),
            (["image", "width"], 0, "bad image size"),
            (["image", "height"], 720.0, "bad image size"),
            (["detections"], KeyError, "no detections list"),
            (["detections", 0, "bbox", "w"], float("nan"), "bad bbox"),
            (["detections", 0, "bbox", "x"], "600", "bad bbox"),
            (["detections", 0, "confidence"], float("inf"), "bad confidence"),
            (["detections", 0, "track_id"], 3.5, "bad track_id"),
            (["detections", 0, "class"], KeyError, "bad detection"),
            (["lane", "valid"], 1, "bad lane"),
            (["lane", "centerline"], None, "bad lane curve"),
            (["lane", "left", "coefficients"], [0.0, 0.0], "bad lane curve"),
            (["lane", "right", "coefficients", 1], float("nan"), "bad lane curve"),
            (["lane", "heading_rad"], None, "bad lane numbers"),
            (["stalls"], {}, "bad stalls"),
            (["stalls", 0, "class"], "free", "bad stall"),
            (["stalls", 0, "confidence"], None, "bad stall"),
            (["stalls", 0, "corners"], [[1, 2, 0.9]] * 3, "bad stall corners"),
            (["stalls", 0, "corners", 2], [1.0, 2.0], "bad stall corners"),
            (["stalls", 0, "corners", 1, 0], float("nan"), "bad stall corners"),
        ]
        for path, value, reason in cases:
            self.assertEqual(envelope.check(changed(path, value)), reason, path)
        self.assertEqual(envelope.check([FRAME]), "not an object")

    def test_time_is_split_without_rounding(self):
        self.assertEqual(envelope.stamp(0), (0, 0))
        self.assertEqual(envelope.stamp(1533333333), (1, 533333333))
        self.assertEqual(envelope.stamp(999999999), (0, 999999999))
        # Larger than a double holds exactly
        big = 9007199254740993
        sec, nanosec = envelope.stamp(big)
        self.assertEqual(sec * envelope.NS_PER_S + nanosec, big)

    def test_topics_keep_runs_and_sources_apart(self):
        self.assertEqual(envelope.topics(FRAME),
                         ("simulation/r_20261011_ab12/carla_front/perception",
                          "simulation/r_20261011_ab12/carla_front/lane"))
        other_run = changed(["simulation", "run_id"], "r_2")
        other_source = changed(["stream"], "carla-rear")
        names = {envelope.topics(f)[0] for f in (FRAME, other_run, other_source)}
        self.assertEqual(len(names), 3)
        self.assertEqual(envelope.optical_frame(FRAME),
                         "r_20261011_ab12/carla_front_optical")


if __name__ == "__main__":
    unittest.main()
