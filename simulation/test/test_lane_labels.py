import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from video_analytics_simulation import lane  # noqa: E402
from video_analytics_simulation.calibration import Rig  # noqa: E402

RIG = Rig(1280, 720, 52.0, 0.6, 0.0, 1.5, pitch_deg=-10.0)


class LaneLabels(unittest.TestCase):
    def test_a_line_on_the_road_is_labelled_where_it_is(self):
        # 1.75 m right of the vehicle, from 2 to 80 m ahead
        x = np.arange(2.0, 80.0, 1.0)
        ground = np.stack([x, np.full_like(x, 1.75), np.zeros_like(x)], axis=1)
        uv, depth = RIG.project(ground)
        rows = lane.anchor_rows(RIG.height)
        columns = lane.boundary_at_rows(uv[depth > 0.5], rows, RIG.width)
        seen = columns >= 0
        self.assertGreater(seen.sum(), 50)
        xy, valid = RIG.uv_to_ground(columns[seen], np.array(rows)[seen])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(xy[:, 1], 1.75, atol=0.01)

    def test_a_line_that_leaves_the_image_has_no_column_there(self):
        rows = lane.anchor_rows(720)
        # From the lower edge out through the right edge at row 400
        columns = lane.boundary_at_rows([[900, 800], [1400, 300]], rows, 1280)
        for row, column in zip(rows, columns):
            self.assertEqual(column >= 0, row > 420, row)
        self.assertTrue((lane.boundary_at_rows([[5, 5]], rows, 1280) == -1).all())

    def test_the_nearest_crossing_of_a_row_counts(self):
        columns = lane.boundary_at_rows([[100, 700], [200, 300], [900, 500]], [400], 1280)
        self.assertAlmostEqual(columns[0], 175.0)

    def test_the_decoder_answers_the_column_it_was_given(self):
        rows = lane.anchor_rows(720)
        wanted = np.linspace(40.0, 1240.0, len(rows))
        place = lane.column_of(wanted, 1280)
        logits = np.full((lane.GRID + 1, len(rows), 4), -30.0)
        for j, value in enumerate(place):
            low = int(np.floor(value))  # two bins that average to the place
            logits[low - 1, j, 1] = np.log(low + 1 - value + 1e-12)
            logits[low, j, 1] = np.log(value - low + 1e-12)
        points = lane.decode_ufld(logits, 1280, 720)[1]
        self.assertEqual(len(points), len(rows))
        # The decoder lists the rows from the lowest in the image
        for (x, y), row, column in zip(points, rows[::-1], wanted[::-1]):
            self.assertEqual(y, row)
            self.assertLessEqual(abs(x - column), 1.0)


if __name__ == "__main__":
    unittest.main()
