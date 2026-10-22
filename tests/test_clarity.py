"""Unit tests for the display clarity calibration seam (alpr/clarity.py).

Presentation scale only: raw OCR conf (mean per-character softmax, ADR-0006) stays
the Events log's ground truth; the overlay and dashboard render this mapping.
"""

import unittest

from alpr.clarity import CLARITY_ANCHORS, clarity


class ClarityTests(unittest.TestCase):
    def test_anchors_map_exactly(self):
        for conf, expected in CLARITY_ANCHORS:
            self.assertAlmostEqual(clarity(conf), expected, places=9)

    def test_is_monotonic(self):
        samples = [0.0, 0.05, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.30, 1.0]
        values = [clarity(c) for c in samples]
        self.assertEqual(values, sorted(values))

    def test_clear_plate_reads_near_100(self):
        # The user-visible contract: a strong (>= 0.22) reading shows as ~98%+.
        self.assertGreaterEqual(clarity(0.22), 98.0)
        self.assertLessEqual(clarity(0.22), 99.0)

    def test_clamped_outside_anchor_range(self):
        self.assertAlmostEqual(clarity(-1.0), CLARITY_ANCHORS[0][1])
        self.assertAlmostEqual(clarity(7.0), CLARITY_ANCHORS[-1][1])

    def test_typical_reading_is_high_but_not_top(self):
        # CCT family's typical event conf ~0.15 lands comfortably high, not maxed.
        typical = clarity(0.15)
        self.assertGreater(typical, 88.0)
        self.assertLess(typical, 94.0)


if __name__ == "__main__":
    unittest.main()
