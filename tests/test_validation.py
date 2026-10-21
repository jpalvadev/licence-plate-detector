"""Unit tests for raw-reading validation (run: python -m unittest discover -s tests)."""

import unittest

from alpr.validation import normalize, classify


class NormalizeTests(unittest.TestCase):
    def test_separators_stripped(self):
        self.assertEqual(normalize("ab 123-cd"), "AB123CD")
        self.assertEqual(normalize("a_123 bcd"), "A123BCD")
        self.assertEqual(normalize("AB 123 CD"), "AB123CD")

    def test_untouched_other_chars(self):
        self.assertEqual(normalize("ab!123cd"), "AB!123CD")


class ClassifyTests(unittest.TestCase):
    def test_car(self):
        self.assertEqual(classify("AB123CD"), "CAR")

    def test_motorcycle(self):
        self.assertEqual(classify("A123BCD"), "MOTORCYCLE")

    def test_car_before_motorcycle(self):
        # The two regexes are mutually exclusive (position 2 is a letter for cars),
        # but the order documents the car-first rule from the design.
        self.assertEqual(classify("AB123CD"), "CAR")

    def test_rejects_old_and_bad_formats(self):
        self.assertIsNone(classify("ABC123"))       # old 1995-2016 plate
        self.assertIsNone(classify("AB1234CD"))
        self.assertIsNone(classify("AB12CD3"))
        self.assertIsNone(classify(""))


if __name__ == "__main__":
    unittest.main()
