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
        self.assertEqual(classify("AB123CD", include_legacy=True), "CAR")

    def test_rejects_motorcycle_format(self):
        # ADR-0007: motorcycles are out of the domain in either era.
        self.assertIsNone(classify("A123BCD"))
        self.assertIsNone(classify("A123BCD", include_legacy=True))
        self.assertIsNone(classify("123ABC"))       # legacy motorcycle era

    def test_legacy_only_when_format_set_includes_it(self):
        # ADR-0008: MERCOSUR_ONLY (default) does not validate the legacy shape...
        self.assertIsNone(classify("ABC123"))
        # ...MERCOSUR_PLUS_LEGACY does, and proves no vehicle class (car/cargo shared it).
        self.assertEqual(classify("ABC123", include_legacy=True), "LEGACY")

    def test_rejects_bad_formats(self):
        self.assertIsNone(classify("AB1234CD"))
        self.assertIsNone(classify("AB12CD3"))
        self.assertIsNone(classify("ABCD123"))  # separator re-issue: no lenient regex
        self.assertIsNone(classify(""))

    def test_rejects_bad_formats_with_legacy_included(self):
        self.assertIsNone(classify("AB1234CD", include_legacy=True))
        self.assertIsNone(classify("AB12CD3", include_legacy=True))
        self.assertIsNone(classify("ABCD123", include_legacy=True))
        self.assertIsNone(classify("", include_legacy=True))


if __name__ == "__main__":
    unittest.main()
