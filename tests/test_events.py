"""Unit tests for the CSV events log (run: python -m unittest discover -s tests).

Recognition-event behavior now lives at the tracker seam (tests/test_tracker.py, ADR-0006).
"""

import tempfile
import unittest
from pathlib import Path

from alpr.events import EventLog, HEADER, class_for, status_for, STATUS_CAR, STATUS_LEGACY, STATUS_UNVALIDATED


class StatusTests(unittest.TestCase):
    def test_status_for_maps_the_domain_statuses(self):
        self.assertEqual(status_for("CAR"), STATUS_CAR)
        self.assertEqual(status_for("LEGACY"), STATUS_LEGACY)
        self.assertEqual(status_for(""), STATUS_UNVALIDATED)

    def test_no_motorcycle_status(self):
        # ADR-0007: motorcycles never enter the domain; the status set is three-valued.
        self.assertNotIn("VALIDATED_MOTORCYCLE", (STATUS_CAR, STATUS_LEGACY, STATUS_UNVALIDATED))

    def test_class_column_only_proven_by_car(self):
        # ADR-0008: the legacy match names a format, not a class.
        self.assertEqual(class_for("CAR"), "CAR")
        self.assertEqual(class_for("LEGACY"), "")
        self.assertEqual(class_for(""), "")


class EventLogTests(unittest.TestCase):
    def test_header_written_once_and_rows_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            log.append("T1", "AB123CD", "CAR", "VALIDATED_CAR", 0.9123)
            log.append("T2", "AB!23CD", "", "UNVALIDATED", 0.40)
            lines = log.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], ",".join(HEADER))
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[2], "T2,AB!23CD,,UNVALIDATED,0.4000")
            # Reopening (a restart) does not rewrite the header.
            again = EventLog(log.path)
            self.assertEqual(len(again.path.read_text(encoding="utf-8").splitlines()), 3)

    def test_nested_path_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "data" / "events.csv")
            self.assertTrue(log.path.exists())


if __name__ == "__main__":
    unittest.main()
