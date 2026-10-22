"""Unit tests for the CSV events log (run: python -m unittest discover -s tests).

Recognition-event behavior now lives at the tracker seam (tests/test_tracker.py, ADR-0006).
ADR-0009: only validated plates reach the log, and the class column is gone.
"""

import tempfile
import unittest
from pathlib import Path

from alpr.events import EventLog, HEADER, status_for, STATUS_CAR, STATUS_LEGACY


class StatusTests(unittest.TestCase):
    def test_status_for_maps_the_domain_statuses(self):
        self.assertEqual(status_for("CAR"), STATUS_CAR)
        self.assertEqual(status_for("LEGACY"), STATUS_LEGACY)

    def test_status_for_rejects_non_format_match(self):
        # ADR-0009: there is no VALIDAR status anymore; the tracker filters before
        # calling, so a non format match is a programming error (fail-fast convention).
        with self.assertRaises(ValueError):
            status_for("")

    def test_status_set_is_two_valued(self):
        # ADR-0007: two-wheelers never enter the domain; the status set is two-valued.
        self.assertEqual({STATUS_CAR, STATUS_LEGACY}, {"PATENTE_MERCOSUR", "PATENTE_ALFANUM"})

    def test_no_class_column(self):
        # ADR-0009: with two-wheelers out of the domain the class column is a tautology.
        self.assertNotIn("vehicle_class", HEADER)
        self.assertEqual(HEADER, ("timestamp", "patente", "status", "confidence"))


class EventLogTests(unittest.TestCase):
    def test_header_written_once_and_rows_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            log.append("T1", "AB123CD", STATUS_CAR, 0.9123)
            log.append("T2", "ABC123", STATUS_LEGACY, 0.40)
            lines = log.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], ",".join(HEADER))
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[2], "T2,ABC123,PATENTE_ALFANUM,0.4000")
            # Reopening (a restart) does not rewrite the header.
            again = EventLog(log.path)
            self.assertEqual(len(again.path.read_text(encoding="utf-8").splitlines()), 3)

    def test_nested_path_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "data" / "events.csv")
            self.assertTrue(log.path.exists())


if __name__ == "__main__":
    unittest.main()
