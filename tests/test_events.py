"""Unit tests for the dedup tracker + CSV events log (run: python -m unittest discover -s tests)."""

import tempfile
import unittest
from pathlib import Path

from alpr.events import EventLog, EventTracker, HEADER


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


class TrackerTests(unittest.TestCase):
    def test_dedup_window_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            tracker = EventTracker(EventLog(path), dedup_window_s=10.0)
            t0 = 100.0
            self.assertTrue(tracker.record("V:AB123CD", t0, "T0", "AB123CD", 0.80))
            self.assertFalse(tracker.record("V:AB123CD", t0 + 2, "T1", "AB123CD", 0.95))
            self.assertTrue(tracker.record("V:AB123CD", t0 + 20, "T2", "AB123CD", 0.80))
            self.assertFalse(tracker.record("V:AB123CD", t0 + 21, "T3", "AB123CD", 0.80))
            # Unvalidated readings key on normalized raw text and carry no vehicle class.
            self.assertTrue(tracker.record("U:AB!23CD", t0 + 40, "T4", "AB! 23CD", 0.30))
            self.assertFalse(tracker.record("U:AB!23CD", t0 + 41, "T5", "AB! 23CD", 0.30))
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)  # header + 3 created events (T0, T2, T4)
            self.assertEqual(lines[1], "T0,AB123CD,CAR,VALIDATED_CAR,0.8000")
            self.assertEqual(lines[-1], "T4,AB!23CD,,UNVALIDATED,0.3000")

    def test_old_keys_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = EventTracker(EventLog(Path(tmp) / "e.csv"), dedup_window_s=10.0)
            tracker.record("V:AA111AA", 0.0, "T", "AA111AA", 0.5)
            tracker.sweep(50.0)
            self.assertEqual(tracker.last_seen, {})


if __name__ == "__main__":
    unittest.main()
