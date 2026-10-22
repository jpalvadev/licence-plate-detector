"""Unit tests for the plate-track tracker (run: python -m unittest discover -s tests).

Seam: the public PlateTracker interface (record / end_pass / flush). A closed
track is a row in the Events log CSV only when its consensus matches an active
format (ADR-0009); ADR-0006 gives event identity (plate track).
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

from alpr.events import STATUS_CAR, STATUS_LEGACY, EventLog
from alpr.media import filename_for
from alpr.tracker import PlateTracker


def iso(now: float) -> str:
    """Same rendering the tracker uses for first-sighting timestamps."""
    return datetime.fromtimestamp(now, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def rows(log: EventLog) -> list[list[str]]:
    lines = log.path.read_text(encoding="utf-8").splitlines()
    return [line.split(",") for line in lines[1:]]


class PlateTrackLifecycleTests(unittest.TestCase):
    def test_open_track_writes_no_row_until_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            self.assertTrue(tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AB123CD", confidence=0.5))
            for i in range(0, 4):  # passes 0..3: only 3 missed passes after the match on pass 0
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(rows(log), [])

    def test_track_closes_after_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AB123CD", confidence=0.5)
            for i in range(0, 5):  # pass 4 is the 4th missed pass: tolerance exceeded, track closes
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "AB123CD", STATUS_CAR, "0.5000"]],
            )

    def test_exact_string_reemits_after_window(self):
        # The guard is a window, not a memory: the same consensus string closing
        # after dedup_window_s expires must log a fresh row.
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            for i in range(12):
                if i == 0:
                    tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AB123CD", confidence=0.5)
                if i == 6:
                    tracker.record(now=1024.0, box=(600, 0, 700, 30), raw="AB123CD", confidence=0.5)
                tracker.end_pass(now=1000.0 + 4.0 * i)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "AB123CD", STATUS_CAR, "0.5000"], [iso(1024.0), "AB123CD", STATUS_CAR, "0.5000"]],
            )

    def test_moving_box_stays_one_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            readings = [("AD537NP", 0.30), ("AD537XP", 0.40), ("AD537XP", 0.45), ("AD537XP", 0.35), ("AB537XP", 0.30)]
            for i, (raw, conf) in enumerate(readings):
                tracker.record(now=1000.0 + i, box=(2 * i, 0, 100 + 2 * i, 30), raw=raw, confidence=conf)
                tracker.end_pass(now=1000.0 + i)
            for i in range(5, 9):  # drift away until tolerance closes the single track
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "AD537XP", STATUS_CAR, "0.4500"]],
            )

    def test_jittery_box_stays_one_track(self):
        # Real detector boxes for one plate jump (plate vs bumper crops): raw IoU of
        # consecutive boxes is < 0.3, but dilated IoU >= 0.3 keeps it one track.
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            boxes = [(0, 0, 100, 30), (60, 2, 160, 32), (120, 0, 220, 30)]
            texts = ["AD537XP", "AD537NP", "AD537XP"]
            for i in range(3):
                tracker.record(now=1000.0 + i, box=boxes[i], raw=texts[i], confidence=0.4)
                tracker.end_pass(now=1000.0 + i)
            for i in range(3, 7):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "AD537XP", STATUS_CAR, "0.4000"]],
            )

    def test_same_pass_duplicate_box_merges(self):
        # The detector double-boxes one plate sometimes: a second overlapping
        # sighting in the SAME pass must attach to the same track, one row.
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AG010DT", confidence=0.5)
            tracker.record(now=1000.0, box=(5, 0, 95, 30), raw="010DT", confidence=0.4)
            tracker.end_pass(now=1000.0)
            for i in range(1, 6):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "AG010DT", STATUS_CAR, "0.5000"]],
            )

    def test_two_cars_side_by_side_stay_separate(self):
        # Dilation must not merge neighbouring cars: boxes 300 px apart never overlap.
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AB123CD", confidence=0.5)
            tracker.record(now=1000.0, box=(300, 0, 400, 30), raw="XY789ZW", confidence=0.6)
            tracker.end_pass(now=1000.0)
            tracker.record(now=1001.0, box=(60, 2, 160, 32), raw="AB123CD", confidence=0.5)
            tracker.end_pass(now=1001.0)
            for i in range(2, 7):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [
                    [iso(1000.0), "XY789ZW", STATUS_CAR, "0.6000"],
                    [iso(1000.0), "AB123CD", STATUS_CAR, "0.5000"],
                ],
            )

    def test_disjoint_boxes_are_separate_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AB123CD", confidence=0.5)
            tracker.record(now=1000.0, box=(500, 0, 600, 30), raw="XY789ZW", confidence=0.6)
            for i in range(0, 5):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [
                    [iso(1000.0), "AB123CD", STATUS_CAR, "0.5000"],
                    [iso(1000.0), "XY789ZW", STATUS_CAR, "0.6000"],
                ],
            )


class RecognitionConsensusTests(unittest.TestCase):
    def test_validated_reading_beats_unvalidated_sightings(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="517 RP", confidence=0.9)
            tracker.end_pass(now=1000.0)
            tracker.record(now=1001.0, box=(2, 0, 102, 30), raw="AD537XP", confidence=0.4)
            for i in range(1, 6):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "AD537XP", STATUS_CAR, "0.4000"]],
            )

    def test_plurality_reading_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            for i, raw in enumerate(["AB123CE", "AB123CD", "AB123CD"]):
                tracker.record(now=1000.0 + i, box=(2 * i, 0, 100 + 2 * i, 30), raw=raw, confidence=0.5)
                tracker.end_pass(now=1000.0 + i)
            for i in range(3, 7):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "AB123CD", STATUS_CAR, "0.5000"]],
            )

    def test_plurality_tie_broken_by_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AB123CD", confidence=0.5)
            tracker.end_pass(now=1000.0)
            tracker.record(now=1001.0, box=(2, 0, 102, 30), raw="AB123CE", confidence=0.7)
            for i in range(1, 6):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "AB123CE", STATUS_CAR, "0.7000"]],
            )

    def test_unvalidated_consensus_logs_no_row(self):
        # ADR-0009: a track whose consensus matches no format produces no entry at all.
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            for i, raw in enumerate(["517RP", "517 RP", "5L7RP"]):
                tracker.record(now=1000.0 + i, box=(2 * i, 0, 100 + 2 * i, 30), raw=raw, confidence=0.5 + 0.2 * i)
                tracker.end_pass(now=1000.0 + i)
            for i in range(3, 7):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(rows(log), [])


class FormatSetTests(unittest.TestCase):
    # The format set (ADR-0008) is one knob; the observable effect at this seam is
    # which rows the SAME sightings produce.
    def test_legacy_shape_logs_no_row_under_mercosur_only(self):
        # ADR-0009: knob off — the legacy consensus matches no active format, so nothing logs.
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="ABC 123", confidence=0.7)
            for i in range(0, 5):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(rows(log), [])

    def test_legacy_row_under_plus_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0, include_legacy=True)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="ABC 123", confidence=0.7)
            for i in range(0, 5):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "ABC123", STATUS_LEGACY, "0.7000"]],
            )

    def test_consensus_prefers_validated_legacy_reading(self):
        # Knob on: a legacy-shaped reading counts as validated for consensus, so it wins
        # over a higher-confidence unvalidated sighting of the same track.
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0, include_legacy=True)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="ABC12", confidence=0.9)
            tracker.end_pass(now=1000.0)
            tracker.record(now=1001.0, box=(2, 0, 102, 30), raw="ABC 123", confidence=0.4)
            for i in range(1, 6):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "ABC123", STATUS_LEGACY, "0.4000"]],
            )

    def test_mercosur_car_still_wins_over_legacy_shape(self):
        # Knob on: a true Mercosur reading and a legacy-shaped misread of the same
        # track are both validated; equal counts tie-break by confidence, so the
        # higher-confidence sighting wins — deterministic, no special-casing.
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0, include_legacy=True)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AB123CD", confidence=0.5)
            tracker.end_pass(now=1000.0)
            tracker.record(now=1001.0, box=(2, 0, 102, 30), raw="ABC123", confidence=0.4)
            for i in range(1, 6):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "AB123CD", STATUS_CAR, "0.5000"]],
            )


class DedupGuardTests(unittest.TestCase):
    def test_exact_string_within_window_suppresses_second_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AB123CD", confidence=0.5)
            for i in range(0, 5):  # closes at pass 4 (now=1004)
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(len(rows(log)), 1)
            tracker.record(now=1005.0, box=(800, 0, 900, 30), raw="ab 123 cd", confidence=0.55)
            for i in range(5, 10):  # split track of the same car closes at pass 9 (now=1009)
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(len(rows(log)), 1)  # guard: same exact string inside the window


class StreamLossFlushTests(unittest.TestCase):
    def test_flush_closes_open_tracks_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AG517RP", confidence=0.8)
            closed = tracker.flush(now=1001.5)  # stream died: log what is still open
            self.assertEqual(len(closed), 1)
            self.assertEqual(
                rows(log),
                [[iso(1000.0), "AG517RP", STATUS_CAR, "0.8000"]],
            )

    def test_flush_on_dead_stream_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AB123CD", confidence=0.5)
            for i in range(0, 5):
                tracker.end_pass(now=1000.0 + i)
            self.assertEqual(tracker.flush(now=1005.0), [])  # already closed: nothing re-written
            self.assertEqual(len(rows(log)), 1)


class PlateImageArtifactTests(unittest.TestCase):
    # ADR-0011: every logged event gets exactly one Plate image artifact — a JPG
    # named after its row's fields, from the best-confidence sighting the
    # consensus winner was voted from (never a higher-conf crop of another text).

    def _crops(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        red = np.full((16, 64, 3), (0, 0, 255), dtype=np.uint8)
        green = np.full((16, 64, 3), (0, 200, 0), dtype=np.uint8)
        blue = np.full((16, 64, 3), (255, 0, 0), dtype=np.uint8)
        return red, green, blue

    def test_logged_event_writes_image_of_best_winner_crop(self):
        red, green, blue = self._crops()
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            media = Path(tmp) / "media"
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0, media_dir=media)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AG841RU", confidence=0.10, crop=red)
            tracker.record(now=1001.0, box=(2, 0, 102, 30), raw="AG841RU", confidence=0.22, crop=green)
            tracker.record(now=1002.0, box=(4, 0, 104, 30), raw="AG841RV", confidence=0.30, crop=blue)
            closed = tracker.flush(now=1003.0)
            ts, text, status, conf = closed[0]
            self.assertEqual(text, "AG841RU")  # plurality winner despite AG841RV's higher conf
            files = sorted(media.glob("*.jpg"))
            self.assertEqual([f.name for f in files], [filename_for(text, conf, ts)])
            back = cv2.imdecode(np.frombuffer(files[0].read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(tuple(int(v) for v in back[8, 32]), (0, 200, 0))  # the 0.22 green, not blue/red

    def test_unvalidated_event_writes_no_image(self):
        _, green, _ = self._crops()
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            media = Path(tmp) / "media"
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0, media_dir=media)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="517RP", confidence=0.5, crop=green)
            self.assertEqual(tracker.flush(now=1001.0), [])
            self.assertFalse(media.exists())  # no row, no artifact

    def test_dedup_suppression_skips_second_image(self):
        red, green, _ = self._crops()
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            media = Path(tmp) / "media"
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0, media_dir=media)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AG841RU", confidence=0.5, crop=green)
            tracker.flush(now=1001.0)
            self.assertEqual(len(rows(log)), 1)
            tracker.record(now=1002.0, box=(800, 0, 900, 30), raw="AG841RU", confidence=0.55, crop=red)
            tracker.flush(now=1003.0)
            self.assertEqual(len(rows(log)), 1)  # guard still suppresses the row
            self.assertEqual(len(list(media.glob("*.jpg"))), 1)  # and no artifact for it

    def test_no_image_when_media_dir_unset(self):
        _, green, _ = self._crops()
        with tempfile.TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.csv")
            tracker = PlateTracker(log, iou_thresh=0.3, max_misses=3, dedup_window_s=10.0)
            tracker.record(now=1000.0, box=(0, 0, 100, 30), raw="AG841RU", confidence=0.5, crop=green)
            self.assertEqual(len(tracker.flush(now=1001.0)), 1)
            self.assertFalse((Path(tmp) / "media").exists())


if __name__ == "__main__":
    unittest.main()
