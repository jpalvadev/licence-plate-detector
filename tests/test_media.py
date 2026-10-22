"""Unit tests for the plate image artifact seam (alpr/media.py, ADR-0011).

Seam: save_plate_image / filename_for — one JPG per logged Recognition event,
named ``{plate}_{raw-conf}_{sanitized-timestamp}.jpg`` so each photo pairs 1:1
with its Events log row.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

from alpr.media import context_crop, filename_for, save_plate_image


class FilenameTests(unittest.TestCase):
    def test_windows_safe_filename_from_worked_example(self):
        name = filename_for("AG841RU", 0.2136, "2026-10-21T20:11:01-03:00")
        self.assertEqual(name, "AG841RU_0.2136_2026-10-21T20-11-01-03-00.jpg")

    def test_no_colon_survives(self):
        name = filename_for("ABC123", 0.4, "2026-01-01T00:00:00+00:00")
        self.assertNotIn(":", name)
        self.assertEqual(name, "ABC123_0.4000_2026-01-01T00-00-00+00-00.jpg")


class ContextCropTests(unittest.TestCase):
    # ADR-0011 amendment: artifacts show vehicle context, so the plate box is
    # grown by pad fraction of its own size per side, clamped to the frame.

    def test_pad_grows_box_per_side(self):
        frame = np.arange(60 * 200).reshape(60, 200).astype(np.uint8)  # h=60, w=200
        crop = context_crop(frame, (40, 20, 100, 36), 0.5)  # box 60x16 -> mx=30, my=8
        self.assertEqual(crop.shape[:2], (32, 120))  # (10,12)-(130,44) region

    def test_clamps_to_frame_edges(self):
        frame = np.zeros((60, 200, 3), dtype=np.uint8)
        crop = context_crop(frame, (0, 0, 40, 10), 0.5)  # wants (-20,-5)-(60,15)
        self.assertEqual(crop.shape[:2], (15, 60))  # clamped at top-left

    def test_zero_pad_is_the_tight_box(self):
        frame = np.zeros((60, 200, 3), dtype=np.uint8)
        crop = context_crop(frame, (40, 20, 100, 36), 0.0)
        self.assertEqual(crop.shape[:2], (16, 60))


class ImageTests(unittest.TestCase):
    def test_saved_file_is_a_decodable_jpeg(self):
        crop = np.zeros((16, 64, 3), dtype=np.uint8)
        crop[:, :] = (20, 180, 60)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_plate_image(Path(tmp), "AG841RU", 0.2136, "2026-10-21T20:11:01-03:00", crop)
            self.assertEqual(path.name, "AG841RU_0.2136_2026-10-21T20-11-01-03-00.jpg")
            data = path.read_bytes()
            self.assertEqual(data[:2], b"\xff\xd8")  # JPEG magic

    def test_quality_keeps_flat_crops_faithful(self):
        crop = np.zeros((16, 64, 3), dtype=np.uint8)
        crop[:, :] = (20, 180, 60)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_plate_image(Path(tmp) / "media", "AG841RU", 0.2136, "t", crop)
            back = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(back.shape, crop.shape)
            self.assertTrue(np.abs(back.astype(int) - crop.astype(int)).max() <= 2)


if __name__ == "__main__":
    unittest.main()
