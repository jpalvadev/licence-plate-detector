"""Unit tests for config parsing (run: python -m unittest discover -s tests)."""

import os
import tempfile
import unittest
from pathlib import Path

from alpr.config import load_settings, parse_roi_center_in, _roi, _rtsp_url


class RoiTests(unittest.TestCase):
    def test_empty_is_full_frame(self):
        self.assertEqual(_roi(""), (0.0, 0.0, 1.0, 1.0))

    def test_valid_rect(self):
        self.assertEqual(_roi("0.15,0.35,0.85,0.75"), (0.15, 0.35, 0.85, 0.75))

    def test_rejects_bad_rects(self):
        with self.assertRaises(ValueError):
            _roi("0.5,0.5")
        with self.assertRaises(ValueError):
            _roi("0.8,0.1,0.2,0.9")  # left >= right

    def test_center_test(self):
        roi = (0.25, 0.25, 0.75, 0.75)
        self.assertTrue(parse_roi_center_in(roi, 0.5, 0.5))
        self.assertFalse(parse_roi_center_in(roi, 0.9, 0.5))


class RtspUrlTests(unittest.TestCase):
    def test_plain_endpoint(self):
        self.assertEqual(_rtsp_url("1.2.3.4:554/mystream", "", ""), "rtsp://1.2.3.4:554/mystream")

    def test_credentials_percent_encoded(self):
        url = _rtsp_url("host:554/a", "a b", "p@ss")
        self.assertEqual(url, "rtsp://a%20b:p%40ss@host:554/a")

    def test_scheme_prefix_stripped(self):
        self.assertEqual(_rtsp_url("rtsp://host:554/a", "", ""), "rtsp://host:554/a")


class LoadSettingsTests(unittest.TestCase):
    def test_reads_env_file(self):
        keys = [
            "RTSP_ENDPOINT", "RTSP_USER", "RTSP_PASS", "VIDEO_W", "VIDEO_H", "DETECT_FPS",
            "DEDUP_WINDOW_S", "EVENTS_CSV", "ROI", "DET_CONF_THRESH", "DETECTOR_MODEL",
            "OCR_MODEL", "HTTP_HOST", "HTTP_PORT", "RENDER_W", "RENDER_H", "JPEG_QUALITY",
            "MODELS_DIR", "QUEUE_MAXLEN",
        ]
        for key in keys:
            os.environ.pop(key, None)
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "RTSP_ENDPOINT=cam.local:554/live\nRTSP_USER=u1\nRTSP_PASS=s e@cret\n"
                "DETECT_FPS=2\nROI=0.1,0.2,0.9,0.8\nHTTP_PORT=9000\nTRACK_IOU_THRESH=0.5\nTRACK_MAX_MISSES=2\n",
                encoding="utf-8",
            )
            settings = load_settings(env_file)
            self.assertEqual(settings.rtsp_url, "rtsp://u1:s%20e%40cret@cam.local:554/live")
            self.assertEqual(settings.video_w, 1920)
            self.assertEqual(settings.detect_fps, 2.0)
            self.assertEqual(settings.roi, (0.1, 0.2, 0.9, 0.8))
            self.assertEqual(settings.http_port, 9000)
            self.assertEqual(settings.ocr_model, "cct_s_v2_global")
            self.assertEqual(settings.track_iou_thresh, 0.5)
            self.assertEqual(settings.track_max_misses, 2)

    def test_requires_endpoint(self):
        for key in ["RTSP_ENDPOINT", "RTSP_USER", "RTSP_PASS"]:
            os.environ.pop(key, None)
        with tempfile.TemporaryDirectory() as tmp:
            # Hermetic: an existing, endpoint-free env file (dotenv discovery from a test
            # file under the repo root would otherwise pick up the operator's real .env).
            env_file = Path(tmp) / "blank.env"
            env_file.write_text("VIDEO_W=1920\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(env_file)
            # Debug mode tolerates a missing endpoint (require_endpoint=False).
            settings = load_settings(env_file, require_endpoint=False)
            self.assertEqual(settings.rtsp_url, "rtsp://")


if __name__ == "__main__":
    unittest.main()
