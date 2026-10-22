"""Live pipeline: capture frames, detection passes at DETECT_FPS, render stream at source FPS.

Design (Q5/Q8/Q15/Q31): the capture thread fills a bounded drop-oldest buffer; the
pipeline consumer runs detection+OCR only every ``1/DETECT_FPS`` seconds but renders a
frame (reusing the overlays from the last detection pass) for every frame at the
render scale, pushing JPEGs for the MJPEG stream.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from .config import Settings, parse_roi_center_in
from .events import EventLog, status_for
from .tracker import PlateTracker
from .ingest import FrameBuffer, RtspCapture, RtspReader
from .detector import PlateDetector
from .ocr import PlateOcr
from .validation import classify, normalize
from .clarity import clarity
from .media import context_crop

LOGGER = logging.getLogger(__name__)

GREEN = (0, 200, 60)
GREY = (140, 140, 140)


class Overlay:
    """A persisted sighting overlay: source-space box + rendered label + status."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int, label: str, validated: bool):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.label = label
        self.validated = validated


class Pipeline:
    """Orchestrator for capture → detection passes → render stream + event log."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.rtsp = RtspCapture(settings)
        self.detector = PlateDetector(settings.models_dir / f"{settings.detector_model}.onnx", settings.det_conf)
        self.ocr = PlateOcr(settings.models_dir / f"{settings.ocr_model}.onnx")
        self.log = EventLog(settings.events_csv)
        self.tracker = PlateTracker(
            self.log,
            iou_thresh=settings.track_iou_thresh,
            max_misses=settings.track_max_misses,
            dedup_window_s=settings.dedup_window_s,
            include_legacy=settings.include_legacy,
            media_dir=settings.media_dir,
        )

        self.jpeg_buffer = FrameBuffer(maxlen=15)
        self.last_overlay: list[Overlay] = []
        self.last_frame: np.ndarray | None = None
        self.last_detect_mono = 0.0
        self._new_events: list[str] = []
        self._events_lock = threading.Lock()

    def start(self) -> None:
        self.rtsp.start()

    def process_frame(self, frame: np.ndarray, timestamp: float) -> None:
        """Render one frame; run a detection pass when its turn comes up."""
        now_mono = time.monotonic()
        interval = 1.0 / self.settings.detect_fps if self.settings.detect_fps > 0 else 0.0
        if now_mono - self.last_detect_mono >= interval:
            self.last_detect_mono = now_mono
            self.last_overlay = self.detection_pass(frame, timestamp)
        rendered = self.render(frame)
        jpg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])[1].tobytes()
        self.jpeg_buffer.append(jpg)
        self.last_frame = frame

    def detection_pass(self, frame: np.ndarray, timestamp: float) -> list[Overlay]:
        """Detection pass + ROI test + OCR + validation + tracker feed. Returns overlays."""
        settings = self.settings
        h, w = frame.shape[:2]
        overlays: list[Overlay] = []
        for x1, y1, x2, y2, det_conf in self.detector.detect(frame):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if not parse_roi_center_in(settings.roi, cx / w, cy / h):
                continue
            crop = self._crop(frame, (x1, y1, x2, y2))
            if crop is None:
                continue
            raw, conf = self.ocr.recognize(crop)
            if not raw:
                continue
            # ADR-0011: the artifact keeps a CONTEXT crop (MEDIA_PAD margin) so the
            # photo can show the vehicle; OCR above stays on the tight crop (Q4/Q17
            # accuracy ground — argmax pixels only).
            artifact = context_crop(frame, (x1, y1, x2, y2), settings.media_pad) or crop
            self.tracker.record(
                now=timestamp, box=(x1, y1, x2, y2), raw=raw, confidence=conf, crop=artifact
            )
            text = normalize(raw)
            format_match = classify(text, include_legacy=settings.include_legacy)
            if format_match:
                status = status_for(format_match)
                label = f"{text} {status} {clarity(conf):.0f}%"  # display scale, see .clarity
                overlays.append(Overlay(x1, y1, x2, y2, label, True))
            else:
                overlays.append(Overlay(x1, y1, x2, y2, f"? {text}", False))
        self._report_events(self.tracker.end_pass(timestamp))
        return overlays

    def _report_events(self, closed: list[tuple[str, str, str, float]]) -> None:
        """Console feedback for Recognition events logged at close (via the tracker)."""
        if not closed:
            return
        with self._events_lock:
            for ts, plate, status, conf in closed:
                self._new_events.append(f"{ts} {plate} {status} {conf:.2f}")

    def new_events(self) -> list[str]:
        with self._events_lock:
            drained = self._new_events
            self._new_events = []
            return drained

    def take_jpegs(self) -> list[bytes]:
        return self.jpeg_buffer.take()

    def run(self) -> None:
        """Main consumer loop (run in its own thread)."""
        self.start()
        offline_push_mono = 0.0
        offline_flushed = False
        while True:
            frames = self.rtsp.buffer.take()
            if frames:
                for frame, timestamp in frames:
                    self.process_frame(frame, timestamp)
                offline_flushed = False
                continue
            time.sleep(0.01)
            now_mono = time.monotonic()
            if self.rtsp.offline:
                if not offline_flushed:  # stream died: log what is still open (ADR-0006)
                    offline_flushed = True
                    self._report_events(self.tracker.flush(time.time()))
                if now_mono - offline_push_mono >= 0.25:
                    offline_push_mono = now_mono
                    self._push_offline_frame()

    def _push_offline_frame(self) -> None:
        """Keep the stream alive with a banner frame so browsers refresh + show offline."""
        frame = self.last_frame
        if frame is None:
            s = self.settings
            frame = np.zeros((s.video_h, s.video_w, 3), dtype=np.uint8)
        rendered = self.render(frame, banner=self.rtsp.offline_reason)
        jpg = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])[1].tobytes()
        self.jpeg_buffer.append(jpg)

    def render(self, frame: np.ndarray, banner: str | None = None) -> np.ndarray:
        """Downscale (design Q31: native source stays, stream is RENDER_WxRENDER_H)."""
        settings = self.settings
        out = cv2.resize(frame, (settings.render_w, settings.render_h), interpolation=cv2.INTER_AREA)
        sx = settings.render_w / settings.video_w
        sy = settings.render_h / settings.video_h
        for ov in self.last_overlay:
            color = GREEN if ov.validated else GREY
            pt1 = (round(ov.x1 * sx), round(ov.y1 * sy))
            pt2 = (round(ov.x2 * sx), round(ov.y2 * sy))
            cv2.rectangle(out, pt1, pt2, color, 2)
            size = max(0.45, min(0.8, (pt2[1] - pt1[1]) / 42.0))
            cv2.putText(out, ov.label, (pt1[0], pt1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
        if banner:
            text = f"STREAM OFFLINE ({banner})"
            cv2.rectangle(out, (0, 0), (out.shape[1], 34), (30, 30, 30), -1)
            cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 1, cv2.LINE_AA)
        return out

    def _crop(self, frame: np.ndarray, box: tuple[int, int, int, int], margin_frac: float = 0.03):
        """Crop the plate box with a small margin; None when too small to read."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        mx = max(2, round((x2 - x1) * margin_frac))
        my = max(2, round((y2 - y1) * margin_frac))
        x1c, y1c = max(0, x1 - mx), max(0, y1 - my)
        x2c, y2c = min(w, x2 + mx), min(h, y2 + my)
        if x2c - x1c < 10 or y2c - y1c < 6:
            return None
        return frame[y1c:y2c, x1c:x2c]
