"""Single-source configuration loaded from a ``.env`` file (python-dotenv)."""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Everything tunable about the app, populated from environment variables."""

    rtsp_url: str
    video_w: int
    video_h: int
    detect_fps: float
    dedup_window_s: float
    track_iou_thresh: float
    track_max_misses: int
    include_legacy: bool
    events_csv: Path
    media_dir: Path
    media_pad: float
    roi: tuple[float, float, float, float]
    det_conf: float
    detector_model: str
    ocr_model: str
    http_host: str
    http_port: int
    render_w: int
    render_h: int
    jpeg_quality: int
    models_dir: Path
    queue_maxlen: int


def load_settings(env_path: Path = Path(".env"), *, require_endpoint: bool = True) -> Settings:
    """Read the ``.env`` file into a typed :class:`Settings`. Raises on missing config."""
    load_dotenv(dotenv_path=env_path if env_path.exists() else None, override=False)

    endpoint = os.environ.get("RTSP_ENDPOINT", "").strip()
    if require_endpoint and not endpoint:
        msg = "RTSP_ENDPOINT is not set — copy .env.example to .env and fill it in."
        raise ValueError(msg)

    return Settings(
        rtsp_url=_rtsp_url(endpoint, os.environ.get("RTSP_USER", ""), os.environ.get("RTSP_PASS", "")),
        video_w=_int("VIDEO_W", 1920),
        video_h=_int("VIDEO_H", 1080),
        detect_fps=_float("DETECT_FPS", 3.0),
        dedup_window_s=_float("DEDUP_WINDOW_S", 10.0),
        track_iou_thresh=_float("TRACK_IOU_THRESH", 0.3),
        track_max_misses=_int("TRACK_MAX_MISSES", 6),
        include_legacy=_plate_formats("PLATE_FORMATS", "MERCOSUR_ONLY"),
        events_csv=Path(os.environ.get("EVENTS_CSV", "events.csv")),
        media_dir=Path(os.environ.get("MEDIA_DIR", "media")),
        media_pad=_float("MEDIA_PAD", 0.6),
        roi=_roi(os.environ.get("ROI", "")),
        det_conf=_float("DET_CONF_THRESH", 0.4),
        detector_model=os.environ.get("DETECTOR_MODEL", "yolo-v9-s-608-license-plates-end2end").strip(),
        ocr_model=os.environ.get("OCR_MODEL", "cct_s_v2_global").strip(),
        http_host=os.environ.get("HTTP_HOST", "0.0.0.0").strip(),
        http_port=_int("HTTP_PORT", 8080),
        render_w=_int("RENDER_W", 960),
        render_h=_int("RENDER_H", 540),
        jpeg_quality=_int("JPEG_QUALITY", 70),
        models_dir=Path(os.environ.get("MODELS_DIR", "models")),
        queue_maxlen=_int("QUEUE_MAXLEN", 10),
    )


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _plate_formats(name: str, default: str) -> bool:
    """``PLATE_FORMATS`` enum (ADR-0008) → whether the format set includes legacy cars."""
    raw = os.environ.get(name, "").strip() or default
    if raw == "MERCOSUR_ONLY":
        return False
    if raw == "MERCOSUR_PLUS_LEGACY":
        return True
    msg = f"PLATE_FORMATS must be MERCOSUR_ONLY or MERCOSUR_PLUS_LEGACY (got {raw!r})"
    raise ValueError(msg)


def _roi(text: str) -> tuple[float, float, float, float]:
    """Parse ``left,top,right,bottom`` normalized fractions; empty means full frame."""
    if not text.strip():
        return (0.0, 0.0, 1.0, 1.0)
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 4:
        msg = f"ROI must be 'left,top,right,bottom' (got {text!r})"
        raise ValueError(msg)
    left, top, right, bottom = parts
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        msg = f"ROI fractions must satisfy 0 <= left < right <= 1 (and same for top/bottom): {text!r}"
        raise ValueError(msg)
    return (left, top, right, bottom)


def _rtsp_url(endpoint: str, user: str, passwd: str) -> str:
    """Build the ffmpeg input URL from endpoint + split credentials.

    ``RTSP_ENDPOINT`` carries no scheme and no credentials (design decision Q4);
    ``rtsp://USER:PASS@host:port/path`` is assembled here, percent-encoding the
    credentials so they cannot break the URL structure.
    """
    bare = endpoint.strip()
    for scheme in ("rtsp://", "rtspu://"):
        if bare.lower().startswith(scheme):
            bare = bare[len(scheme):]
            break
    auth = ""
    if user or passwd:
        auth = f"{urllib.parse.quote(user, safe='')}:{urllib.parse.quote(passwd, safe='')}@"
    return f"rtsp://{auth}{bare}"


def parse_roi_center_in(roi: Sequence[float], cx_norm: float, cy_norm: float) -> bool:
    """True when a sighting's normalized box center falls inside the ROI (design Q8)."""
    left, top, right, bottom = roi
    return left <= cx_norm <= right and top <= cy_norm <= bottom
