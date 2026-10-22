"""Plate image artifacts: one JPG per logged Recognition event (ADR-0011).

Display evidence only — the Events log stays the ground truth. Filename
``{plate}_{raw-conf}_{sanitized-timestamp}.jpg`` pairs each photo 1:1 with its
log row (raw 4-decimal conf, ISO timestamp with ``:`` -> ``-`` because Windows
forbids colons in filenames).
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

JPEG_QUALITY = 95  # evidence photos: near-lossless, one-line re-anchor if ever needed


def filename_for(plate: str, confidence: float, timestamp: str) -> str:
    """Artifact name matching the Events log row it belongs to."""
    return f"{plate}_{confidence:.4f}_{timestamp.replace(':', '-')}.jpg"


def context_crop(frame: np.ndarray, box: tuple[float, float, float, float], pad_frac: float) -> np.ndarray | None:
    """Plate box grown by ``pad_frac`` of its own size per side (ADR-0011 amendment:
    artifacts show vehicle context, not just rubber), clamped to the frame;
    None when the grown box is degenerate."""
    pad = max(0.0, pad_frac)  # negative knobs would shrink the box — not a thing
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (round(v) for v in box)
    mx, my = round((x2 - x1) * pad), round((y2 - y1) * pad)
    x1c, y1c = max(0, x1 - mx), max(0, y1 - my)
    x2c, y2c = min(w, x2 + mx), min(h, y2 + my)
    if x2c - x1c < 2 or y2c - y1c < 2:
        return None
    return frame[y1c:y2c, x1c:x2c]


def save_plate_image(
    media_dir: Path, plate: str, confidence: float, timestamp: str, crop: np.ndarray
) -> Path:
    """JPEG-encode one plate crop at JPEG_QUALITY into ``media_dir``; return the written path."""
    ok, buf = cv2.imencode(".jpg", crop, (int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY))
    if not ok:
        raise RuntimeError(f"JPEG encoding failed for plate {plate!r}")
    path = Path(media_dir) / filename_for(plate, confidence, timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())
    return path
