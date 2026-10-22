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
