"""Plate detector for ``yolo-v9-t-384-license-plates-end2end.onnx``.

Contract mirrors open-image-models' ``YoloV9Detector`` (MIT): BGR letterbox input (pad
114), HWC→CHW + BGR→RGB, /255 float32, batch dim. The end2end export runs NMS inside
the graph; output rows are ``[batch_idx, x1, y1, x2, y2, class_id, score]``.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from .onnx_session import create_session


def _letterbox(img: np.ndarray, new_shape: int, color: tuple[int, int, int] = (114, 114, 114)):
    """Resize with aspect ratio kept; returns (image, ratio, (dw, dh)) like open-image-models."""
    h0, w0 = img.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (dw, dh)


class PlateDetector:
    """Runs one detection pass on a BGR frame; returns plate boxes in source coordinates."""

    def __init__(self, model_path: Path, conf_thresh: float = 0.4):
        self.session = create_session(model_path, name="plate-detector")
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        # NCHW square input like [1, 3, 384, 384]; batch may be dynamic ('batch'/None).
        self.img_size = shape[-1]
        if shape[-1] != shape[-2] or not isinstance(self.img_size, int):
            msg = f"Detector model needs a square numeric input: {shape}"
            raise ValueError(msg)
        self.output_name = self.session.get_outputs()[0].name
        self.conf_thresh = conf_thresh

    def preprocess(self, frame_bgr: np.ndarray):
        im, (rx, ry), (dw, dh) = _letterbox(frame_bgr, self.img_size)
        im = im.transpose((2, 0, 1))[::-1]  # HWC→CHW, BGR→RGB
        im = (im / 255.0).astype(np.float32)
        return np.expand_dims(im, 0), (rx, ry), (dw, dh)

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Return ``[(x1, y1, x2, y2, confidence)]`` clamped to the source frame."""
        x, (rx, _ry), (dw, dh) = self.preprocess(frame_bgr)
        raw = self.session.run([self.output_name], {self.input_name: x})[0]
        rows = np.asarray(raw, dtype=np.float32)
        if rows.ndim == 3:  # [batch, n, 7]
            rows = rows[0]
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 7:
            return []
        rows = rows[rows[:, 0] == 0]  # batch index 0 (end2end exports keep the batch column)
        h, w = frame_bgr.shape[:2]
        results: list[tuple[int, int, int, int, float]] = []
        for row in rows[rows[:, 6] >= self.conf_thresh]:
            x1 = (row[1] - dw) / rx
            y1 = (row[2] - dh) / rx
            x2 = (row[3] - dw) / rx
            y2 = (row[4] - dh) / rx
            x1 = max(0, min(w - 1, round(x1)))
            y1 = max(0, min(h - 1, round(y1)))
            x2 = max(0, min(w - 1, round(x2)))
            y2 = max(0, min(h - 1, round(y2)))
            if x2 > x1 and y2 > y1:
                results.append((x1, y1, x2, y2, float(row[6])))
        return results
