"""Whole-plate OCR over fast-plate-ocr ONNX models (CCT family or the 7-slot AR CNN).

Contract mirrors ``fast_plate_ocr.inference.plate_recognizer`` (MIT): input tensor is
NHWC uint8 (the graph normalizes pixels internally); plate-classification heads are
fixed slots — output reshapes to ``(slots, len(alphabet))`` and each slot's argmax
gives a character (trailing pad chars stripped). Input size/channels are read from the
model's own input tensor so ``cct_s_v2_global`` (64x128 RGB) and ``arg_cnn_ocr``
(70x140 whatever channel count it exports) both work.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from .onnx_session import create_session

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PAD_CHAR = "_"


class PlateOcr:
    """Recognize text on one cropped plate; returns ``(raw_reading, confidence)``."""

    def __init__(self, model_path: Path):
        self.session = create_session(model_path, name="plate-ocr")
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        if len(shape) != 4:
            msg = f"OCR model input must be (N, H, W, C): {shape}"
            raise ValueError(msg)
        self.in_h, self.in_w, self.in_c = (shape[1], shape[2], shape[3])
        for dim in (self.in_h, self.in_w, self.in_c):
            if not isinstance(dim, int):
                msg = f"OCR model needs static H/W/C dims: {shape}"
                raise ValueError(msg)
        outputs = [o.name for o in self.session.get_outputs()]
        self.plate_output = "plate" if "plate" in outputs else outputs[0]
        self.vocab_size = len(ALPHABET)

    def preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        if self.in_c == 1:
            img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
            img = img[..., None]
        else:
            img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = img[..., None]
        return np.asarray(img, dtype=np.uint8)[None, ...]

    def recognize(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        """Decode one crop: returns ``(raw_reading, mean_char_confidence)``."""
        raw = self.session.run([self.plate_output], {self.input_name: self.preprocess(crop_bgr)})[0]
        probs = np.asarray(raw, dtype=np.float32).reshape(-1)
        slots = probs.size // self.vocab_size
        if slots * self.vocab_size != probs.size:
            msg = f"OCR output size {probs.size} not divisible by vocab {self.vocab_size}"
            raise ValueError(msg)
        probs = probs.reshape(slots, self.vocab_size)
        picks = np.argmax(probs, axis=-1)
        text = "".join(ALPHABET[i] for i in picks).rstrip(PAD_CHAR)
        if not text:
            return "", 0.0
        # Confidence: mean probability of the chosen character over the non-padding slots.
        conf = float(np.mean(probs[: len(text), picks[: len(text)]]))
        return text, conf
