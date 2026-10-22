"""Display-scale clarity % — human-facing reading quality (overlay + dashboard only).

The raw ``confidence`` in the Events log is the OCR mean per-character softmax
probability (ADR-0006); for the CCT family that number lives around 0.10–0.25
even on perfect crops, which reads as a baffling "22%". This seam maps that raw
value onto a human scale calibrated to the observed model distribution: a
typical clear reading sits at ~88–94% and a strong one at ~98–99%. Model-relative
presentation calibration, NOT a probability that the reading is correct (the
upstream benchmark, 97.9% plate acc at those confs, is the accuracy ground).
Single hard-coded table (no `.env` knob): re-anchor the curve by editing the
anchors if the production OCR family changes (e.g. to ``arg_cnn_ocr``).
"""

from __future__ import annotations

# (raw conf, clarity %) breakpoints; linear between anchors, clamped outside.
CLARITY_ANCHORS = ((0.00, 30.0), (0.10, 70.0), (0.14, 88.0), (0.18, 94.0), (0.22, 98.0), (1.00, 99.0))


def clarity(conf: float) -> float:
    """Map a raw OCR confidence to the display clarity percentage (0–100 scale)."""
    if conf <= CLARITY_ANCHORS[0][0]:
        return CLARITY_ANCHORS[0][1]
    for (c0, p0), (c1, p1) in zip(CLARITY_ANCHORS, CLARITY_ANCHORS[1:]):
        if conf <= c1:
            return p0 + (conf - c0) * (p1 - p0) / (c1 - c0)
    return CLARITY_ANCHORS[-1][1]
