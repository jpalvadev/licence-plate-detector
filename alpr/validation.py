"""Raw reading → normalized text → Vehicle class (design Q9).

Normalization: uppercase, strip spaces/dashes/underscores. Car format is tried first,
motorcycle second. Alphabet is A–Z only (no Ñ): 26^4 × 10^3 ≈ 450M keys per DNRPA.
"""

from __future__ import annotations

import re

CAR_RE = re.compile(r"[A-Z]{2}[0-9]{3}[A-Z]{2}")
MOTORCYCLE_RE = re.compile(r"[A-Z][0-9]{3}[A-Z]{3}")


def normalize(raw: str) -> str:
    """Uppercase and strip spaces, dashes and underscores (the only allowed separators)."""
    return re.sub(r"[ \-_]", "", raw.upper())


def classify(text: str) -> str | None:
    """Car first, motorcycle second; ``None`` when neither matches fully."""
    if CAR_RE.fullmatch(text):
        return "CAR"
    if MOTORCYCLE_RE.fullmatch(text):
        return "MOTORCYCLE"
    return None
