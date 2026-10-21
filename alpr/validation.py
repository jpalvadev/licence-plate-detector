"""Raw reading → normalized text → format match (design Q9; format set per ADR-0008).

Normalization: uppercase, strip spaces/dashes/underscores. The car format is tried
first; the legacy car format is tried only when the format set includes it.
Motorcycles are out of the domain in either era (ADR-0007). Alphabet is A–Z only
(no Ñ): 26^4 × 10^3 ≈ 450M keys per DNRPA.
"""

from __future__ import annotations

import re

CAR_RE = re.compile(r"[A-Z]{2}[0-9]{3}[A-Z]{2}")
LEGACY_RE = re.compile(r"[A-Z]{3}[0-9]{3}")


def normalize(raw: str) -> str:
    """Uppercase and strip spaces, dashes and underscores (the only allowed separators)."""
    return re.sub(r"[ \-_]", "", raw.upper())


def classify(text: str, *, include_legacy: bool = False) -> str | None:
    """Car first; legacy car second only when the format set includes it.

    ``include_legacy`` is the format set resolved from ``PLATE_FORMATS`` (ADR-0008):
    the legacy ``ABC 123`` shape classifies as ``LEGACY`` only when it is on.
    ``None`` when no format in the set matches fully.
    """
    if CAR_RE.fullmatch(text):
        return "CAR"
    if include_legacy and LEGACY_RE.fullmatch(text):
        return "LEGACY"
    return None
