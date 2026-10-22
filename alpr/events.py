"""Recognition events: append-only CSV events log (ADR-0003; row timing per ADR-0006;
validated-only rows and no class column per ADR-0009)."""

from __future__ import annotations

from pathlib import Path

HEADER = ("timestamp", "patente", "status", "confidence")
STATUS_CAR = "PATENTE_MERCOSUR"
STATUS_LEGACY = "PATENTE_ALFANUM"


def status_for(format_match: str) -> str:
    """Map a format match (``validation.classify`` output) to an Events log status.

    ADR-0009: only validated plates reach the log, so anything that is not a format
    match has no status to return — fail fast like the other config seams.
    """
    if format_match == "CAR":
        return STATUS_CAR
    if format_match == "LEGACY":
        return STATUS_LEGACY
    raise ValueError(f"not a format match: {format_match!r}")


class EventLog:
    """Append-only CSV on disk; one row written when a validated Recognition event closes."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            with path.open("w", newline="", encoding="utf-8") as fh:
                fh.write(",".join(HEADER) + "\n")

    def append(self, timestamp: str, plate: str, status: str, confidence: float) -> None:
        row = [timestamp, plate, status, f"{confidence:.4f}"]
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            fh.write(",".join(row) + "\n")
            fh.flush()

    def tail(self, n: int = 15) -> list[str]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return lines[-n:]
