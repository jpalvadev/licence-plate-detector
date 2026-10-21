"""Recognition events: append-only CSV events log (ADR-0003; row timing per ADR-0006)."""

from __future__ import annotations

from pathlib import Path

HEADER = ("timestamp", "plate", "vehicle_class", "status", "confidence")
STATUS_CAR = "VALIDATED_CAR"
STATUS_LEGACY = "VALIDATED_LEGACY"
STATUS_UNVALIDATED = "UNVALIDATED"


def status_for(format_match: str) -> str:
    """Map a format match (``validation.classify`` output) to an Events log status."""
    if format_match == "CAR":
        return STATUS_CAR
    if format_match == "LEGACY":
        return STATUS_LEGACY
    return STATUS_UNVALIDATED


def class_for(format_match: str) -> str:
    """Events log ``vehicle_class`` column: only the car format proves a class.

    The legacy match is a format, not a class — car and cargo variants shared those
    characters (ADR-0008) — so it leaves the column empty.
    """
    return format_match if format_match == "CAR" else ""


class EventLog:
    """Append-only CSV on disk; one row written when a Recognition event closes."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            with path.open("w", newline="", encoding="utf-8") as fh:
                fh.write(",".join(HEADER) + "\n")

    def append(self, timestamp: str, plate: str, vehicle_class: str, status: str, confidence: float) -> None:
        row = [timestamp, plate, vehicle_class, status, f"{confidence:.4f}"]
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            fh.write(",".join(row) + "\n")
            fh.flush()

    def tail(self, n: int = 15) -> list[str]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return lines[-n:]
