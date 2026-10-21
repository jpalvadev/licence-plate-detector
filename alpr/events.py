"""Recognition events: append-only CSV events log (ADR-0003; row timing per ADR-0006)."""

from __future__ import annotations

from pathlib import Path

HEADER = ("timestamp", "plate", "vehicle_class", "status", "confidence")
STATUS_CAR = "VALIDATED_CAR"
STATUS_MOTORCYCLE = "VALIDATED_MOTORCYCLE"
STATUS_UNVALIDATED = "UNVALIDATED"


def status_for(vehicle_class: str) -> str:
    """Map a vehicle class (``validation.classify`` output) to an Events log status."""
    if vehicle_class == "CAR":
        return STATUS_CAR
    if vehicle_class == "MOTORCYCLE":
        return STATUS_MOTORCYCLE
    return STATUS_UNVALIDATED


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
