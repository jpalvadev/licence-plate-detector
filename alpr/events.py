"""Recognition events: dedup window tracker + append-only CSV events log (ADR-0003)."""

from __future__ import annotations

from pathlib import Path

from .validation import classify, normalize

HEADER = ("timestamp", "plate", "vehicle_class", "status", "confidence")
STATUS_CAR = "VALIDATED_CAR"
STATUS_MOTORCYCLE = "VALIDATED_MOTORCYCLE"
STATUS_UNVALIDATED = "UNVALIDATED"


class EventLog:
    """Append-only CSV on disk; one row written at event creation, never rewritten."""

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


class EventTracker:
    """Turns sightings into Recognition events (design Q6/Q15/Q20).

    Keyed by the validated plate string (or normalized raw text for unvalidated
    readings): the same key seen again within the dedup window extends the existing
    event (no new CSV row; overlay confidence may upgrade), while a sighting after the
    window starts a new event and appends a row.
    """

    def __init__(self, log: EventLog, dedup_window_s: float):
        self.log = log
        self.dedup_window_s = dedup_window_s
        self.last_seen: dict[str, float] = {}

    def sweep(self, now: float) -> None:
        """Drop keys whose last sighting aged past the window (bounded memory)."""
        for key in [k for k, t in self.last_seen.items() if now - t > self.dedup_window_s]:
            del self.last_seen[key]

    def record(self, key: str, now: float, timestamp: str, raw: str, confidence: float) -> bool:
        """Log a sighting; returns True when this sighting created a new event."""
        self.sweep(now)
        if key in self.last_seen:
            self.last_seen[key] = now
            return False
        self.last_seen[key] = now
        text = normalize(raw)
        vehicle_class = classify(text)
        if vehicle_class:
            status = STATUS_CAR if vehicle_class == "CAR" else STATUS_MOTORCYCLE
            plate = text
        else:
            status = STATUS_UNVALIDATED
            plate = text
        self.log.append(timestamp, plate, vehicle_class or "", status, confidence)
        return True
