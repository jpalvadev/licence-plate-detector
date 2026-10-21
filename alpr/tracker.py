"""Plate tracks: the spatial identity of Recognition events (ADR-0006).

A sighting (a box + an OCR reading from one detection pass) extends the open plate
track whose last box overlaps it most when the IoU reaches ``iou_thresh``; an
unmatched sighting opens a new track. A track tolerates up to ``max_misses``
consecutive detection passes in which it was not seen; exceeding that closes it —
as does stream loss (:meth:`flush`) — and one closed track yields exactly one
Events log row carrying the plurality reading of its sightings (validated
readings preferred, ties broken by higher confidence). ``dedup_window_s`` is only
a last-line guard: a closed event's exact consensus string suppresses further
identical rows inside the window, plugging duplicate rows from split tracks.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .events import EventLog, class_for, status_for
from .validation import classify, normalize

# IoU is computed on boxes dilated by this fraction of their own size on each side:
# real detector boxes for one plate jump around (plate vs bumper crops), and raw IoU
# on small boxes fragments tracks on moving footage (integration finding, ADR-0006).
_IOU_DILATE_FRAC = 0.5


class _Sighting:
    """One detection-pass sighting accumulated on a track: normalized text + confidence."""

    def __init__(self, text: str, confidence: float):
        self.text = text
        self.confidence = confidence


class _Track:
    """An open plate track: last box, pass bookkeeping and all sightings so far."""

    def __init__(self, box: tuple[float, float, float, float], now: float, sighting: _Sighting):
        self.box = box
        self.first_now = now
        self.last_pass = -1
        self.sightings: list[_Sighting] = [sighting]


class PlateTracker:
    """Turns sightings into Recognition events bounded by plate tracks (ADR-0006).

    Public seam: feed sightings with :meth:`record`, close passes with :meth:`end_pass`
    and stream death with :meth:`flush`; the observable effect is the rows appended
    to the Events log at event **close** (timestamp = first sighting, confidence =
    winning sighting).
    """

    def __init__(self, log: EventLog, iou_thresh: float = 0.3, max_misses: int = 6, dedup_window_s: float = 10.0, include_legacy: bool = False):
        self.log = log
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.dedup_window_s = dedup_window_s
        self.include_legacy = include_legacy  # format set knob (ADR-0008): what counts as validated
        self.tracks: list[_Track] = []
        self.pass_index = 0
        self.closed_strings: dict[str, float] = {}

    def record(self, now: float, box: tuple[float, float, float, float], raw: str, confidence: float) -> bool:
        """Attach a sighting to the best-overlapping open track, else open a new one."""
        sighting = _Sighting(normalize(raw), confidence)
        best: _Track | None = None
        best_iou = 0.0
        for track in self.tracks:
            # No per-pass claim guard: the detector double-boxes one plate sometimes,
            # and overlapping boxes in the same pass are the same car; distinct
            # cars never reach ``iou_thresh`` even dilated (unit-tested).
            overlap = self._iou(track.box, box)
            if overlap >= self.iou_thresh and overlap > best_iou:
                best, best_iou = track, overlap
        if best is None:
            opened = _Track(box, now, sighting)
            opened.last_pass = self.pass_index  # a fresh sighting counts as this pass's match
            self.tracks.append(opened)
            return True
        best.box = box
        best.last_pass = self.pass_index
        best.sightings.append(sighting)
        return False

    def end_pass(self, now: float) -> list[tuple[str, str, str, str, float]]:
        """Close every track whose miss count passed the tolerance; one row each."""
        overdue = [t for t in self.tracks if self.pass_index - t.last_pass > self.max_misses]
        self.tracks = [t for t in self.tracks if self.pass_index - t.last_pass <= self.max_misses]
        self.pass_index += 1
        return self._close(overdue, now)

    def flush(self, now: float) -> list[tuple[str, str, str, str, float]]:
        """Stream death: close all open tracks immediately (bounded work, no-op after)."""
        open_tracks, self.tracks = self.tracks, []
        return self._close(open_tracks, now)

    def _close(self, tracks: list[_Track], now: float) -> list[tuple[str, str, str, str, float]]:
        """Consensus per closing track; the exact-string guard suppresses re-emission."""
        rows: list[tuple[str, str, str, str, float]] = []
        for track in tracks:
            text, confidence = self._consensus(track)
            self._prune_closed_strings(now)
            if text in self.closed_strings:
                continue  # guard: same exact string already closed inside the window
            self.closed_strings[text] = now
            format_match = classify(text, include_legacy=self.include_legacy)
            row = (self._iso(track.first_now), text, class_for(format_match or ""), status_for(format_match or ""), confidence)
            self.log.append(*row)
            rows.append(row)
        return rows

    def _consensus(self, track: _Track) -> tuple[str, float]:
        """Plurality reading across the track's sightings; validated preferred, ties by confidence."""
        votes: dict[str, list[_Sighting]] = {}
        for sighting in track.sightings:
            votes.setdefault(sighting.text, []).append(sighting)
        candidates = [t for t in votes if classify(t, include_legacy=self.include_legacy)] or list(votes)
        winner = max(candidates, key=lambda t: (len(votes[t]), max(s.confidence for s in votes[t])))
        return winner, max(s.confidence for s in votes[winner])

    def _prune_closed_strings(self, now: float) -> None:
        """Forget guarded strings whose window has expired."""
        for text in [t for t, seen in self.closed_strings.items() if now - seen > self.dedup_window_s]:
            del self.closed_strings[text]

    def _iou(self, a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        """IoU of the boxes dilated by ``_IOU_DILATE_FRAC`` of their size (see finding below)."""
        ax1, ay1, ax2, ay2 = self._dilate(a)
        bx1, by1, bx2, by2 = self._dilate(b)
        iw = min(ax2, bx2) - max(ax1, bx1)
        ih = min(ay2, by2) - max(ay1, by1)
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def _dilate(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Grow a box by ``_IOU_DILATE_FRAC`` of its own size on every side."""
        x1, y1, x2, y2 = box
        dx = (x2 - x1) * _IOU_DILATE_FRAC
        dy = (y2 - y1) * _IOU_DILATE_FRAC
        return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)

    def _iso(self, now: float) -> str:
        """First-sighting timestamp rendered like the pipeline's frames."""
        return datetime.fromtimestamp(now, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
