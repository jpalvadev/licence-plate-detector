"""RTSP ingestion: ffmpeg pipe (imageio-ffmpeg), capture thread, bounded drop-oldest buffer.

Design (Q5/Q11/Q12): ffmpeg subprocess pipes ``rawvideo bgr24`` frames to our capture
thread; frames land in a bounded buffer that **drops the oldest** frame when full
(zero-drift, no memory growth). When the stream dies the process exits at EOF; the
thread relaunches ffmpeg with exponential backoff (capped) and marks itself offline,
so the dashboard can show a STREAM OFFLINE banner — the server never crashes.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque

import imageio_ffmpeg
import numpy as np

LOGGER = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-unsafe core + lock wrapper: bounded frame handoff, drop-oldest."""

    def __init__(self, maxlen: int):
        self._frames: deque[tuple[np.ndarray, float]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, item: tuple[np.ndarray, float]) -> None:
        with self._lock:
            self._frames.append(item)

    def take(self) -> list[tuple[np.ndarray, float]]:
        """Drain everything currently buffered (oldest first)."""
        with self._lock:
            drained = list(self._frames)
            self._frames.clear()
            return drained

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


class RtspReader:
    """Low-level rawvideo reader over a ffmpeg subprocess. Shared by live and debug paths."""

    FFMPEG_ARGS_COMMON = ("-loglevel", "error", "-hide_banner")

    def ffmpeg_exe(self) -> str:
        # imageio-ffmpeg bundles its own static ffmpeg: no system dependency (Q28).
        return imageio_ffmpeg.get_ffmpeg_exe()

    def live_command(self, rtsp_url: str, w: int, h: int) -> list[str]:
        return [
            self.ffmpeg_exe(),
            *self.FFMPEG_ARGS_COMMON,
            "-i",
            rtsp_url,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-",
        ]

    def file_command(self, path: str, w: int, h: int) -> list[str]:
        """Same rawvideo pipe for local debug files (``--source file.mp4|jpg``, Q16)."""
        return [
            self.ffmpeg_exe(),
            *self.FFMPEG_ARGS_COMMON,
            "-i",
            path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-",
        ]

    def read_exact(self, stream, nbytes: int) -> bytes:
        """Read exactly nbytes or return a short read (EOF)."""
        chunks: list[bytes] = []
        remaining = nbytes
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def frame_of(self, buf: bytes, w: int, h: int) -> np.ndarray:
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)


class RtspCapture:
    """The dedicated capture thread + its bounded buffer and offline state."""

    def __init__(self, settings):
        self.settings = settings
        self.buffer = FrameBuffer(settings.queue_maxlen)
        self.offline = True  # nothing streamed yet
        self.offline_reason = "not started"
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_forever, name="rtsp-capture")
        self._thread.start()

    def _run_forever(self) -> None:
        attempt = 0
        while True:
            ok = self._stream_once()
            self.offline = True
            self.offline_reason = "stream ended" if ok else "ffmpeg launch failed"
            delay = min(2.0**attempt, 15.0)
            LOGGER.warning("RTSP stream down; reconnecting in %.0fs (attempt %d)", delay, attempt + 1)
            time.sleep(delay)
            # A reconnect that actually produced frames resets the backoff ladder.
            attempt = 0 if ok else attempt + 1

    def _stream_once(self) -> bool:
        """Run ffmpeg until EOF; feed frames to the buffer. True if any frame came through."""
        w, h = self.settings.video_w, self.settings.video_h
        nbytes = w * h * 3
        cmd = RtspReader.live_command(self.settings.rtsp_url, w, h)
        reader = RtspReader()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)  # noqa: S1731
        except OSError as exc:
            LOGGER.error("Could not launch ffmpeg: %s", exc)
            return False
        produced = False
        try:
            while True:
                buf = reader.read_exact(proc.stdout, nbytes)
                if len(buf) < nbytes:
                    break  # EOF: stream died
                self.buffer.append((reader.frame_of(buf, w, h), time.time()))
                produced = True
                self.offline = False
        finally:
            try:
                proc.stdout.close()
            except (AttributeError, BrokenPipeError, OSError):
                pass
            proc.wait()
        return produced
