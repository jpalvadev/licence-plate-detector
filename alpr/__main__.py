"""CLI entry point: ``python -m alpr`` (live) or ``python -m alpr --source FILE`` (debug).

Debug mode (Q16) runs the identical pipeline over a local ``.mp4``/``.jpg`` through the
same ffmpeg rawvideo pipe; annotated frames are saved to ``debug_out/`` and events print
to stdout.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .config import load_settings
from .ingest import RtspCapture, RtspReader
from .pipeline import Pipeline
from .server import serve


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="alpr", description="Argentine Mercosur ALPR (RTSP + dashboard)")
    parser.add_argument("--env", default=".env", type=Path, help="dotenv file (default ./.env)")
    parser.add_argument("--source", type=Path, help="debug mode: local .mp4/.jpg instead of RTSP")
    parser.add_argument("--debug-out", default="debug_out", type=Path, help="debug annotated frames dir")
    parser.add_argument("--host", help="override HTTP_HOST")
    parser.add_argument("--port", type=int, help="override HTTP_PORT")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    debug = args.source is not None
    try:
        settings = load_settings(args.env, require_endpoint=not debug)
    except ValueError as exc:
        print(f"Config error: {exc}")
        return 2
    if args.host:
        settings = replace(settings, http_host=args.host)
    if args.port:
        settings = replace(settings, http_port=args.port)

    missing = [
        settings.models_dir / f"{name}.onnx"
        for name in (settings.detector_model, settings.ocr_model)
        if not (settings.models_dir / f"{name}.onnx").exists()
    ]
    if missing:
        print("Missing model files: " + ", ".join(str(p) for p in missing))
        print("Fetch them first with:  python scripts/fetch_models.py")
        return 2

    if debug:
        return _run_debug(settings, args.source, args.debug_out)
    serve(settings)
    return 0


def _run_debug(settings, source: Path, out_dir: Path) -> int:
    """Single-process run of the identical pipeline over a local file."""
    pipeline = Pipeline(settings)
    reader = RtspReader()
    cmd = reader.file_command(str(source))
    if source.suffix.lower() == ".jpg" or source.suffix.lower() == ".jpeg":
        # Single image: ffmpeg rawvideo needs a frame count; use the image once.
        cmd = [cmd[0], "-loglevel", "error", "-i", str(source), "-frames:v", "1", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)  # noqa: S1731
    nbytes = settings.video_w * settings.video_h * 3
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    frame_idx = 0
    while True:
        buf = reader.read_exact(proc.stdout, nbytes)
        if len(buf) < nbytes:
            break
        frame = reader.frame_of(buf, settings.video_w, settings.video_h)
        pipeline.process_frame(frame, time.time())
        for line in pipeline.new_events():
            print("EVENT:", line)
        if written < 600:
            for jpg in pipeline.take_jpegs():
                Path(out_dir / f"frame_{frame_idx:06d}.jpg").write_bytes(jpg)
                written += 1
                break
        frame_idx += 1
    for line in pipeline.new_events():
        print("EVENT:", line)
    print(f"Wrote {written} annotated frame(s) to {out_dir}; events appended to {settings.events_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
