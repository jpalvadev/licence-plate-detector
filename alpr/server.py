"""FastAPI dashboard: MJPEG stream (multipart), plain HTML page, raw events CSV.

Design (Q7/Q17/Q31): no auth, LAN-only. ``GET /`` serves the dashboard, ``GET /stream``
is the MJPEG multipart stream rendered at RENDER_WxRENDER_H with overlays from the
last detection pass, and ``GET /plates.csv`` serves the events log file as-is.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, Response, StreamingResponse

from .config import Settings
from .pipeline import Pipeline


def dashboard_html() -> str:
    """Dashboard page shipped inside the package (``alpr/dashboard.html``)."""
    return (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


async def mjpeg_chunks(pipeline: Pipeline):
    """Generator for multipart/x-mixed-replace JPEG frames."""
    boundary = b"--frame"
    while True:
        frames = pipeline.take_jpegs()
        if not frames:
            await asyncio.sleep(0.03)
            continue
        for jpg in frames:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
            yield b"\r\n\r\n" + jpg + b"\r\n"


def make_app(pipeline: Pipeline):
    """Build the FastAPI app bound to a started pipeline."""
    app = FastAPI(docs_url=None, redoc_url=None)

    @app.get("/", response_class="text/html")
    def index():
        return Response(dashboard_html(), media_type="text/html; charset=utf-8")

    @app.get("/stream")
    async def stream():
        return StreamingResponse(mjpeg_chunks(pipeline), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/plates.csv")
    def plates_csv():
        return pipeline.log.path.read_text(encoding="utf-8")

    return app


def serve(settings: Settings) -> None:
    """Start the pipeline consumer thread (it spawns the capture thread) then run uvicorn."""
    import threading
    import uvicorn

    pipeline = Pipeline(settings)
    threading.Thread(target=pipeline.run, name="pipeline", daemon=True).start()
    app = make_app(pipeline)
    uvicorn.run(app, host=settings.http_host, port=settings.http_port, log_level="warning")
