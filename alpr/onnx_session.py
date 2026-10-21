"""ONNX Runtime session factory (ADR-0002): try OpenVINO first, CPU baseline always."""

from __future__ import annotations

import logging
from pathlib import Path

import onnxruntime as ort

LOGGER = logging.getLogger(__name__)


def create_session(model_path: Path, name: str) -> ort.InferenceSession:
    """Load an ONNX model, preferring OpenVINOExecutionProvider when installed.

    Falls back to a pure ``CPUExecutionProvider`` session on any load failure, so a
    CPU-only Linux server always works (docs/adr/0002-onnx-runtime-cpu-baseline.md).
    """
    available = ort.get_available_providers()
    if "OpenVINOExecutionProvider" in available:
        providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as exc:  # provider rejected this particular graph
        LOGGER.warning("%s: %s; retrying with CPUExecutionProvider only", name, exc)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    LOGGER.info("%s using providers: %s", name, session.get_providers())
    return session
