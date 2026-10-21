# Argentine Mercosur License Plate Detector (ALPR)

Automatic license plate recognition for **Argentine Mercosur plates** from a single
RTSP camera on a **CPU-only Linux server**, 100% free & open source — no paid APIs,
no cloud, no PyTorch, no Ultralytics runtime.

Pipeline: **detect** (ONNX plate detector) → **OCR** (ONNX whole-plate reader) →
**validate** against the two Argentine formats → **dedup events** appended to a CSV
and shown on a live **MJPEG web dashboard**.

| Plate | Format | Example | Regex |
|---|---|---|---|
| Car | `AA 123 BB` | `AB 123 CD` | `^[A-Z]{2}[0-9]{3}[A-Z]{2}$` |
| Motorcycle | `A 123 BCD` | `A 123 BCD` | `^[A-Z][0-9]{3}[A-Z]{3}$` |

Alphabet is A–Z (no Ñ). Readings that match neither format are still logged —
as `UNVALIDATED` rows — so they can be reviewed later.

## Architecture

```
RTSP camera ──ffmpeg(imageio-ffmpeg)──► capture thread ──► bounded frame buffer
                                                              │ (drop-oldest, ~10 frames)
                                        pipeline thread ◄─────┘
                     detection+OCR every 1/DETECT_FPS (default 3 FPS) │ render every frame
                                        │                                     ▼
                              event tracker (dedup 10s)          JPEG buffer ──► /stream (MJPEG @960×540)
                                        ▼
                              events.csv (append-only) ──► GET /plates.csv + dashboard
```

- Detection/OCR run at `DETECT_FPS` (3); the dashboard stream renders at the full source
  rate reusing the overlays from the last detection pass — CPU budget is tunable without
  dropping stream smoothness.
- If the RTSP stream dies, ffmpeg exits, the capture thread relaunches it with
  exponential backoff (capped at 15s) and the stream shows a `STREAM OFFLINE` banner.
  The server never crashes.
- Full design rationale lives in `docs/adr/` (see ADR-0001…0004).

## Models & provenance

Weights are **not committed**; `scripts/fetch_models.py` downloads them (with pinned
SHA-256 checksums) into the gitignored `models/` dir:

| Model | Role | Size | Source (MIT repos) |
|---|---|---|---|
| `yolo-v9-t-384-license-plates-end2end.onnx` | detector (384×384, NMS in-graph) | 7.7 MB | [ankandrew/open-image-models](https://github.com/ankandrew/open-image-models) `assets` release |
| `cct_s_v2_global.onnx` | OCR — default | 5.0 MB | [ankandrew/fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) `arg-plates` release |
| `arg_cnn_ocr.onnx` | OCR — fallback (Argentina-only, fixed 7 slots) | 7.6 MB | same release |

**Verified benchmark** (upstream, Argentina plate set incl. Mercosur car +
motorcycle): CCT-S v2 global scores **97.19% plate / 99.51% char** accuracy — the OCR
is genuinely trained on the newer Argentine Mercosur plates.

**License caveat (ADR-0004):** both upstream repos declare MIT, but the detector's
training lineage (YOLOv9 reference trainer) sits in a gray zone. This repo ships no
weights, and using the detector weights the way this project does (inference, MIT repo
declaration) is the pragmatic choice; if you need a fully clean lineage, fine-tune an
Apache YOLOX-nano / RF-DETR-nano on the MIT `arg-plates` dataset + the MIT
`vinihcampos/plates-generator` synthetic generator — the pipeline accepts any ONNX
detector through the same `PlateDetector` contract (documented in `alpr/detector.py`).

## Setup (bare venv, no Docker)

Target: Python 3.12+ on Linux.

```bash
python -m venv .venv && . .venv/activate
pip install onnxruntime numpy opencv-python-headless fastapi uvicorn python-dotenv imageio-ffmpeg
python scripts/fetch_models.py          # detector + default OCR; add --all for the fallback too
cp .env.example .env                    # fill in RTSP endpoint + credentials
```

## Run

```bash
python -m alpr                  # live: capture thread + pipeline + dashboard on HTTP_PORT
python -m alpr --source clip.mp4    # debug: identical pipeline over a local .mp4/.jpg
```

Open the dashboard at `http://<server>:8080/` — MJPEG stream with bounding boxes,
decoded plate, vehicle class, confidence, plus the tail of the events log.
`GET /plates.csv` serves the raw CSV for download.

All tuning knobs are in `.env` (see `.env.example`): endpoint, credentials, ROI
(normalized `left,top,right,bottom`), `DETECT_FPS`, `DEDUP_WINDOW_S`, model choice
(`OCR_MODEL=cct_s_v2_global|arg_cnn_ocr`), render size, HTTP port.

## systemd (example unit)

```ini
# /etc/systemd/system/alpr.service
[Unit]
Description=Argentine Mercosur ALPR

[Service]
WorkingDirectory=/opt/alpr
ExecStart=/opt/alpr/.venv/bin/python -m alpr
Restart=always
```

*(Copy the repo to `/opt/alpr`, activate the venv's packages system-wide or point
`ExecStart` at the venv python.)*

## Development

```bash
python -m unittest discover -s tests   # validation / events / config unit tests
```

## Repo layout

```
alpr/               package: config, ingest, detector, ocr, validation, events,
                    pipeline, server (+ dashboard.html)
scripts/fetch_models.py                pinned-URL ONNX downloader with checksum verify
models/                                gitignored ONNX weights
docs/adr/                              decision records (model stack, runtime, CSV log, detector caveat)
docs/agents/, GLOSSARY.md              agent conventions + canonical vocabulary
tests/                                 stdlib unittest suite
```

## Known limitations

- The **detector** is generic license-plate trained (all countries) — its bounding boxes
  are country-agnostic; Argentine-specificity comes from the **OCR** (CCT-S v2 global,
  trained on the AR `arg-plates` set) plus the Mercosur regex validation, and false
  detections from non-AR plates get filtered out as UNVALIDATED at worst. See ADR-0004.
- A hung (but not dead) ffmpeg can stall the capture read; the reconnect loop covers
  crashes/exits, not wedged sockets.
- No auth on the dashboard by design — LAN only (Q17).

MIT-style pipeline code; consult `docs/adr/0001` and `0004` for model licensing.
