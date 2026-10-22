# Argentine License Plate Detector (ALPR)

Automatic license plate recognition for **Argentine plates** from a single
RTSP camera on a **CPU-only Windows server** (Python 3.12.7), 100% free & open source —
no paid APIs, no cloud, no PyTorch, no Ultralytics runtime.

Pipeline: **detect** (ONNX plate detector) → **OCR** (ONNX whole-plate reader) →
**validate** against the configured **format set** (`PLATE_FORMATS`, ADR-0008) → **one event row per validated car pass**
(appended to a CSV; event identity is the plate track, not the plate string)
and shown on a live **MJPEG web dashboard**.

| Plate | Format | Example | Regex |
|---|---|---|---|
| Car (Mercosur, current) | `AB 123 CD` | `AB 123 CD` | `^[A-Z]{2}[0-9]{3}[A-Z]{2}$` |
| Legacy car (1995–2016) | `ABC 123` | `ABC 123` | `^[A-Z]{3}[0-9]{3}$` — only with `PLATE_FORMATS=MERCOSUR_PLUS_LEGACY` |

Alphabet is A–Z (no Ñ). Motorcycles are **out of the domain in either era** (ADR-0007).
Readings that match no format in the configured format set produce **no entry at all**:
no CSV row, no dashboard row (ADR-0009) — they stay visible only as `?` boxes on the
live stream. With legacy mode on, neighbouring countries sharing the 3+3 shape
(Paraguay, Peru) validate as `VALIDATED_LEGACY` noise — the status column keeps
them filterable (ADR-0008).

## Architecture

```
RTSP camera ──ffmpeg(imageio-ffmpeg)──► capture thread ──► bounded frame buffer
                                                              │ (drop-oldest, ~10 frames)
                                        pipeline thread ◄─────┘
                     detection+OCR every 1/DETECT_FPS (default 3 FPS) │ render every frame
                                        │                                     ▼
                        plate tracker (one row per validated track)  JPEG buffer ──► /stream (MJPEG @960×540)
                                        ▼
                              events.csv (append-only) ──► GET /plates.csv + dashboard
```

- Detection/OCR run at `DETECT_FPS` (3); the dashboard stream renders at the full source
  rate reusing the overlays from the last detection pass — CPU budget is tunable without
  dropping stream smoothness.
- If the RTSP stream dies, ffmpeg exits, the capture thread relaunches it with
  exponential backoff (capped at 15s) and the stream shows a `STREAM OFFLINE` banner.
  The server never crashes.
- Full design rationale lives in `docs/adr/` (see ADR-0001…0009).

## Models & provenance

Weights are **not committed**; `scripts/fetch_models.py` downloads them (with pinned
SHA-256 checksums) into the gitignored `models/` dir:

| Model | Role | Size | Source (MIT repos) |
|---|---|---|---|
| `yolo-v9-s-608-license-plates-end2end.onnx` | detector — **default** (608×608, NMS in-graph) | 28.6 MB | [ankandrew/open-image-models](https://github.com/ankandrew/open-image-models) `assets` release |
| `yolo-v9-t-{640,512,416,384,256}-license-plates-end2end.onnx` | detector tiers — swap via `DETECTOR_MODEL` (ADR-0010) | 7.8 MB each | same release |
| `cct_s_v2_global.onnx` | OCR — default | 5.3 MB | [ankandrew/fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) `arg-plates` release |
| `cct_{xs_v2,s_v1,xs_v1}_global.onnx` | OCR siblings — swap via `OCR_MODEL` (ADR-0010) | 2.1–7.6 MB | same release |
| `arg_cnn_ocr.onnx` | OCR — fallback (Argentina-only, fixed 7 slots — keep with `MERCOSUR_ONLY`) | 8.0 MB | same release |

**Verified benchmark** (upstream, Argentina plate set incl. Mercosur car +
motorcycle): CCT-S v2 global scores **97.19% plate / 99.51% char** accuracy — the OCR
is genuinely trained on the newer Argentine Mercosur plates. Its three zoo siblings all
measure lower (97.5/97.4/96.0% plate acc), so the default stands.

**Detector tiers (ADR-0010):** the accuracy upgrade path under the CPU-only / no-fine-tuning
constraints is the input-size tier, not the architecture. `s-608` (mAP50-95 **0.772** vs
0.687 for the old `t-384`) is the default: ~101 ms/pass measured on the dev CPU — still ~10
inferences/s against the 3 FPS budget — and on an identical 100 s `sample3.mp4` Test-rig
window it logged 16 events / **15 distinct plate strings** vs `t-384`'s 16 / 10 (full model-vs-model
tables: [BENCHMARK.md](BENCHMARK.md)). Every tier stays `.env`-switchable with its accuracy/size/
latency annotated in `.env.example`; step down to `t-640` if the real server CPU starves.

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

Target: **Windows**, Python 3.12.x (the production server runs 3.12.7). All commands are
cmd.exe, run from the repo root; see ADR-0005 for the Windows-only / on-demand decision.

```bat
python -m venv .venv
call .venv\Scripts\activate
pip install onnxruntime numpy opencv-python-headless fastapi uvicorn python-dotenv imageio-ffmpeg
python scripts\fetch_models.py          :: detector + default OCR; add --all for the fallback too
copy .env.example .env                  :: fill in RTSP endpoint + credentials
```

## Run

```bat
python -m alpr                  :: live: capture thread + pipeline + dashboard on HTTP_PORT
python -m alpr --source clip.mp4 :: debug: identical pipeline over a local .mp4/.jpg
```

Open the dashboard at `http://<server>:8080/` — MJPEG stream with bounding boxes,
decoded plate, status, confidence, plus the tail of the events log.
`GET /plates.csv` serves the raw CSV for download.

All tuning knobs are in `.env` (see `.env.example`): endpoint, credentials, ROI
(normalized `left,top,right,bottom`), `DETECT_FPS`, `TRACK_IOU_THRESH` and
`TRACK_MAX_MISSES` (plate-track association), `DEDUP_WINDOW_S` (exact-string
guard only), `PLATE_FORMATS` (format set: `MERCOSUR_ONLY|MERCOSUR_PLUS_LEGACY`, ADR-0008),
model choice (`DETECTOR_MODEL` — any zoo tier, ADR-0010; `OCR_MODEL=cct_s_v2_global|cct_xs_v2_global|cct_s_v1_global|cct_xs_v1_global|arg_cnn_ocr`), render size, HTTP port.

## Deploying to another machine (step by step)

Nothing is copied by hand except the ONNX weights, which the fetch script re-downloads
pinned by checksum. Weights, `.env`, `events.csv` and sample footage never leave a
machine through git (see `.gitignore`).

```bat
git clone https://github.com/jpalvadev/licence-plate-detector.git
cd licence-plate-detector
python -m venv .venv
call .venv\Scripts\activate
pip install onnxruntime numpy opencv-python-headless fastapi uvicorn python-dotenv imageio-ffmpeg
python scripts\fetch_models.py          :: verify checksums, then copy your .env from the old machine
python -m alpr                          :: open the dashboard at http://<machine>:8080/
```

Started on demand — no service manager (ADR-0005): relaunch it manually after a reboot;
the Events log survives because the CSV is append-only.

## Test rig: simulated camera with mediamtx

Validates the whole RTSP ingest path on one machine — the app reads `sample.mp4`
looped through mediamtx exactly as it reads the real camera.

```bat
:: window A: mediamtx (stock config: RTSP server on port 8554, no edits needed)
cd C:\path\to\mediamtx && .\mediamtx.exe

:: window B: loop the sample into mediamtx (drop a sample.mp4 at the repo root first;
:: it is gitignored — real plates must never enter the repo)
ffmpeg -re -stream_loop -1 -i sample.mp4 -c copy -f rtsp rtsp://127.0.0.1:8554/mystream

:: window C: point the app at the rig (process env beats .env, no file edit needed)
set RTSP_ENDPOINT=127.0.0.1:8554/mystream
python -m alpr
```

Expected: detections + events rows on the sample's plates; killing window B mid-run must
trigger the `STREAM OFFLINE` banner and the reconnect/backoff ladder, not a crash.

## Development

```bash
python -m unittest discover -s tests   # validation / events / config / tracker unit tests
```

## Repo layout

```
alpr/               package: config, ingest, detector, ocr, validation, events,
                    pipeline, server (+ dashboard.html)
scripts/fetch_models.py                pinned-URL ONNX downloader with checksum verify
BENCHMARK.md                           measured model-vs-model tables (Test rig on sample3.mp4)
models/                                gitignored ONNX weights
docs/adr/                              decision records (model stack, runtime, CSV log, detector caveat, Windows/on-demand run, plate domain + format set, validated-only event log, PoC detector-tier upgrade)
docs/agents/, GLOSSARY.md              agent conventions + canonical vocabulary
tests/                                 stdlib unittest suite
```

## Known limitations

- The **detector** is generic license-plate trained (all countries) — its bounding boxes
  are country-agnostic; Argentine-specificity comes from the **OCR** (CCT-S v2 global,
  trained on the AR `arg-plates` set) plus the regex validation of the configured format
  set, and false detections from non-AR plates get filtered out: they match no format, so
  no row is logged (ADR-0009; in legacy mode, neighbouring 3+3 shapes — Paraguay, Peru —
  validate instead; filter by status). See ADR-0004 and ADR-0008.
- A hung (but not dead) ffmpeg can stall the capture read; the reconnect loop covers
  crashes/exits, not wedged sockets.
- No auth on the dashboard by design — LAN only (Q17).
- Started on demand (Q8 / ADR-0005): closing the terminal stops the app; relaunch by hand —
  supervision (NSSM, Task Scheduler) can be added later without touching the app.

Code is MIT-licensed (`LICENSE`); consult `docs/adr/0001` and `0004` for model licensing.
