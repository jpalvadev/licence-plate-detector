# Deploying on the work server — step by step (Windows, cmd.exe)

Printable checklist. Run every command from the repo root in **cmd.exe**.
Cross-references: README.md ("Deploying to another machine", "Test rig"), ADR-0005
(started on demand — no service manager), ADR-0010 (detector tiers), BENCHMARK.md.

What you are standing up: detect → OCR → validate → `events.csv` + one plate
JPG per event in `media/` (ADR-0011) + MJPEG dashboard on LAN HTTP, CPU-only,
100 % local. Nothing phones home except the one-off pip + model downloads.

---

## 0. Prerequisites (check on the server first)

| Need | Why | Check with |
|---|---|---|
| Python ≥ 3.12 | repo targets 3.12.x (dev used 3.12.7) | `python --version` |
| Internet at deploy time | PyPI (~pip install) + GitHub (~34 MB weights) — one-off only; the app itself never needs the net afterwards | try the pip step; if it fails, pre-download at home (see §6 fallback) |
| GitHub account | repo is private | step 1 |
| The RTSP camera reachable **from the server's network** | the server, not your laptop, reads the stream | camera's `host:port/path` |
| ~150 MB free disk | code + venv + weights + runtime artifacts | — |

## 1. Get the code

Option A — git clone (private repo: needs GitHub login):

```bat
git clone https://github.com/jpalvadev/licence-plate-detector.git
cd licence-plate-detector
```

Option B — USB copy. Copy the folder WITHOUT: `.venv`, `models`, `media`,
`events.csv`, `*.mp4`, `__pycache__`, `prompt.txt`, `sample*` (all runtime
or private). Do copy `.env.example`, and optionally the old `.env` (it holds
the camera credentials — carried by hand, never by git).

## 2. Python check

```bat
python --version
```

If < 3.12, install from python.org first, then continue.

## 3. Virtual env

```bat
python -m venv .venv
call .venv\Scripts\activate
```

## 4. Dependencies (pinned versions in pyproject.toml)

```bat
pip install onnxruntime numpy opencv-python-headless fastapi uvicorn python-dotenv imageio-ffmpeg
```

## 5. ONNX weights (never in git — re-fetched pinned by SHA-256)

```bat
python scripts\fetch_models.py
```

Fetches the two defaults: detector `yolo-v9-s-608-license-plates-end2end`
(28.6 MB) + OCR `cct_s_v2_global` (5.3 MB), verifies checksums, prints
`OK ... sha256 verified`. Add `--all` for every tier/sibling (~90 MB), or
`--names cct_xs_v2_global` for one.

## 6. If the server has no internet: carry artifacts by hand

- `models/` (just the two `.onnx` files) copied onto the server; the fetch
  script verifies checksums against them, so corruption is caught.
- pip wheels: run the app on the dev machine to confirm, or copy a working
  `.venv` from an identical Windows version (usually fine, not guaranteed).

## 7. Configure `.env` (the only file you must edit)

```bat
copy .env.example .env
notepad .env
```

| Key | Set to | Notes |
|---|---|---|
| `RTSP_ENDPOINT` | `host:port/path` of the real camera | NOT 127.0.0.1 — must be reachable from the server |
| `RTSP_USER` / `RTSP_PASS` | camera credentials | special characters are percent-encoded for you (config.py) |
| `VIDEO_W` / `VIDEO_H` | real camera resolution | ffmpeg resizes if the camera differs |
| `ROI` | e.g. `0.15,0.35,0.85,0.75` | normalized `left,top,right,bottom`; empty = full frame; cropping to the gate boosts speed + kills off-lane detections |
| `DETECT_FPS` | 3 default | CPU budget knob (ADR-0010); lower if the server is weak |
| `DEDUP_WINDOW_S` | 10 default | one event per plate string per window |
| `EVENTS_CSV` / `MEDIA_DIR` | defaults fine | paths are relative to the repo root — always launch from there |
| `PLATE_FORMATS` | `MERCOSUR_ONLY` or `MERCOSUR_PLUS_LEGACY` | operator choice (ADR-0008); legacy ON brings Paraguay/Peru 3+3 as `PATENTE_ALFANUM` noise |
| `HTTP_HOST` / `HTTP_PORT` | `0.0.0.0` / pick a free port | 8090 here because 8080 is taken locally; no auth — LAN only (Q17) |
| `DETECTOR_MODEL` / `OCR_MODEL` | defaults = best accuracy | step down only if the server CPU starves (§9) |

## 8. Sanity checks before trusting the camera

```bat
python -m unittest discover -s tests        :: must print "Ran 58 tests ... OK"
```

Optional full-RTSP rehearsal (README §Test rig): run `mediamtx.exe` + feed a
sample via ffmpeg on the server, point `.env` at `127.0.0.1:8554/mystream`,
confirm detections/rows, then switch `RTSP_ENDPOINT` to the real camera.

## 9. Launch

```bat
python -m alpr
:: or double-click start.bat (same command), but keep the terminal open:
.venv\Scripts\python.exe -m alpr
```

Watch for: live boxes on the dashboard, rows appearing in `events.csv`,
one JPG per logged event under `media/` named `{plate}_{conf}_{timestamp}.jpg`.

Then, from any LAN machine: open `http://<server>:<HTTP_PORT>/`.

If the server CPU can't sustain `s-608` at 3 FPS:
`python scripts\fetch_models.py --names yolo-v9-t-640-license-plates-end2end`
then `DETECTOR_MODEL=yolo-v9-t-640-license-plates-end2end` (58 ms/pass,
mAP50-95 0.758 — tables in BENCHMARK.md).

## 10. Keeping it alive (ADR-0005: started on demand)

- Closing the terminal/SSH window **stops the app** — relaunch it the same way.
- After a reboot: re-run §9; `events.csv` + `media/` survive (append-only,
  gitignored runtime artifacts).
- Optional later: Task Scheduler or NSSM to wrap the launch line; the app
  needs no change for either.

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `STREAM OFFLINE` banner then reconnects | normal: ffmpeg exited, capture relaunches with backoff (capped 15 s) — not a crash |
| banner never recovers | hung-but-alive ffmpeg stalls the read (known limitation) → kill + relaunch the process |
| boxes never fire | camera res mismatch (`VIDEO_W/H`), `ROI` misplaced, or `DETECT_CONF_THRESH` vs angle/lighting |
| rows exist, dashboard empty | browser at another port than `HTTP_PORT` |
| wrong/other machine owns the port | pick a free `HTTP_PORT` |
| JPEG names garbled/colliding | none: names carry plate + raw conf + timestamp, 1:1 with CSV rows |
| checkpoint drift after a model swap | conf is model-scale in CSV/photos; display % re-anchors in `alpr/clarity.py` (expected) |

## 12. Never commit

`.env` (credentials), `events.csv`, `media/`, `models/`, `*.mp4` are already
`.gitignore`d — real plates never enter the repo; that is the PoC policy.
