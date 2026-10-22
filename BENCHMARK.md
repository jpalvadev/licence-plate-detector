# Model benchmark — accuracy & speed on the Test rig (sample3.mp4)

Run date: 2026-10-21 · dev machine CPU only, ONNX `CPUExecutionProvider` (the production
server is a slower CPU — relative speedups carry, absolute ms do not).

## Method

- Rig: `mediamtx` (RTSP :8554) ← `ffmpeg -re -ss 52 -t 115` looping **the same 115 s window**
  of `sample3.mp4` (busiest window found by a full-clip scan with the fastest detector:
  76,737 frames decoded, 358 hit-frames, 87 sampled boxes at t = 52 s).
- Each model combo ran the **identical app** (`python -m alpr`) over that identical window
  for 100 s with per-run knobs: `DETECT_FPS=3`, `DET_CONF_THRESH=0.4`,
  `PLATE_FORMATS=MERCOSUR_PLUS_LEGACY`, fresh `EVENTS_CSV` per run.
- Accuracy metric = validated **Recognition events** logged in 100 s (rows + distinct
  plate strings + mean confidence). Rows are an *imperfect* proxy: one real plate can log
  several rows (split tracks + OCR char instability per ADR-0006, softmax conf ≈ 0.15), and
  legacy-mode rows can be neighbouring-country noise. The upstream zoo mAP and the AR
  benchmark are the accuracy ground; rig rows are the tie-breaker.
- Latency: standalone `PlateDetector`/`PlateOcr` calls measured on this CPU (ms), plus the
  333 ms/pass budget implied by `DETECT_FPS=3`.

## Detector tiers (OCR fixed: `cct_s_v2_global`)

| Model | Events (CAR/LEGACY) | Distinct strings | Mean conf | CPU ms/pass | Upstream P / R / mAP50 / mAP50-95 | Size |
|---|---|---|---|---|---|---|
| **`yolo-v9-s-608-license-plates-end2end`** | **16 (8/8)** | **15** | 0.155 | 101 | 0.957 / 0.917 / 0.966 / **0.772** | 28.6 MB |
| `yolo-v9-t-640-license-plates-end2end` | 10 (5/5) | 9 | 0.145 | 58 | 0.966 / 0.896 / 0.958 / 0.758 | 7.8 MB |
| `yolo-v9-t-512-license-plates-end2end` | 9 (5/4) | 8 | 0.151 | 36 | 0.955 / 0.901 / 0.948 / 0.724 | 7.8 MB |
| `yolo-v9-t-416-license-plates-end2end` | 12 (8/4) | 11 | 0.149 | 25 | 0.940 / 0.894 / 0.940 / 0.702 | 7.8 MB |
| `yolo-v9-t-384-license-plates-end2end` *(old default)* | 16 (7/9) | 10 | 0.149 | 19 | 0.942 / 0.863 / 0.920 / 0.687 | 7.8 MB |
| `yolo-v9-t-256-license-plates-end2end` | 4 (3/1) | 3 | 0.169 | 11 | 0.937 / 0.797 / 0.858 / 0.606 | 7.8 MB |

Reading: `s-608` logs as many rows as `t-384` (16) but **50 % more distinct plate strings**
(15 vs 10) — it recovers real plates the smaller tiers miss outright (its extra strings have
near-mates in other runs' sets: `NWZ451`/`NWZ465`, `QMZ461`/`NMZ461`…), while 0.772 mAP50-95
is the honest upstream accuracy statement. `t-416`/`t-640` sit in the middle at ~4× less CPU.
`t-256` is unusable here (4 rows). Rows where a smaller tier matches a `s-608` string (e.g.
`AF453GF`, `AG539UE`, `NMA273`, `NMZ461` appear in every run) are the consensus floor.

## OCR models (detector fixed: `yolo-v9-s-608…`)

| Model | Events (CAR/LEGACY) | Distinct strings | Mean conf | CPU ms/crop | AR plate acc (upstream val mean) | Size |
|---|---|---|---|---|---|---|
| **`cct_s_v2_global`** | **16 (8/8)** | **15** | **0.155** | 16.2 | **97.9 %** (benchmark 97.2 / 99.5 char) | 5.3 MB |
| `cct_xs_v2_global` | 9 (5/4) | 8 | 0.156 | 2.2 | 97.4 % | 3.3 MB |
| `cct_s_v1_global` | 16 (9/7) | 15 | 0.144 | 23.4 | 97.5 % | 7.6 MB |
| `cct_xs_v1_global` | 12 (5/7) | 11 | 0.130 | 1.6 | 96.0 % | 2.1 MB |
| `arg_cnn_ocr` | 18 (11/7) | 17 | 0.105 | 1.6 | — (fixed 7-slot, AR-only) | 8.0 MB |

Reading: `cct_s_v2_global` wins every comparable axis: same rows as `s_v1` with **higher
confidence and 3× less CPU** (the v1 row count is inflated: 9 CAR rows include
misvalidations like `AK453GF`/`KMZ461` — same plate, different string). `arg_cnn_ocr` logs
the most rows (18) at the **lowest mean conf (0.105)** — its surplus rows are misreads that
still match a regex; more rows ≠ more truth. `xs` variants lose whole events (9/12 rows):
on small/angled crops they under-detect, not just misread.

## Verdict (accuracy first)

- **Detector: `yolo-v9-s-608-license-plates-end2end`** — best upstream mAP (0.772), best rig
  distinct-strings (15); 101 ms/pass on dev CPU ≈ 10 inferences/s vs the 3 FPS floor (333 ms).
  Step down to `t-640` (58 ms, 0.758) if the server CPU starves — one `.env` line.
- **OCR: `cct_s_v2_global`** — best accuracy AND better conf/CPU than every sibling.
- Both defaults are the uncommented knobs in `.env` / `.env.example`; every other row above is
  a commented alternative there (fetched via `scripts/fetch_models.py`, checksum-pinned).
