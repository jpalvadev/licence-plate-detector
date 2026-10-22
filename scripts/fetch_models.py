"""Fetch the vendored ONNX weights into models/ (gitignored) with integrity checks.

Model provenance is documented in the README + docs/adr/0004 (license caveat included).
Detector tiers and OCR siblings are all fetchable; the .env knob picks the live one
(accuracy/size/latency trade-offs in .env.example and ADR-0010).
Checksums below were pinned at fetch time (2026-10) from the upstream GitHub releases;
re-run with --force to re-download, or bump the manifest if upstream re-exports a model.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import urllib.request

MANIFEST = {
    # --- Plate detectors: open-image-models `assets` release, all MIT-declared, NMS in-graph.
    # Accuracy figures are the upstream zoo eval (P/R/mAP50/mAP50-95); sizes and dev-CPU
    # latencies (19-101 ms/pass) documented per ADR-0010 — pick the tier in .env.
    "yolo-v9-s-608-license-plates-end2end": {
        "url": "https://github.com/ankandrew/open-image-models/releases/download/assets/"
        "yolo-v9-s-608-license-plates-end2end.onnx",
        "size": 28_612_350,
        "sha256": "2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a",
        "note": "best accuracy: P 0.957 R 0.917 mAP50-95 0.772; ~101 ms/pass dev CPU (~5x t-384)",
    },
    "yolo-v9-t-384-license-plates-end2end": {
        "url": "https://github.com/ankandrew/open-image-models/releases/download/assets/"
        "yolo-v9-t-384-license-plates-end2end.onnx",
        "size": 7_771_218,
        "sha256": "888397b96d761c89db40bc9c305838e8652660f5e282c2cadebbe8d2951a77a8",
        "note": "MIT repo; YOLOv9 trainer lineage caveat per ADR-0004; mAP50-95 0.687, ~19 ms dev CPU",
    },
    "yolo-v9-t-640-license-plates-end2end": {
        "url": "https://github.com/ankandrew/open-image-models/releases/download/assets/"
        "yolo-v9-t-640-license-plates-end2end.onnx",
        "size": 7_835_770,
        "sha256": "c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076",
        "note": "mAP50-95 0.758, ~58 ms dev CPU — cheaper accuracy step",
    },
    "yolo-v9-t-512-license-plates-end2end": {
        "url": "https://github.com/ankandrew/open-image-models/releases/download/assets/"
        "yolo-v9-t-512-license-plates-end2end.onnx",
        "size": 7_799_480,
        "sha256": "746fdd358ec110418775d7c9d8d07910d48b1a21471f92bf4421f6510d6daade",
        "note": "mAP50-95 0.724, ~36 ms dev CPU",
    },
    "yolo-v9-t-416-license-plates-end2end": {
        "url": "https://github.com/ankandrew/open-image-models/releases/download/assets/"
        "yolo-v9-t-416-license-plates-end2end.onnx",
        "size": 7_777_518,
        "sha256": "0469d81f176081671ca4caca1e05f94758eae2f9f659bb8a6c410216f2173fe4",
        "note": "mAP50-95 0.702, ~25 ms dev CPU",
    },
    "yolo-v9-t-256-license-plates-end2end": {
        "url": "https://github.com/ankandrew/open-image-models/releases/download/assets/"
        "yolo-v9-t-256-license-plates-end2end.onnx",
        "size": 7_751_055,
        "sha256": "938a34ce868da69432fd5a872f24337b86e3dc8753426e16ae18163f615c2a2c",
        "note": "mAP50-95 0.606, ~11 ms dev CPU — fastest, weakest recall",
    },
    # --- OCR: fast-plate-ocr `arg-plates` release. plate_acc = mean of upstream per-batch
    # val results on the AR set; cct_s_v2_global stays best, siblings are .env options.
    "cct_s_v2_global": {
        "url": "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/"
        "cct_s_v2_global.onnx",
        "size": 5_262_230,
        "sha256": "384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b",
        "note": "default: 97.9% plate acc (val mean; benchmark 97.2/99.5); ~16 ms/crop dev CPU",
    },
    "cct_xs_v2_global": {
        "url": "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/"
        "cct_xs_v2_global.onnx",
        "size": 3_344_292,
        "sha256": "8031afb5fdc6b4d80462c9d542f1284ebd2cfddf5dbacd62609848d7e2855f44",
        "note": "97.4% plate acc; ~2.2 ms/crop dev CPU — 7x faster, slightly weaker",
    },
    "cct_s_v1_global": {
        "url": "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/"
        "cct_s_v1_global.onnx",
        "size": 7_623_453,
        "sha256": "5c95b3231fff415b05cb48a3a39fab7c009364c2ee441bef092e0162ab75ea74",
        "note": "97.5% plate acc; ~23 ms/crop dev CPU — older v1, slower than v2-S",
    },
    "cct_xs_v1_global": {
        "url": "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/"
        "cct_xs_v1_global.onnx",
        "size": 2_113_942,
        "sha256": "c1ae2ae031b10f8ab7ed6e3cdaf8d9c753477e299dbad3579c8f705a7d882134",
        "note": "96.0% plate acc; ~1.6 ms/crop dev CPU — smallest/cheapest",
    },
    "arg_cnn_ocr": {
        "url": "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/"
        "arg_cnn_ocr.onnx",
        "size": 7_976_537,
        "sha256": "3b99b144273094f18f4f6a81a21832d334127378ed4247c7d99c0de2e21ff9c4",
        "note": "fallback: Argentina-only fixed 7-slot OCR",
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, dest_dir: Path, *, force: bool) -> None:
    entry = MANIFEST[name]
    dest = dest_dir / f"{name}.onnx"
    if dest.exists() and sha256_of(dest) == entry["sha256"]:
        print(f"OK    {name}: already present and checksum matches")
        return
    if dest.exists() and not force:
        msg = f"stale file with wrong checksum: {dest} (delete or pass --force)"
        raise SystemExit(msg)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"GET   {name} -> {dest}")
    opener = urllib.request.urlopen(entry["url"], timeout=60)  # noqa: URLENUM — public asset URL
    with dest.open("wb") as fh:
        while True:
            chunk = opener.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    digest = sha256_of(dest)
    if digest != entry["sha256"]:
        msg = f"checksum mismatch for {name}: got {digest}, want {entry['sha256']}"
        raise SystemExit(msg)
    if dest.stat().st_size != entry["size"]:
        msg = f"size mismatch for {name}: got {dest.stat().st_size}, want {entry['size']}"
        raise SystemExit(msg)
    print(f"OK    {name}: {entry['size']} bytes, sha256 verified")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Fetch vendored ONNX models")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--names", nargs="*", default=None, help="default: detector + default OCR")
    parser.add_argument("--all", action="store_true", help="fetch every model in the manifest")
    parser.add_argument("--force", action="store_true", help="re-download even when a file exists")
    args = parser.parse_args(argv)

    names = args.names or ["yolo-v9-s-608-license-plates-end2end", "cct_s_v2_global"]
    if args.all:
        names = list(MANIFEST)
    for name in names:
        if name not in MANIFEST:
            print(f"Unknown model {name!r}: pick from {sorted(MANIFEST)}")
            return 2
        fetch(name, args.models_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
