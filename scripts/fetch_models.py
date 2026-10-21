"""Fetch the vendored ONNX weights into models/ (gitignored) with integrity checks.

Model provenance is documented in the README + docs/adr/0004 (license caveat included).
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
    "yolo-v9-t-384-license-plates-end2end": {
        "url": "https://github.com/ankandrew/open-image-models/releases/download/assets/"
        "yolo-v9-t-384-license-plates-end2end.onnx",
        "size": 7_771_218,
        "sha256": "888397b96d761c89db40bc9c305838e8652660f5e282c2cadebbe8d2951a77a8",
        "note": "MIT repo; YOLOv9 trainer lineage caveat per ADR-0004",
    },
    "cct_s_v2_global": {
        "url": "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/"
        "cct_s_v2_global.onnx",
        "size": 5_262_230,
        "sha256": "384bbbd2cea3ef54761d3df70822ef3a349ee1a112aeafddbe0e3ba06bc6e47b",
        "note": "CCT-S v2 global: 97.2% plate / 99.5% char acc on Argentine benchmark",
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
    parser.add_argument("--all", action="store_true", help="include the arg_cnn_ocr fallback too")
    parser.add_argument("--force", action="store_true", help="re-download even when a file exists")
    args = parser.parse_args(argv)

    names = args.names or ["yolo-v9-t-384-license-plates-end2end", "cct_s_v2_global"]
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
