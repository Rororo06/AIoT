"""Download and extract the SisFall dataset.

The original institutional download link (sistemic.udea.edu.co) is dead, so this
script pulls the CC BY 4.0 Hugging Face mirror of the identical archive.

Run:  python tools/download_dataset.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "SisFall_dataset.zip"
EXTRACT_DIR = RAW_DIR / "extracted"
URL = (
    "https://huggingface.co/datasets/Algo-rythmic/Sisfall_Dataset/"
    "resolve/main/SisFall_dataset.zip"
)
EXPECTED_BYTES = 232_407_821
EXPECTED_SHA256 = "7f0dd1583e39fee0f402a6855976eb12517f7eaecdacce871d5b3421793ad665"


def report(done: int, block: int, total: int) -> None:
    if total > 0:
        pct = min(100, done * block * 100 // total)
        sys.stdout.write(f"\r  downloading… {pct:3d}%")
        sys.stdout.flush()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size == EXPECTED_BYTES:
        print(f"archive already present: {ZIP_PATH}")
    else:
        print(f"fetching {URL}")
        urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=report)
        print()

    size = ZIP_PATH.stat().st_size
    if size != EXPECTED_BYTES:
        sys.exit(f"unexpected archive size {size} (expected {EXPECTED_BYTES})")
    digest = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        sys.exit(f"sha256 mismatch\n  got      {digest}\n  expected {EXPECTED_SHA256}")
    print(f"sha256 ok: {digest}")

    target = EXTRACT_DIR / "SisFall_dataset"
    if target.exists():
        print(f"already extracted: {target}")
    else:
        print(f"extracting to {EXTRACT_DIR}")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(EXTRACT_DIR)

    subjects = sorted(p.name for p in target.iterdir() if p.is_dir())
    n_files = sum(1 for _ in target.rglob("*.txt"))
    print(f"{len(subjects)} subject folders, {n_files} trial files")


if __name__ == "__main__":
    main()
