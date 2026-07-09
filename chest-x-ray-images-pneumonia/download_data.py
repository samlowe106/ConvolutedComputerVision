"""Download and arrange the Kermany Chest X-Ray dataset -- no Kaggle account needed.

Source: Kermany et al., "Labeled OCT and Chest X-Ray Images for Classification",
Mendeley Data, DOI 10.17632/rscbjbr9sj.2 -- the file ``ChestXRay2017.zip`` (~1.24 GB).
This is the same data the Kaggle ``chest-xray-pneumonia`` mirror is derived from, but
fetched from the original public host so the project doesn't depend on Kaggle.

The script is idempotent (a ``data_dir/.prepared`` marker short-circuits a finished
setup) and does, skipping anything already done:

1. download the zip into ``data_dir/.cache/`` (resumable-skip if already cached),
2. extract it and locate the ``train/ test/ val/`` root,
3. move those splits into ``data_dir`` (the shipped ``val`` is only 16 images), then
4. hand off to :mod:`split_data` to carve a stratified, patient-grouped validation set.

Usage::

    uv run python chest-x-ray-images-pneumonia/download_data.py
    uv run python chest-x-ray-images-pneumonia/download_data.py --data-dir /some/dir --force
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# ChestXRay2017.zip on Mendeley (DOI 10.17632/rscbjbr9sj.2). The /file_downloaded URL
# 302-redirects to S3; urllib follows redirects automatically.
MENDELEY_URL = (
    "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/"
    "f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded"
)
EXPECTED_BYTES = 1_235_512_464  # size of ChestXRay2017.zip; used as a sanity check
SPLITS = ("train", "val", "test")

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        print(f"\r  downloading... {pct:5.1f}% ({downloaded >> 20} MiB)", end="")


def download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` (skip if already present at the expected size)."""
    if dest.exists() and abs(dest.stat().st_size - EXPECTED_BYTES) < (1 << 20):
        print(f"  cached: {dest} ({dest.stat().st_size >> 20} MiB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # some hosts reject the default urllib agent; install a browser-ish opener
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (download_data.py)")]
    urllib.request.install_opener(opener)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    print()
    tmp.replace(dest)


def find_dataset_root(extract_dir: Path) -> Path:
    """Return the directory that directly contains train/ and test/ subfolders."""
    for candidate in [extract_dir, *sorted(extract_dir.rglob("*"))]:
        if not candidate.is_dir() or "__MACOSX" in candidate.parts:
            continue
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise SystemExit(f"Could not find train/ and test/ under {extract_dir}")


def arrange(dataset_root: Path, data_dir: Path) -> None:
    """Move train/ val/ test/ from the extracted tree into ``data_dir``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        src = dataset_root / split
        dst = data_dir / split
        if dst.exists() or not src.is_dir():
            continue
        shutil.move(str(src), str(dst))


def regroup(data_dir: Path) -> None:
    """Run split_data.py to build the patient-grouped validation split."""
    subprocess.run(
        [sys.executable, str(HERE / "split_data.py"), "--data-dir", str(data_dir)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if .prepared exists"
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    marker = data_dir / ".prepared"

    if marker.exists() and not args.force:
        print(f"Dataset already prepared at {data_dir} (use --force to redo).")
        return

    cache = data_dir / ".cache"
    zip_path = cache / "ChestXRay2017.zip"

    print(f"1/4 Fetching ChestXRay2017.zip -> {zip_path}")
    download(MENDELEY_URL, zip_path)

    print("2/4 Extracting...")
    extract_dir = cache / "extracted"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    dataset_root = find_dataset_root(extract_dir)

    print(f"3/4 Arranging splits into {data_dir}")
    arrange(dataset_root, data_dir)

    print("4/4 Building patient-grouped validation split (split_data.py)")
    regroup(data_dir)

    marker.write_text("ok\n")
    print(f"\nDone. Dataset ready at {data_dir}")


if __name__ == "__main__":
    main()
