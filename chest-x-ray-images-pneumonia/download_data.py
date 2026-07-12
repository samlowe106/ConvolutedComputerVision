"""Fetch the chest X-ray dataset (self-contained: provenance + fetch + CLI).

Source: Kermany et al. (2018), Mendeley Data, DOI 10.17632/rscbjbr9sj.2 --
``ChestXRay2017.zip`` (~1.24 GB). Same data as the Kaggle mirror, from the original public
host. Downloads (hash-verified), extracts, and runs split_data.py to build a stratified,
patient-grouped validation split (the shipped 16-image val set is unusable).

    uv run python chest-x-ray-images-pneumonia/download_data.py [--data-dir DIR]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from visualization.datasets import fetch_archive, find_dir_containing, strip_metadata

URL = (
    "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/"
    "f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded"
)
SHA256 = "13efc055629733dbab07877f8b3c9f81097840dbcdaa326a8542322c2281ce36"
FNAME = "ChestXRay2017.zip"
SUBDIR = "chest-x-ray-pneumonia"  # stable Drive/data subfolder
READY = "train"  # its presence means "already prepared"

HERE = Path(__file__).resolve().parent


def fetch(data_dir):
    """Download + prepare the dataset into ``data_dir`` (idempotent). Returns ``data_dir``."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / READY).exists():
        return data_dir

    extracted = fetch_archive(URL, SHA256, data_dir, FNAME, kind="zip")
    root = find_dir_containing(
        extracted, needs=("train", "test"), subpaths=("train/NORMAL", "train/PNEUMONIA")
    )
    for split in ("train", "val", "test"):
        src, dst = root / split, data_dir / split
        if src.is_dir() and not dst.exists():
            shutil.move(str(src), str(dst))
    strip_metadata(data_dir)
    subprocess.run(
        [sys.executable, str(HERE / "split_data.py"), "--data-dir", str(data_dir)],
        check=True,
    )
    return data_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    fetch(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
