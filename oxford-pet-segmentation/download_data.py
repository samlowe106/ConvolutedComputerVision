"""Fetch the Oxford-IIIT Pet dataset: images plus segmentation trimaps.

Source: Parkhi et al. (2012), https://www.robots.ox.ac.uk/~vgg/data/pets/ . Two archives:
``images.tar.gz`` (the RGB photos, ~800 MB) and ``annotations.tar.gz`` (segmentation masks
in ``annotations/trimaps/*.png``, class 1 = foreground, 2 = background, 3 = boundary). Large,
so this is intended to run on Colab.

The SHA-256s are left unpinned; pooch prints the computed hash on first download, so paste
those values into ``IMAGES_SHA256`` / ``ANNOTATIONS_SHA256`` to enable verification.

    uv run python oxford-pet-segmentation/download_data.py [--data-dir DIR]
"""

import argparse
import shutil
from pathlib import Path

from visualization.datasets import fetch_archive

IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
IMAGES_SHA256 = None  # paste the hash pooch prints on first download
ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
ANNOTATIONS_SHA256 = None
SUBDIR = "oxford-pet"
HERE = Path(__file__).resolve().parent


def fetch(data_dir):
    """Download + arrange images/ and annotations/ under ``data_dir`` (idempotent)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "images").is_dir() and (data_dir / "annotations").is_dir():
        return data_dir

    for url, sha in [
        (IMAGES_URL, IMAGES_SHA256),
        (ANNOTATIONS_URL, ANNOTATIONS_SHA256),
    ]:
        fetch_archive(url, sha, data_dir, url.rsplit("/", 1)[-1], kind="tar")

    extracted = data_dir / ".cache" / "extracted"
    for name in ("images", "annotations"):
        src = extracted / name
        if src.is_dir() and not (data_dir / name).exists():
            shutil.move(str(src), str(data_dir / name))
    return data_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    fetch(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
