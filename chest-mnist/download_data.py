"""Fetch the MedMNIST ChestMNIST dataset (self-contained: provenance + fetch + CLI).

Source: Yang et al. (2023), MedMNIST v2, Zenodo record 10519652 -- ``chestmnist.npz``. This
is the multi-label sibling of PneumoniaMNIST: 28x28 grayscale chest X-rays with a 14-way
multi-label target (one binary flag per finding, from the NIH ChestX-ray14 taxonomy). The
splits are ``{train,val,test}_images`` (N x 28 x 28 uint8) and ``_labels`` (N x 14 uint8).

The SHA-256 is left unpinned; pooch prints the computed hash on first download, so paste it
into ``SHA256`` to enable verification.

    uv run python chest-mnist/download_data.py [--data-dir DIR]
"""

import argparse
from pathlib import Path

from visualization.datasets import fetch_file

URL = "https://zenodo.org/records/10519652/files/chestmnist.npz"
SHA256 = None  # paste the hash pooch prints on first download
FNAME = "chestmnist.npz"
SUBDIR = "chest-mnist"
READY = FNAME
HERE = Path(__file__).resolve().parent


def fetch(data_dir):
    """Download the dataset into ``data_dir`` (idempotent). Returns ``data_dir``."""
    data_dir = Path(data_dir)
    if (data_dir / READY).exists():
        return data_dir
    fetch_file(URL, SHA256, data_dir, FNAME)
    return data_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    fetch(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
