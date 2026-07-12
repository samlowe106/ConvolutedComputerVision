"""Fetch the MedMNIST PneumoniaMNIST dataset (self-contained: provenance + fetch + CLI).

Source: Yang et al. (2023), MedMNIST v2, Zenodo record 10519652 -- ``pneumoniamnist.npz``
(~4 MB), holding ``{train,val,test}_images`` (N x 28 x 28 uint8) and ``_labels`` ({0: normal,
1: pneumonia}). Fetched straight from Zenodo -- no Kaggle account or machine-local tfds build.

    uv run python pneumonia_mnist/download_data.py [--data-dir DIR]
"""

import argparse
from pathlib import Path

from visualization.datasets import fetch_file

URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz"
SHA256 = "e1792d3f03751cb101e99f19a63b3c1941436c988665f47853417b05be250cd8"
FNAME = "pneumoniamnist.npz"
SUBDIR = "pneumonia-mnist"
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
