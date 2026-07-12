"""Fetch the MNIST dataset (self-contained: provenance + fetch + CLI).

Source: the Keras-hosted ``mnist.npz`` (~11 MB), holding ``x_train``/``y_train`` and
``x_test``/``y_test`` -- 28 x 28 uint8 digit images with integer labels 0-9.

    uv run python mnist/download_data.py [--data-dir DIR]
"""

import argparse
from pathlib import Path

from visualization.datasets import fetch_file

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
FNAME = "mnist.npz"
SUBDIR = "mnist"
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
