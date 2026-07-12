"""Fetch the CIFAR-10 dataset (self-contained: provenance + fetch + CLI).

Source: Krizhevsky (2009), the official ``cifar-10-python.tar.gz`` (~163 MB) from
https://www.cs.toronto.edu/~kriz/cifar.html. We download (hash-verified), unpack the
pickled batches, and cache a single ``cifar10.npz`` with ``x_train``/``y_train`` (50000 x
32 x 32 x 3 uint8) and ``x_test``/``y_test`` -- the same shape Keras' ``load_data`` returns.

    uv run python cifar-10/download_data.py [--data-dir DIR]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from visualization.datasets import fetch_archive, find_dir_containing

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
FNAME = "cifar-10-python.tar.gz"
SUBDIR = "cifar-10"
READY = "cifar10.npz"

HERE = Path(__file__).resolve().parent


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _to_images(raw):
    # each row is 3072 = 3 x 32 x 32 (R then G then B), row-major -> NHWC uint8
    return raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def fetch(data_dir):
    """Download + convert the dataset to ``data_dir/cifar10.npz`` (idempotent)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / READY
    if out.exists():
        return data_dir

    extracted = fetch_archive(URL, SHA256, data_dir, FNAME, kind="tar")
    root = find_dir_containing(extracted, needs=("data_batch_1", "test_batch"))

    x_train, y_train = [], []
    for i in range(1, 6):
        batch = _unpickle(root / f"data_batch_{i}")
        x_train.append(batch[b"data"])
        y_train += batch[b"labels"]
    test = _unpickle(root / "test_batch")

    np.savez(
        out,
        x_train=_to_images(np.concatenate(x_train)),
        y_train=np.array(y_train, dtype="uint8"),
        x_test=_to_images(test[b"data"]),
        y_test=np.array(test[b"labels"], dtype="uint8"),
    )
    return data_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    fetch(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
