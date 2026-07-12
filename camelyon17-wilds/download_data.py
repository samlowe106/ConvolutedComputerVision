"""Fetch Camelyon17 (WILDS): a tumor-vs-normal patch dataset with a built-in domain shift.

Source: Bandi et al. / WILDS (Koh et al. 2021), https://wilds.stanford.edu/ . Patches come
from five hospitals; the WILDS split trains on some hospitals and tests on held-out ones, so
it measures out-of-distribution generalization rather than in-distribution accuracy. The
archive is large (~10 GB, 96x96 RGB patches), so this is Colab-only.

We use the official ``wilds`` package to download and lay out the data (it verifies its own
archive). The Colab notebook installs ``wilds`` via ``colab_bootstrap(pip_packages=...)``.

    uv run python camelyon17-wilds/download_data.py [--data-dir DIR]
"""

import argparse
from pathlib import Path

SUBDIR = "camelyon17-wilds"
READY = "camelyon17_v1.0"  # the folder the wilds downloader creates
HERE = Path(__file__).resolve().parent


def fetch(data_dir):
    """Download Camelyon17 into ``data_dir`` via the wilds package (idempotent)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / READY).exists():
        return data_dir
    try:
        from wilds import get_dataset
    except ImportError as exc:
        raise SystemExit(
            "the 'wilds' package is required; on Colab pass "
            "colab_bootstrap(study='camelyon17-wilds', pip_packages=('pooch','seaborn','wilds'))"
        ) from exc
    get_dataset(dataset="camelyon17", download=True, root_dir=str(data_dir))
    return data_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    fetch(parser.parse_args().data_dir)


if __name__ == "__main__":
    main()
