# Convoluted Computer Vision

A handful of personal deep-learning and computer-vision studies, spanning CNNs, transfer learning, transformers, model calibration, segmentation, multi-label, and distribution shift. Each subdirectory is a self-contained study with its own README, notebooks, and data-fetching script.

## Studies

| Study | Task | Highlights |
| --- | --- | --- |
| [chest-x-ray-images-pneumonia](chest-x-ray-images-pneumonia/) | Pneumonia vs. normal (full-res X-ray) | Transfer learning, a CNN+transformer hybrid, and a calibrated backbone ensemble with Grad-CAM. Fixes a broken eval setup ([writeup](chest-x-ray-images-pneumonia/WRITEUP.md)). |
| [pneumonia_mnist](pneumonia_mnist/) | Pneumonia vs. normal (MedMNIST 28x28) | The same task on a curated benchmark; optimizers, a small-CNN ensemble, calibration, and Grad-CAM. |
| [mnist](mnist/) | Handwritten digits | CNN fundamentals warm-up. |
| [cifar-10](cifar-10/) | 10-class natural images | CNNs on color images plus augmentation. |

### Scaffolds (new, to build out on Colab)

| Study | Task | Highlights |
| --- | --- | --- |
| [oxford-pet-segmentation](oxford-pet-segmentation/) | Semantic segmentation (pixel labels) | A small U-Net on the Oxford-IIIT Pet trimaps. Dense prediction, IoU/Dice. |
| [chest-mnist](chest-mnist/) | Multi-label chest findings (28x28) | 14 independent binary labels, per-class AUC. |
| [camelyon17-wilds](camelyon17-wilds/) | Tumor detection under domain shift | Train and test across different hospitals; out-of-distribution generalization. |

## Running on Google Colab

The training notebooks run on Colab (free GPU) unchanged. The first cell calls `colab_bootstrap("<study>")`, which mounts Google Drive, checks that a GPU is attached, and fetches the dataset on first launch. The code stays on GitHub, while the (gitignored) data and `*.keras` checkpoints live on Drive under `MyDrive/cv-checkpoints/`, so they survive session disconnects. Locally the same notebooks fall back to each study's `data/` directory, with no Drive or Colab required.

[![chest 07 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samlowe106/ConvolutedComputerVision/blob/main/chest-x-ray-images-pneumonia/07_ensemble.ipynb)
[![pneumonia-mnist in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samlowe106/ConvolutedComputerVision/blob/main/pneumonia_mnist/01_adam.ipynb)

### Data fetching

Each study has its own self-contained `download_data.py` next to its notebooks. The script declares its download URL and SHA-256 and defines a `fetch(data_dir)` function, composing the shared [pooch](https://www.fatiando.org/pooch/) helpers in [src/visualization/datasets.py](src/visualization/datasets.py). `colab_bootstrap(study=...)` imports that script and runs `fetch()` on first launch, and you can also run it directly:

```bash
uv run python mnist/download_data.py
```

See each study's README for its specific data source.

## Installation

You need a Python version compatible with [pyproject.toml](pyproject.toml) and [uv](https://docs.astral.sh/uv/). Run `uv sync` to install the dependencies, which also installs the shared `visualization` package in editable mode so the notebooks can `import visualization`.

## License

Please consult the [license file](LICENSE).
