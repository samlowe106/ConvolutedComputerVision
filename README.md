# Convoluted Computer Vision

This is a repository is a handful of personal studies in deep learning and computer vision, spanning transformers, transfer learning, and so on.

## Running on Google Colab

The training notebooks run on Colab (free GPU) without changes — they call `visualization.colab_bootstrap()`, which mounts Google Drive and resolves where the data and model checkpoints live. Code stays on GitHub; the (gitignored) dataset and `*.keras` checkpoints live on Drive.

[![Open 06 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samlowe106/ConvolutedComputerVision/blob/main/chest-x-ray-images-pneumonia/06_mobilenetv2_transferring.ipynb)
[![Open 07 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samlowe106/ConvolutedComputerVision/blob/main/chest-x-ray-images-pneumonia/07_ensemble.ipynb)

Open a notebook via a badge above and run all cells — the first cell (`colab_bootstrap`) mounts Drive, checks that a GPU is attached, and **auto-downloads the dataset on first launch** (see below). Trained models persist to `MyDrive/cv-checkpoints/`, so they survive session disconnects.

Locally the same notebooks fall back to the repo's `data/` directory and write
checkpoints next to the notebook — no Drive or Colab required.

### Getting the data

The dataset is fetched from its original public host (Kermany et al., [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2)):

```bash
uv run python chest-x-ray-images-pneumonia/download_data.py
```

This downloads `ChestXRay2017.zip` (~1.24 GB), extracts it, and runs[split_data.py](chest-x-ray-images-pneumonia/split_data.py) to build the stratified, patient-grouped validation split. It is idempotent (a `.prepared` marker short-circuits a finished setup). On Colab `colab_bootstrap` runs this for you the first time the data is missing.

## Installation

Make sure you have a compatible version of Python, as specified in the [pyproject.toml file](pyproject.toml), and that you've installed [poetry](https://python-poetry.org/). From there, you should be able to run `poetry install` to install the necessary dependencies and run the project.

## License

Please consult the [license file](LICENSE).
