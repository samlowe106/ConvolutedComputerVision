# Data Science Template

This is a template for a data science project in a Python Jupyter notebook. This README is mostly a template, and should be customized accordingly during project creation. Here's a checklist after making a new repository from this template:

- [ ] Ensure that you've installed [poetry](https://python-poetry.org/)
- [ ] Run `poetry init` to generate the [pyproject.toml](pyproject.toml) file, then `poetry add` and `poetry install` dependencies
- [ ] Create a `data/` directory and add relevant data files there
- [ ] Properly credit whatever data set is being used in the [data set section](#data-set) section of this README
- [ ] Fill in the [license](LICENSE) and (optionally) update the [license section](#license) of this README
- [ ] Fill out or delete the [contributing section](#contributing) of this README

The [setup.sh](setup.sh) script (Linux only!) has been added to assist with Poetry installation and other miscellaneous setup tasks.

## Data Set

The data set can be found [here]().

## License

Please consult the [license file](LICENSE).

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

## Contributing
