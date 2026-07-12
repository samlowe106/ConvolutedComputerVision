# PneumoniaMNIST

The same normal vs. pneumonia task on the MedMNIST PneumoniaMNIST subset: center-cropped, intensity-normalized 28x28 grayscale images with a real validation split of about 520 images. It "just works" where the full-resolution study struggled, which is itself instructive (see the chest study's [WRITEUP.md](../chest-x-ray-images-pneumonia/WRITEUP.md)).

## Notebooks

| Notebook | Approach |
| --- | --- |
| [01_adam](01_adam.ipynb) | Small dense and CNN nets trained with Adam |
| [02_adamw](02_adamw.ipynb) | The same, with AdamW (decoupled weight decay) |
| [03_small_cnn](03_small_cnn.ipynb) | A from-scratch CNN with val-AUC selection, threshold tuning, and Grad-CAM (mirrors chest [06](../chest-x-ray-images-pneumonia/06_mobilenetv2_transferring.ipynb)) |
| [04_ensemble](04_ensemble.ipynb) | A two-CNN ensemble with test-time augmentation, calibration, balanced and clinical operating points, and Grad-CAM (mirrors chest [07](../chest-x-ray-images-pneumonia/07_ensemble.ipynb)) |

## Concepts covered

Working on a curated benchmark, comparing optimizers (Adam vs. AdamW), class weighting, and seeing how much of the "hard" real-world performance is about data curation rather than the model. Notebooks 03 and 04 add the chest study's toolkit at 28x28: model ensembling, test-time augmentation, probability calibration, threshold selection, and Grad-CAM.

## Data

The MedMNIST `pneumoniamnist.npz` (about 4 MB), fetched from [Zenodo](https://zenodo.org/records/10519652) (Yang et al., MedMNIST v2). No Kaggle account or machine-local `tfds` build is required. The URL and SHA-256 live in the self-contained [download_data.py](download_data.py).

```bash
uv run python pneumonia_mnist/download_data.py
```

On Colab, `colab_bootstrap("pneumonia_mnist")` fetches it on first launch. See the [top-level README](../README.md#running-on-google-colab) for the Colab flow.
