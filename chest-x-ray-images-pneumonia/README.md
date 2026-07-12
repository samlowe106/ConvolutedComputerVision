# Chest X-Ray Pneumonia (full resolution)

Binary classification of normal vs. pneumonia on the full-resolution Kermany pediatric chest X-ray dataset (about 5.9k images). This is the most developed study in the repo. It walks a progression of architectures and fixes a broken evaluation setup along the way (see [WRITEUP.md](WRITEUP.md) for the full post-mortem).

## Notebooks

| Notebook | Approach |
| --- | --- |
| [01_start](01_start.ipynb) | From-scratch CNN baseline (150x150) |
| [02_resizing](02_resizing.ipynb) | From-scratch CNN on 28x28 |
| [02_vgg19_transferring](02_vgg19_transferring.ipynb) | Transfer ImageNet VGG19 conv weights |
| [03_mnist_transferring_and_resizing](03_mnist_transferring_and_resizing.ipynb) | Transfer from the PneumoniaMNIST net |
| [04_gap](04_gap.ipynb) | GlobalAveragePooling head, which kills the dense-head parameter explosion |
| [05_attention](05_attention.ipynb) | Hybrid CNN backbone feeding a transformer encoder over the feature-map tokens |
| [06_mobilenetv2_transferring](06_mobilenetv2_transferring.ipynb) | Pretrained backbone (DenseNet-121, CheXNet-style), frozen then fine-tuned |
| [07_ensemble](07_ensemble.ipynb) | Backbone ensemble with TTA, calibration, a clinical operating point, and Grad-CAM |

Evaluation is shared across them: deterministic val/test loaders, `val_auc`-based selection with early stopping, F1 and sensitivity-tuned decision thresholds, and the `visualization` helpers.

## Concepts covered

Transfer learning, class imbalance with inverse-frequency weighting, patient-grouped splits to avoid leakage, GlobalAveragePooling, CNN+transformer hybrids, model ensembling, test-time augmentation, probability calibration, threshold selection, Grad-CAM, and shortcut-learning diagnosis.

## Data

The data comes from its original public host, Kermany et al. on [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2) (DOI 10.17632/rscbjbr9sj.2), as `ChestXRay2017.zip` (about 1.24 GB). The URL, SHA-256, and prep steps live in the self-contained [download_data.py](download_data.py).

```bash
uv run python chest-x-ray-images-pneumonia/download_data.py
```

The script verifies the SHA-256, extracts the archive, and runs [split_data.py](split_data.py) to build a stratified, patient-grouped validation split (the shipped 16-image validation set is unusable). On Colab, `colab_bootstrap("chest-x-ray-images-pneumonia")` does this on first launch. See the [top-level README](../README.md#running-on-google-colab) for the Colab flow.
