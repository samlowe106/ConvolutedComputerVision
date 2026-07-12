# ChestMNIST (multi-label)

A first multi-label study: each 28x28 chest X-ray can carry several findings at once, so the target is 14 independent binary flags (the NIH ChestX-ray14 taxonomy) rather than one class. The head is 14 sigmoids with binary cross-entropy, scored with per-class AUC. This is the multi-label sibling of [pneumonia_mnist](../pneumonia_mnist/), and a scaffold to build on.

## Notebook

- [notebook.ipynb](notebook.ipynb): npz loading, a small CNN with 14 sigmoid outputs, a baseline run, and per-class AUC.

## Concepts covered

Multi-label vs. single-label classification, sigmoid + binary cross-entropy over many labels, per-class AUC, and heavy label imbalance (most labels are 0).

## Data

The MedMNIST `chestmnist.npz`, fetched from [Zenodo](https://zenodo.org/records/10519652) (Yang et al., MedMNIST v2) by the self-contained [download_data.py](download_data.py). On Colab, `colab_bootstrap("chest-mnist")` fetches it on first launch. The SHA-256 is unpinned; pooch prints it on first download so you can paste it into the script.

```bash
uv run python chest-mnist/download_data.py
```
