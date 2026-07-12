# CIFAR-10

Multi-class classification of 32x32 natural images across 10 classes, a step up from MNIST in difficulty (color, texture, intra-class variation).

## Notebook

- [notebook.ipynb](notebook.ipynb): CNN classifier.

## Concepts covered

CNNs on natural images, data augmentation, and class weighting on a task that is balanced but harder than MNIST.

## Data

The official `cifar-10-python.tar.gz` (about 163 MB) from [Toronto](https://www.cs.toronto.edu/~kriz/cifar.html), fetched and hash-verified, unpacked, and cached as a single `cifar10.npz` by the self-contained [download_data.py](download_data.py). On Colab, `colab_bootstrap("cifar-10")` runs it on first launch.

```bash
uv run python cifar-10/download_data.py
```
