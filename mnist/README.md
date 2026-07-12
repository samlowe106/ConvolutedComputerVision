# MNIST

Multi-class classification of handwritten digits, the canonical warm-up. A small CNN on the 28x28 MNIST digits.

## Notebook

- [notebook.ipynb](notebook.ipynb): CNN classifier.

## Concepts covered

CNN fundamentals, multi-class softmax with categorical cross-entropy, and a baseline to compare the harder studies against.

## Data

The Keras-hosted `mnist.npz` (about 11 MB), fetched and hash-verified by the self-contained [download_data.py](download_data.py). On Colab, `colab_bootstrap("mnist")` runs it on first launch.

```bash
uv run python mnist/download_data.py
```
