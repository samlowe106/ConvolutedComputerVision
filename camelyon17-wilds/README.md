# Camelyon17-WILDS (distribution shift)

A first domain-generalization study. Camelyon17 has 96x96 tumor-vs-normal patches from five hospitals, and the WILDS split trains on some hospitals and tests on held-out ones. The headline number is out-of-distribution accuracy, not in-distribution, which exposes shortcut learning directly: a model can ace the training hospitals and fall apart on a new scanner. This is a scaffold; the data access and split explanation are here, and the model plus WILDS group evaluation are the next steps.

## Notebook

- [notebook.ipynb](notebook.ipynb): WILDS data access, the train / id_val / test (OOD) splits, and next-step notes.

## Concepts covered

Distribution shift and domain generalization, in-distribution vs. out-of-distribution evaluation, per-group and worst-group metrics, and why a single accuracy number hides deployment failures.

## Data

Camelyon17 via the [WILDS](https://wilds.stanford.edu/) package (Koh et al., 2021), about 10 GB. The self-contained [download_data.py](download_data.py) calls the wilds downloader, which verifies its own archive. WILDS is PyTorch-native, and the download is large, so run on Colab with `colab_bootstrap("camelyon17-wilds", pip_packages=("pooch", "seaborn", "wilds"))`.

```bash
uv run python camelyon17-wilds/download_data.py
```
