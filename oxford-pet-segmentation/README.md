# Oxford-IIIT Pet Segmentation

A first semantic-segmentation study: predict a class for every pixel rather than one label per image. Uses the Oxford-IIIT Pet trimaps (foreground, background, boundary) and a small U-Net. This is a scaffold with the data pipeline, a baseline model, and a short training loop; tuning and proper metrics are the next steps.

## Notebook

- [notebook.ipynb](notebook.ipynb): image/mask pipeline, a small U-Net, and a baseline training run.

## Concepts covered

Dense prediction, encoder-decoder / U-Net with skip connections, per-pixel loss, and IoU/Dice metrics (why pixel accuracy misleads when background dominates).

## Data

Oxford-IIIT Pet (Parkhi et al., 2012), the images and trimap masks (about 800 MB total), fetched by the self-contained [download_data.py](download_data.py). It is large and training wants a GPU, so run on Colab, where `colab_bootstrap("oxford-pet-segmentation")` fetches it on first launch. The SHA-256s are unpinned; pooch prints them on first download so you can paste them into the script.

```bash
uv run python oxford-pet-segmentation/download_data.py
```
