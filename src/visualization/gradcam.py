"""Grad-CAM heatmaps, shared across the notebooks.

Works for two model shapes, as long as the model is a linear stack of layers:

* a **flat CNN** (the model's own conv layers, as in the small-CNN notebooks), where the
  feature map is the last convolutional layer, and
* a **nested pretrained backbone** (a sub-``Model`` inside the model, as in the transfer
  notebooks), where the feature map is the backbone's output.

We rebuild two sub-models by re-calling the trained layers: a feature model (input up to
and including the feature layer) and a head model (everything after). Because the feature
model includes the model's own input preprocessing (a ``Rescaling`` layer, or an
applications ``preprocess_input`` folded in as ops), you feed it raw ``[0, 255]`` images;
no external preprocessing is needed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

_CONV = (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)


def _split_at_feature_map(model):
    """Return (feature_model, head_model) split at the last conv layer or nested backbone."""
    layers = [
        layer
        for layer in model.layers
        if not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    feat_idx = max(
        i
        for i, layer in enumerate(layers)
        if isinstance(layer, tf.keras.Model) or isinstance(layer, _CONV)
    )

    inp = tf.keras.Input(model.inputs[0].shape[1:])
    x = inp
    for layer in layers[: feat_idx + 1]:
        x = layer(x)
    feature_model = tf.keras.Model(inp, x)

    head_in = tf.keras.Input(x.shape[1:])
    h = head_in
    for layer in layers[feat_idx + 1 :]:
        h = layer(h)
    head_model = tf.keras.Model(head_in, h)
    return feature_model, head_model


def gradcam_heatmaps(model, images):
    """Return ``(heatmaps, scores)`` for a batch of raw ``[0, 255]`` images.

    ``heatmaps`` is ``(N, h, w)`` normalized to 0..1 at the feature-map resolution;
    ``scores`` is the ``(N,)`` sigmoid output. Resize the heatmaps to the image size for
    display.
    """
    feature_model, head_model = _split_at_feature_map(model)
    x = tf.cast(images, tf.float32)
    with tf.GradientTape() as tape:
        conv = feature_model(x)
        tape.watch(conv)
        score = head_model(conv)[:, 0]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)  # GAP over spatial dims
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    return cam.numpy(), score.numpy()


def show_gradcam(image, model, class_name="pneumonia", true_label=None, title=None):
    """Render a single Grad-CAM graphic: the image beside its heatmap overlay.

    ``image`` is a single ``(H, W, C)`` raw image. Returns the predicted score.
    """
    cam, score = gradcam_heatmaps(model, image[None])
    h, w = image.shape[:2]
    heat = tf.image.resize(cam[0][..., None], (h, w)).numpy()[..., 0]

    img = np.asarray(image).astype("uint8")
    gray = img.shape[-1] == 1
    shown = img.squeeze() if gray else img
    cmap = "gray" if gray else None

    fig, (ax_img, ax_cam) = plt.subplots(1, 2, figsize=(9, 5))
    ax_img.imshow(shown, cmap=cmap)
    ax_img.set_title(f"input (true: {true_label})" if true_label else "input")
    ax_img.axis("off")
    ax_cam.imshow(shown, cmap=cmap)
    heatmap = ax_cam.imshow(heat, cmap="jet", alpha=0.45)
    ax_cam.set_title(f"Grad-CAM (predicted {class_name}: {score[0]:.2f})")
    ax_cam.axis("off")
    fig.colorbar(heatmap, ax=ax_cam, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return score[0]
