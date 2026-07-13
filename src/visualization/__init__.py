"""Reusable helpers shared across the notebooks in this repo.

Installed as an editable package (via ``uv sync``), so notebooks just::

    from visualization import summary_graphics, show_confusion_matrix, reset_keras
    from visualization import colab_bootstrap, gradcam_heatmaps, show_gradcam

Implementation lives in the submodules (``colab``, ``plots``, ``gradcam``, ``datasets``);
this module only re-exports the public API.
"""

from .colab import colab_bootstrap
from .gradcam import gradcam_heatmaps, show_gradcam
from .plots import (
    plot_binary_confusion_matrix,
    plot_confusion_matrix,
    reset_keras,
    show_confusion_matrix,
    summary_graphics,
)
from .weights import class_weights, label_pos_weights

__all__ = [
    "colab_bootstrap",
    "gradcam_heatmaps",
    "show_gradcam",
    "reset_keras",
    "show_confusion_matrix",
    "plot_binary_confusion_matrix",
    "summary_graphics",
    "plot_confusion_matrix",
    "class_weights",
    "label_pos_weights",
]
