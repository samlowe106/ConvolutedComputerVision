"""Reusable training/evaluation helpers shared across the notebooks in this repo.

Installed as an editable package (via ``uv sync``), so notebooks just::

    from visualization import summary_graphics, show_confusion_matrix, reset_keras
    from visualization import colab_bootstrap
"""

import gc

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix


def _report_gpu(in_colab):
    """Print whether a GPU is visible -- a forgotten GPU runtime trains on CPU."""
    n_gpu = len(tf.config.list_physical_devices("GPU"))
    if n_gpu:
        print(f"[colab_bootstrap] {n_gpu} GPU(s) visible")
    elif in_colab:
        print(
            "[colab_bootstrap] WARNING: no GPU visible -- training will be very slow. "
            "Set Runtime > Change runtime type > GPU, then Runtime > Restart."
        )
    else:
        print("[colab_bootstrap] no GPU (local CPU run)")


def _ensure_dataset(data_root, dataset_script):
    """Run the repo-relative ``dataset_script`` to fetch data if ``data_root`` is empty."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    if not dataset_script or os.path.isdir(os.path.join(data_root, "train")):
        return
    repo_root = Path(__file__).resolve().parents[2]  # src/visualization/__init__.py
    script = repo_root / dataset_script
    print(f"[colab_bootstrap] data missing -> running {dataset_script}")
    subprocess.run([sys.executable, str(script), "--data-dir", data_root], check=True)


def colab_bootstrap(
    data_subdir,
    *,
    drive_data_root="/content/drive/MyDrive/datasets",
    drive_ckpt_root="/content/drive/MyDrive/cv-checkpoints",
    local_data_dir="data",
    local_ckpt_dir=".",
    pip_packages=("seaborn", "tensorflow-datasets"),
    mount_drive=True,
    dataset_script=None,
):
    """Resolve where data and model checkpoints live, doing Colab setup if needed.

    Returns ``(data_root, ckpt_root)`` so a notebook can run unchanged locally or on
    Google Colab. Locally it just returns ``(local_data_dir, local_ckpt_dir)`` with no
    side effects beyond creating the checkpoint dir. On Colab it:

    1. ``pip install``\\ s only the extras Colab lacks (``pip_packages``) -- deliberately
       NOT tensorflow, so Colab's preinstalled *GPU* build is left intact (the project's
       pinned ``tensorflow-cpu`` would otherwise downgrade it).
    2. mounts Google Drive (so data + checkpoints persist across sessions), and
    3. points ``data_root``/``ckpt_root`` at Drive.

    In both environments it reports GPU visibility and, if ``dataset_script`` is given
    (a repo-relative path) and ``data_root`` has no ``train/`` subdir, runs that script
    (``python <repo>/<dataset_script> --data-dir <data_root>``) to fetch the data.

    Imports specific to each environment (``subprocess``, ``google.colab``) are done
    lazily inside so importing this module never requires them.
    """
    import os
    import sys

    in_colab = "google.colab" in sys.modules
    if not in_colab:
        os.makedirs(local_ckpt_dir, exist_ok=True)
        print(
            f"[colab_bootstrap] local run -> data={local_data_dir!r}, "
            f"ckpt={local_ckpt_dir!r}"
        )
        _ensure_dataset(local_data_dir, dataset_script)
        _report_gpu(in_colab)
        return local_data_dir, local_ckpt_dir

    # 1. install only the missing extras -- NOT tensorflow (keep Colab's GPU build)
    if pip_packages:
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *pip_packages], check=True
        )

    # 2. mount Drive so data + checkpoints survive session disconnects
    if mount_drive:
        from google.colab import drive

        drive.mount("/content/drive")

    data_root = os.path.join(drive_data_root, data_subdir)
    ckpt_root = drive_ckpt_root
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(data_root, exist_ok=True)

    _ensure_dataset(data_root, dataset_script)
    if not os.path.isdir(os.path.join(data_root, "train")):
        print(
            f"[colab_bootstrap] WARNING: {data_root!r} has no train/ data. Upload the "
            "dataset there (train/ val/ test/), or pass dataset_script= to fetch it."
        )
    print(f"[colab_bootstrap] Colab run -> data={data_root!r}, ckpt={ckpt_root!r}")
    _report_gpu(in_colab)
    return data_root, ckpt_root


def reset_keras():
    """Free memory between model runs -- call before building each model.

    Notebooks that train several models in one kernel session accumulate TF graph
    state and matplotlib figures, which can eventually crash the kernel. This
    closes open figures, clears the Keras backend, and runs a gc pass.
    """
    plt.close("all")
    tf.keras.backend.clear_session()
    gc.collect()


def plot_binary_confusion_matrix(
    ax, y_true, y_pred, class_names=("Normal", "Pneumonia")
):
    """Draw a 2x2 confusion matrix (with per-row/column fractions) onto ``ax``."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    n = len(y_pred)
    neg, pos = class_names
    ax.xaxis.set_ticklabels(
        [
            f"{neg} ({(cm[0][0] + cm[1][0]) / n:.3f})",
            f"{pos} ({(cm[0][1] + cm[1][1]) / n:.3f})",
        ]
    )
    ax.yaxis.set_ticklabels(
        [
            f"{neg} ({(cm[0][0] + cm[0][1]) / n:.3f})",
            f"{pos} ({(cm[1][0] + cm[1][1]) / n:.3f})",
        ]
    )


def show_confusion_matrix(
    y_true,
    y_pred,
    class_names=("Normal", "Pneumonia"),
    ax=None,
    title="Confusion Matrix",
):
    """Draw a confusion matrix, creating a standalone figure when ``ax`` is None.

    Shared by ``summary_graphics`` (which passes its own subplot ``ax``) and the
    threshold-tuning cells (which want a standalone figure).
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(5, 5))
    plot_binary_confusion_matrix(ax, y_true, y_pred, class_names)
    ax.set_title(title)
    if standalone:
        plt.show()


def summary_graphics(history, model, test_ds, class_names=("Normal", "Pneumonia")):
    """Plot validation recall/precision (+ AUC if tracked), train/val loss, and
    the test-set confusion matrix.

    ``model`` is evaluated on ``test_ds``; the true labels are read from it, so
    no separate ``y_true`` is needed.
    """
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred = np.round(model.predict(test_ds))

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(16, 5)
    ax = ax.flatten()

    # validation metrics that matter on imbalanced data (recall = 1 - false-neg rate)
    ax[0].plot(history.history["val_recall"], label="Recall", color="g")
    ax[0].plot(history.history["val_precision"], label="Precision", color="b")
    if "val_auc" in history.history:
        ax[0].plot(history.history["val_auc"], label="AUC", color="purple")
    ax[0].grid(True)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Validation score")
    ax[0].set_ylim(0, 1)
    ax[0].legend(loc="lower right")
    ax[0].set_title("Validation Metrics")

    # loss
    ax[1].plot(history.history["loss"], label="Loss")
    ax[1].plot(history.history["val_loss"], label="Val Loss")
    ax[1].grid(True)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend(loc="upper right")
    ax[1].set_title("Loss")

    show_confusion_matrix(y_true, y_pred, ax=ax[2], class_names=class_names)
    plt.show()
    plt.close(fig)  # free the figure so they don't pile up across many models


def plot_confusion_matrix(
    y_true, X_test, model, classes, title="Confusion matrix", cmap=plt.cm.Blues
):
    """Row-normalized multiclass confusion matrix for a softmax ``model``."""
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(classes))
    cm = cm.numpy().astype("float")
    cm = cm / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    threshold = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], ".2f"),
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
        )

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()
