"""Reusable training/evaluation plots shared across the notebooks in this repo.

The notebooks live one directory below the repo root, so import with:

    import sys

    sys.path.insert(0, "..")  # repo root, so `visualization` is importable
    from visualization import summary_graphics, plot_confusion_matrix
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix


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
