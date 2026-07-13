"""Class weights for imbalanced training, cached per dataset.

A training split's label distribution is fixed, so we compute the inverse-frequency class
weights once and cache them next to the data. Later runs, and the other notebooks that share
a dataset, then skip the recount. For the image datasets that recount otherwise decodes
every training image just to read its label, so the cache is a real saving there.
"""

import json
from pathlib import Path

import numpy as np


def _extract_labels(source):
    import tensorflow as tf

    # image_dataset_from_directory attaches file_paths + class_names, so derive labels from
    # the file list without decoding a single image. That decode pass (thousands of JPEGs
    # read one at a time from a Drive mount) is what makes this slow on Colab.
    paths = getattr(source, "file_paths", None)
    names = getattr(source, "class_names", None)
    if paths is not None and names:
        index = {name: i for i, name in enumerate(names)}
        return np.array([index[Path(p).parent.name] for p in paths])
    if isinstance(source, tf.data.Dataset):
        return np.concatenate([np.asarray(y) for _, y in source], axis=0).ravel()
    return np.asarray(source).ravel()


def class_weights(source=None, *, cache_dir=None, normalize=True):
    """Return inverse-frequency class weights ``{label: weight}`` (up-weights the minority).

    ``source`` is a ``tf.data`` dataset yielding ``(x, y)``, or an array of labels; it is
    only read on a cache miss. When ``cache_dir`` is given, the weights are cached to
    ``<cache_dir>/class_weights.json`` and reused, since a dataset's training class balance
    is fixed. Pass a dataset (not a pre-computed label array) so that a cache hit can skip
    the iteration entirely.
    """
    cache = Path(cache_dir) / "class_weights.json" if cache_dir is not None else None
    if cache is not None and cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}

    if source is None:
        raise ValueError(
            "source is required when there is no cached class_weights.json"
        )
    labels = _extract_labels(source)
    values, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    weights = {int(v): float(total / (len(values) * c)) for v, c in zip(values, counts)}
    if normalize:
        top = max(weights.values())
        weights = {label: w / top for label, w in weights.items()}

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({str(k): v for k, v in weights.items()}, indent=2))
    return weights


def label_pos_weights(labels, cache_dir=None):
    """Per-label positive weights for MULTI-LABEL training: ``n_neg / n_pos`` per column.

    Keras ``class_weight`` is single-label only, so for a multi-label target (an ``(N, L)``
    array of 0/1 flags) the analog is a length-``L`` weight applied to the positive term of
    a weighted binary cross-entropy, up-weighting each label's rarer positive. Cached like
    :func:`class_weights` (the label matrix is small and in memory, so this is cheap; the
    cache just avoids repeating it across runs).
    """
    cache = (
        Path(cache_dir) / "label_pos_weights.json" if cache_dir is not None else None
    )
    if cache is not None and cache.exists():
        return np.asarray(json.loads(cache.read_text()), dtype="float32")

    labels = np.asarray(labels)
    pos = labels.sum(axis=0)
    weights = ((labels.shape[0] - pos) / np.maximum(pos, 1)).astype("float32")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(weights.tolist()))
    return weights
