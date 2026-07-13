"""Colab/local bootstrap: resolve where data and checkpoints live, fetch the dataset,
and report GPU visibility.

``colab_bootstrap`` is the entry point the notebooks call; the rest supports it.
"""

import tensorflow as tf


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


def _load_study_fetcher(study):
    """Import a study directory's self-contained ``download_data.py`` module."""
    import importlib.util
    import sys
    from pathlib import Path

    study_dir = Path(__file__).resolve().parents[2] / study
    # put the study dir on the path so the script's own imports (e.g. split_data) resolve
    if str(study_dir) not in sys.path:
        sys.path.insert(0, str(study_dir))
    name = "_download_data_" + study.replace("/", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, study_dir / "download_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def colab_bootstrap(
    study=None,
    *,
    drive_data_root="/content/drive/MyDrive/datasets",
    drive_ckpt_root="/content/drive/MyDrive/cv-checkpoints",
    local_data_dir="data",
    local_ckpt_dir=".",
    pip_packages=("pooch", "seaborn"),
    mount_drive=True,
):
    """Resolve where data and model checkpoints live, doing Colab setup if needed.

    Returns ``(data_root, ckpt_root)`` so a notebook can run unchanged locally or on
    Google Colab. ``study`` is a study directory name (e.g. ``"mnist"``); when given, that
    directory's self-contained ``download_data.py`` is imported and its ``fetch(data_root)``
    is called (idempotent), and ``data_root`` points at the fetched data. On Colab the Drive
    subfolder comes from the script's ``SUBDIR`` constant.

    On Colab it also:

    1. ``pip install``\\ s only the extras Colab lacks (``pip_packages``), not tensorflow,
       so Colab's preinstalled GPU build is left intact (reinstalling tensorflow from the
       project's dependencies would disturb that build).
    2. mounts Google Drive (so data + checkpoints persist across sessions), and
    3. points ``data_root``/``ckpt_root`` at Drive.

    In both environments it reports GPU visibility. Environment-specific imports
    (``subprocess``, ``google.colab``) are done lazily so importing this module is cheap.
    """
    import os
    import sys

    fetcher = _load_study_fetcher(study) if study else None
    in_colab = "google.colab" in sys.modules

    if not in_colab:
        os.makedirs(local_ckpt_dir, exist_ok=True)
        print(
            f"[colab_bootstrap] local run -> data={local_data_dir!r}, "
            f"ckpt={local_ckpt_dir!r}"
        )
        if fetcher:
            fetcher.fetch(local_data_dir)
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

    subdir = getattr(fetcher, "SUBDIR", study) if fetcher else "data"
    data_root = os.path.join(drive_data_root, subdir)
    ckpt_root = drive_ckpt_root
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(data_root, exist_ok=True)

    if fetcher:
        fetcher.fetch(data_root)  # idempotent: skips if already on Drive
    print(f"[colab_bootstrap] Colab run -> data={data_root!r}, ckpt={ckpt_root!r}")
    _report_gpu(in_colab)
    return data_root, ckpt_root
