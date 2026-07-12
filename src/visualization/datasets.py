"""Shared, dataset-agnostic fetch helpers used by each study's ``download_data.py``.

Every study keeps a **self-contained** ``download_data.py`` next to its notebooks: it
declares its own download URL + SHA256 (+ a stable ``SUBDIR``) and defines ``fetch(data_dir)``
by composing the helpers here, so the pooch/download/extract boilerplate lives in one place
rather than being copied per dataset.

``colab_bootstrap(study="<dir>")`` imports that directory's ``download_data.py`` and calls
its ``fetch()``. To add a study: drop a ``download_data.py`` in its directory following the
same shape (see any existing one).
"""

from __future__ import annotations

from pathlib import Path


def fetch_file(url, sha256, data_dir, fname):
    """Download a single file to ``data_dir/fname`` (hash-verified, cached). Returns Path."""
    import pooch

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        pooch.retrieve(
            url=url,
            known_hash="sha256:" + sha256,
            fname=fname,
            path=data_dir,
            progressbar=True,
        )
    )


def fetch_archive(url, sha256, data_dir, fname, kind="zip"):
    """Download + extract an archive under ``data_dir/.cache``; return extracted paths.

    ``kind`` is ``"zip"`` or ``"tar"``. The archive is cached, so re-runs skip the download.
    """
    import pooch

    data_dir = Path(data_dir)
    (data_dir / ".cache").mkdir(parents=True, exist_ok=True)
    processor = (
        pooch.Untar(extract_dir="extracted")
        if kind == "tar"
        else pooch.Unzip(extract_dir="extracted")
    )
    return pooch.retrieve(
        url=url,
        known_hash="sha256:" + sha256,
        fname=fname,
        path=data_dir / ".cache",
        processor=processor,
        progressbar=True,
    )


def find_dir_containing(paths, needs, subpaths=()):
    """Return the first (non-``__MACOSX``) ancestor dir that holds every ``needs`` child.

    ``subpaths`` are extra relative paths that must also exist under it -- e.g. requiring
    ``train/NORMAL`` so we skip the ``__MACOSX`` tree that mirrors folder names with ``._``
    metadata files.
    """
    seen = set()
    for p in map(Path, paths):
        for anc in [p, *p.parents]:
            if anc in seen:
                continue
            seen.add(anc)
            if "__MACOSX" in anc.parts:
                continue
            if all((anc / n).exists() for n in needs) and all(
                (anc / s).exists() for s in subpaths
            ):
                return anc
    raise SystemExit(f"Could not find {needs} in the downloaded archive")


def strip_metadata(root):
    """Remove macOS junk (``.DS_Store``, ``._*``) so image loaders don't choke on it."""
    for junk in Path(root).rglob("*"):
        if junk.is_file() and (junk.name == ".DS_Store" or junk.name.startswith("._")):
            junk.unlink()
