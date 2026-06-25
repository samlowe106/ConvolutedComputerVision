"""Regroup the Kaggle Chest X-Ray (Kermany) dataset into a usable split.

The dataset ships with a 16-image validation set (8 NORMAL + 8 PNEUMONIA),
which is far too small to select models or drive early stopping on. This script
re-pools ``train`` + ``val`` and carves a fresh validation set out of it that is

* **stratified** - each class is split independently, so the natural class
  ratio is preserved in both train and val;
* **patient-grouped** - all images from one patient land in the same split, so
  the model can never "see" a val patient during training (the dataset has
  ~2.4 pneumonia images per patient and up to 6 per NORMAL study).

The official ``test`` set is left untouched so results stay comparable to other
work on this benchmark. Running the script repeatedly is safe and deterministic:
it always re-pools train + val first, so the split is a pure function of
``--seed`` and ``--val-frac``.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

CLASSES = ("NORMAL", "PNEUMONIA")
DATA_DIR = Path(__file__).resolve().parent / "data"

# Patient/study identifier embedded in each filename. Grouping on these keeps
# every image of a patient together. NORMAL ids keep the optional ``NORMALn-``
# prefix so e.g. ``NORMAL2-IM-1345`` and ``IM-1345`` are treated as distinct.
_PNEUMONIA_ID = re.compile(r"^(person\d+)")
_NORMAL_ID = re.compile(r"^((?:NORMAL\d+-)?IM-\d+)")


def patient_id(cls: str, filename: str) -> str:
    """Return the patient/study id used to group images, or the stem if absent."""
    pattern = _PNEUMONIA_ID if cls == "PNEUMONIA" else _NORMAL_ID
    match = pattern.match(filename)
    return match.group(1) if match else Path(filename).stem


def gather_pool(data_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """Collect (filename, current_path) per class from the train + val pool."""
    pool: dict[str, list[tuple[str, Path]]] = {cls: [] for cls in CLASSES}
    for split in ("train", "val"):
        for cls in CLASSES:
            src = data_dir / split / cls
            if not src.is_dir():
                continue
            for path in sorted(src.iterdir()):
                if path.is_file():
                    pool[cls].append((path.name, path))
    return pool


def plan_split(
    pool: dict[str, list[tuple[str, Path]]], val_frac: float, seed: int
) -> dict[Path, tuple[str, str]]:
    """Map each source path -> (target split, class). Patients never cross splits."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    plan: dict[Path, tuple[str, str]] = {}
    for cls in CLASSES:
        entries = pool[cls]
        names = [name for name, _ in entries]
        if len({(cls, n) for n in names}) != len(names):
            dupes = [n for n, c in Counter(names).items() if c > 1]
            raise SystemExit(f"Duplicate filenames within {cls}: {dupes[:5]} ...")
        groups = [patient_id(cls, name) for name in names]
        train_idx, val_idx = next(splitter.split(entries, groups=groups))
        for idx in train_idx:
            plan[entries[idx][1]] = ("train", cls)
        for idx in val_idx:
            plan[entries[idx][1]] = ("val", cls)
    return plan


def apply_plan(plan: dict[Path, tuple[str, str]], data_dir: Path) -> None:
    for split in ("train", "val"):
        for cls in CLASSES:
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)
    for src, (split, cls) in plan.items():
        dst = data_dir / split / cls / src.name
        if src.resolve() != dst.resolve():
            shutil.move(str(src), str(dst))


def verify(plan: dict[Path, tuple[str, str]], pool_total: int) -> None:
    """Assert counts are conserved and no patient straddles train and val."""
    placed = len(plan)
    if placed != pool_total:
        raise SystemExit(f"File count changed: pooled {pool_total}, placed {placed}")
    for cls in CLASSES:
        patients: dict[str, set[str]] = {"train": set(), "val": set()}
        for src, (split, c) in plan.items():
            if c == cls:
                patients[split].add(patient_id(cls, src.name))
        overlap = patients["train"] & patients["val"]
        if overlap:
            raise SystemExit(f"Patient leakage in {cls}: {sorted(overlap)[:5]} ...")


def summarize(plan: dict[Path, tuple[str, str]], data_dir: Path) -> None:
    counts: Counter[tuple[str, str]] = Counter((s, c) for s, c in plan.values())
    test_counts = {
        cls: sum(1 for _ in (data_dir / "test" / cls).iterdir()) for cls in CLASSES
    }
    print(f"{'split':<8}{'NORMAL':>10}{'PNEUMONIA':>12}{'total':>10}{'%pneu':>8}")
    rows = {
        "train": (counts[("train", "NORMAL")], counts[("train", "PNEUMONIA")]),
        "val": (counts[("val", "NORMAL")], counts[("val", "PNEUMONIA")]),
        "test": (test_counts["NORMAL"], test_counts["PNEUMONIA"]),
    }
    for split, (normal, pneu) in rows.items():
        total = normal + pneu
        pct = pneu / total if total else 0.0
        print(f"{split:<8}{normal:>10}{pneu:>12}{total:>10}{pct:>8.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--dry-run", action="store_true", help="Show the plan without moving files"
    )
    args = parser.parse_args()

    pool = gather_pool(args.data_dir)
    pool_total = sum(len(v) for v in pool.values())
    if pool_total == 0:
        raise SystemExit(f"No images found under {args.data_dir}/(train|val)")

    plan = plan_split(pool, args.val_frac, args.seed)
    verify(plan, pool_total)

    moves = sum(
        1
        for src, (split, cls) in plan.items()
        if src.resolve() != (args.data_dir / split / cls / src.name).resolve()
    )
    print(f"Pool: {pool_total} images | val-frac={args.val_frac} seed={args.seed}")
    print(f"Files to move: {moves}{' (dry run)' if args.dry_run else ''}\n")

    if not args.dry_run:
        apply_plan(plan, args.data_dir)
    summarize(plan, args.data_dir)


if __name__ == "__main__":
    main()
