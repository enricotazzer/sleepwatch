"""Subject-level cross-validation folds.

The fold file is generated once (``sleepwatch splits make``) and reused by every experiment, so
all results are computed on identical test subjects. Inner splits for tuning and early stopping
are drawn only from an outer fold's training subjects.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from sleepwatch.config import PROJECT_ROOT
from sleepwatch.provenance import git_revision, timestamp

SPLITS_FILE = PROJECT_ROOT / "configs" / "splits" / "subject_folds.json"


def make_folds(nights: list[tuple[str, int]], n_folds: int, seed: int) -> dict[str, int]:
    """Assign each subject to one fold, balancing the number of nights per fold."""
    subjects = np.array([subject for subject, _ in nights])
    splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = {}
    for fold, (_, test_idx) in enumerate(splitter.split(subjects, groups=subjects)):
        for subject in np.unique(subjects[test_idx]):
            folds[str(subject)] = fold
    return dict(sorted(folds.items()))


def save_folds(folds: dict[str, int], path: Path, n_folds: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "created": timestamp(),
        "git": git_revision(),
        "n_folds": n_folds,
        "seed": seed,
        "folds": folds,
    }
    path.write_text(json.dumps(info, indent=2) + "\n")


def load_folds(path: Path = SPLITS_FILE) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `sleepwatch splits make` first")
    return json.loads(path.read_text())["folds"]


def outer_splits(folds: dict[str, int]) -> list[tuple[list[str], list[str]]]:
    """``(train_subjects, test_subjects)`` for each fold, in fold order."""
    return [
        (
            sorted(s for s, f in folds.items() if f != fold),
            sorted(s for s, f in folds.items() if f == fold),
        )
        for fold in sorted(set(folds.values()))
    ]


def inner_split(
    train_subjects: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split training subjects into fit and validation subjects (for tuning/early stopping)."""
    subjects = sorted(train_subjects)
    order = np.random.default_rng(seed).permutation(len(subjects))
    n_val = max(1, math.ceil(val_fraction * len(subjects)))
    val = sorted(subjects[i] for i in order[:n_val])
    fit = sorted(subjects[i] for i in order[n_val:])
    return fit, val
