"""Synthetic nights with known answers, for alignment, loader and feature tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from sleepwatch.constants import EPOCH_S, N2
from sleepwatch.data.loader import Night

REC_LOCAL = "2022-01-10 23:00:00"  # US Eastern, standard time (UTC-5)
REC_START = 1641873600.0  # 2022-01-11 04:00:00 UTC


def still(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrist lying flat: gravity on +z only."""
    return np.zeros_like(t), np.zeros_like(t), np.ones_like(t)


def make_night(
    n_epochs: int = 60,
    hr_fn=lambda t: np.full_like(t, 60.0),
    hr_dt: float = 5.0,
    hr_phase: float = 0.0,
    motion_fn=still,
    motion_hz: float = 50.0,
    expert: np.ndarray | None = None,
    rec_start: float = REC_START,
) -> Night:
    """An in-memory night with signals covering exactly the label window."""
    end = rec_start + n_epochs * EPOCH_S
    ht = np.arange(rec_start + hr_phase, end, hr_dt)
    mt = np.arange(rec_start, end, 1 / motion_hz)
    x, y, z = motion_fn(mt)
    expert = np.full(n_epochs, N2, np.uint8) if expert is None else np.asarray(expert, np.uint8)
    return Night(
        subject="Bidslab99",
        night=1,
        rec_start=rec_start,
        rec_start_local=REC_LOCAL,
        expert=expert,
        dreem=expert.copy(),
        hr=pd.DataFrame({"t": ht, "hr": hr_fn(ht)}),
        motion=pd.DataFrame({"t": mt, "x": x, "y": y, "z": z}).astype(
            {"x": np.float32, "y": np.float32, "z": np.float32}
        ),
    )


def write_night_files(
    folder: Path,
    hr: pd.DataFrame,
    motion: pd.DataFrame,
    expert: np.ndarray,
    dreem: np.ndarray,
    rec_local: str = REC_LOCAL,
) -> None:
    """Write hr.csv / motion.csv / labels.mat in the dataset's on-disk formats."""
    folder.mkdir(parents=True, exist_ok=True)
    hr[["t", "hr"]].to_csv(folder / "hr.csv", header=False, index=False)
    motion[["t", "x", "y", "z"]].rename(columns={"t": "Timestamp"}).to_csv(
        folder / "motion.csv", index=False
    )
    savemat(
        folder / "labels.mat",
        {
            "recStart": rec_local,
            "expert_label": np.asarray(expert, np.uint8)[None, :],
            "dreem_label": np.asarray(dreem, np.uint8)[None, :],
        },
    )
