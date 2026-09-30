"""Time alignment between raw signals and 30 s label epochs.

Epoch ``k`` (0-based) covers ``[rec_start + 30k, rec_start + 30(k+1))``, which is the dataset's
1-based formula ``floor((t - recStart) / 30) + 1`` shifted by one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sleepwatch.constants import EPOCH_S


def epoch_index(t: np.ndarray | pd.Series, rec_start: float) -> np.ndarray:
    """0-based epoch of each timestamp; negative before ``rec_start``."""
    return np.floor((np.asarray(t, dtype=float) - rec_start) / EPOCH_S).astype(np.int64)


def crop(df: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    """Rows with ``start <= t < end``."""
    return df[(df["t"] >= start) & (df["t"] < end)].reset_index(drop=True)


def epoch_grid(rec_start: float, n_epochs: int, hz: int) -> np.ndarray:
    """Centres of ``1/hz``-second slots covering all epochs: ``n_epochs * 30 * hz`` timestamps,
    so reshaping a resampled signal to ``(n_epochs, -1)`` gives one row per epoch."""
    return rec_start + (np.arange(n_epochs * EPOCH_S * hz) + 0.5) / hz


def resample(t: np.ndarray, values: np.ndarray, grid: np.ndarray, max_gap: float) -> np.ndarray:
    """Linearly interpolate ``values`` (sampled at sorted times ``t``) onto ``grid``.

    Grid points get NaN when they fall outside the data or when the two samples around them are
    more than ``max_gap`` seconds apart, so gaps are never bridged.
    """
    t = np.asarray(t, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.full(len(grid), np.nan)
    if len(t) < 2:
        return out
    right = np.searchsorted(t, grid, side="right")  # first sample strictly after the grid point
    inside = (right >= 1) & (right <= len(t) - 1)
    r = right[inside]
    bridged = t[r] - t[r - 1] <= max_gap
    idx = np.flatnonzero(inside)[bridged]
    out[idx] = np.interp(grid[idx], t, values)
    return out
