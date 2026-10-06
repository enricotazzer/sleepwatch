"""Label-free screen for implausible heart-rate stretches (Phase 3b, decision 47).

An epoch is *suspect* when its heart rate stays at least ``above_median_bpm`` above its night
version's median for at least ``min_epochs`` consecutive epochs; a missing epoch breaks a run.
The rule can't tell a sensor artifact from a real abrupt tachycardia, so suspect stretches are
reported (:func:`describe`) and the detector treats their heart rate as missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from sleepwatch.anomaly.scores import runs
from sleepwatch.constants import EPOCH_S


class ScreenSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    above_median_bpm: float = 40.0
    min_epochs: int = 10  # 5 min


def suspect_night(hr: np.ndarray, settings: ScreenSettings) -> np.ndarray:
    """Suspect epochs of one night version, from its epoch heart rate in epoch order."""
    hr = np.asarray(hr, dtype=float)
    out = np.zeros(len(hr), dtype=bool)
    if np.isnan(hr).all():
        return out
    with np.errstate(invalid="ignore"):
        high = hr >= np.nanmedian(hr) + settings.above_median_bpm  # NaN compares False
    for a, b in runs(high):
        if b - a >= settings.min_epochs:
            out[a:b] = True
    return out


def suspect_epochs(
    rows: pd.DataFrame, settings: ScreenSettings, hr_column: str = "hr_mean"
) -> pd.Series:
    """Suspect flag per row. ``rows`` hold whole nights (any order), one version each per
    ``subject``, ``night`` and, if present, ``version``."""
    keys = ["subject", "night", *(["version"] if "version" in rows else [])]
    out = pd.Series(False, index=rows.index)
    ordered = rows.sort_values([*keys, "epoch"])
    for _, group in ordered.groupby(keys, sort=False):
        out.loc[group.index] = suspect_night(group[hr_column].to_numpy(), settings)
    return out


def describe(hr: np.ndarray, settings: ScreenSettings) -> dict | None:
    """Minutes and peak epoch heart rate of one night version's suspect epochs (None if none)."""
    hr = np.asarray(hr, dtype=float)
    mask = suspect_night(hr, settings)
    if not mask.any():
        return None
    return {"minutes": float(mask.sum() * EPOCH_S / 60), "peak_bpm": float(np.nanmax(hr[mask]))}


def note(found: dict) -> str:
    """The data-quality note a summary carries for a suspect stretch."""
    return (
        f"implausible heart-rate stretch: {found['minutes']:.0f} min at up to "
        f"{found['peak_bpm']:.0f} bpm, ignored by the detector (possible measurement artifact)"
    )
