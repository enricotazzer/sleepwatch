"""Night-level anomaly channels, the night-level personal baseline, and alarm calibration.

Channels (all one-sided, in the "worse sleep" direction):

- ``hr_window``: highest mean heart-rate z over any 30-min window of the sleep period;
- ``hr_night``: mean heart-rate z over the sleep period;
- ``frag``: wake bouts (>= 1 min) per hour after sleep onset (stage-free: movement bursts,
  runs with population activity z above a threshold), z-scored against the person's earlier
  nights;
- ``onset``: minutes from recording start to persistent sleep (stage-free: the actigraphy
  onset), z-scored against the person's earlier nights.

Alarm: each channel's tail probability under a null (clean nights of other people, scored the
same way); a night's score is its smallest tail probability. Tail probabilities are discrete, so
the threshold is the largest value at which at most ``target_fpr`` of the null nights (scored
leave-one-out) would be flagged; the rate actually reached is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from sleepwatch.constants import EPOCH_S, WAKE

CHANNELS = ("hr_window", "hr_night", "frag", "onset")
MATCHING = {"hr": ("hr_window", "hr_night"), "frag": ("frag",), "onset": ("onset",)}
NIGHT_LEVEL = ("frag", "onset")  # z-scored against the person's earlier nights
EPOCHS_PER_HOUR = 3600 // EPOCH_S


class ChannelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_epochs: int = 60
    persistent_epochs: int = 20
    wake_bout_epochs: int = 2
    burst_z: float = 3.0


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of each run of True values."""
    padded = np.r_[False, np.asarray(mask, bool), False].astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def sleep_period(stage, hours_since_onset, staged: bool, persistent: int):
    """``(onset, end)`` epochs of the sleep period, or ``(None, None)`` if no sleep was found.

    Staged: onset is the first epoch of the first run of ``persistent`` sleep epochs, end is
    after the last sleep epoch. Stage-free: onset is the actigraphy onset, end is the night end.
    """
    if staged:
        stage = np.asarray(stage, float)
        asleep = ~np.isnan(stage) & (stage != WAKE)
        long_runs = [r for r in runs(asleep) if r[1] - r[0] >= persistent]
        if not long_runs:
            return None, None
        return long_runs[0][0], int(np.flatnonzero(asleep)[-1]) + 1
    after = np.flatnonzero(np.asarray(hours_since_onset, float) >= 0)
    return (int(after[0]), len(stage)) if len(after) else (None, None)


def _window_max(z: np.ndarray, width: int) -> tuple[float, int]:
    """Highest mean over windows of ``width`` (at least half valid) and the window's start."""
    valid = ~np.isnan(z)
    if len(z) <= width:
        return (float(np.nanmean(z)) if valid.any() else np.nan), 0
    sums = np.convolve(np.where(valid, z, 0.0), np.ones(width), "valid")
    counts = np.convolve(valid.astype(float), np.ones(width), "valid")
    means = np.where(counts >= width / 2, sums / np.maximum(counts, 1), -np.inf)
    start = int(np.argmax(means))
    return (float(means[start]) if np.isfinite(means[start]) else np.nan), start


def night_channels(night: pd.DataFrame, staged: bool, settings: ChannelSettings) -> dict:
    """Raw channel values for one night (epoch-sorted rows with ``stage``, ``z_hr``,
    ``z_act_pop`` and ``hours_since_onset``), plus where they come from."""
    n = len(night)
    onset, end = sleep_period(
        night["stage"].to_numpy(),
        night["hours_since_onset"].to_numpy(),
        staged,
        settings.persistent_epochs,
    )
    if onset is None:
        return {
            "hr_window": np.nan,
            "hr_night": np.nan,
            "frag": np.nan,
            "onset": n * EPOCH_S / 60,
            "sleep_onset": None,
            "sleep_end": None,
            "window": None,
            "bouts": [],
        }
    z = night["z_hr"].to_numpy()[onset:end]
    hr_window, start = _window_max(z, settings.window_epochs)
    width = min(settings.window_epochs, end - onset)
    if staged:
        wake = night["stage"].to_numpy()[onset:end] == WAKE
        bouts = [r for r in runs(wake) if r[1] - r[0] >= settings.wake_bout_epochs]
    else:
        bouts = runs(night["z_act_pop"].to_numpy()[onset:end] > settings.burst_z)
    hours = (end - onset) / EPOCHS_PER_HOUR
    return {
        "hr_window": hr_window,
        "hr_night": float(np.nanmean(z)) if (~np.isnan(z)).any() else np.nan,
        "frag": len(bouts) / hours if hours > 0 else np.nan,
        "onset": onset * EPOCH_S / 60,
        "sleep_onset": onset,
        "sleep_end": end,
        "window": (onset + start, onset + start + width),
        "bouts": [(onset + a, onset + b) for a, b in bouts],
    }


@dataclass
class NightPrior:
    """Population mean, within-person between-night SD and shrinkage per night-level channel."""

    mean: dict[str, float]
    sd: dict[str, float]
    kappa: dict[str, float]


def fit_night_prior(values: pd.DataFrame) -> NightPrior:
    """From per-night channel values (columns ``subject`` + ``NIGHT_LEVEL``) of training nights."""
    mean, sd, kappa = {}, {}, {}
    for channel in NIGHT_LEVEL:
        frame = values[["subject", channel]].dropna()
        groups = frame.groupby("subject")[channel]
        sizes = groups.size()
        within = float(((frame[channel] - groups.transform("mean")) ** 2).sum())
        within /= max(int((sizes - 1).clip(lower=0).sum()), 1)
        between = float(groups.mean().var(ddof=1) - within * (1 / sizes).mean())
        mean[channel] = float(frame[channel].mean())
        sd[channel] = float(np.sqrt(within)) if within > 0 else 1.0
        kappa[channel] = max(within / max(between, within * 1e-3), 1e-6) if within > 0 else 1e-6
    return NightPrior(mean, sd, kappa)


def night_z(value: float, earlier: pd.Series, prior: NightPrior, channel: str, personal: bool):
    """``value`` against the person's earlier nights (shrunk toward the population mean)."""
    expected = prior.mean[channel]
    if personal:
        earlier = earlier.dropna()
        expected += float((earlier - prior.mean[channel]).sum()) / (
            len(earlier) + prior.kappa[channel]
        )
    return (value - expected) / prior.sd[channel]


def tail_p(null: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Share of null values at least as large (with +1 smoothing); NaN values get p = 1."""
    null = np.sort(null[~np.isnan(null)])
    values = np.asarray(values, float)
    above = len(null) - np.searchsorted(null, values, side="left")
    p = (1 + above) / (1 + len(null))
    return np.where(np.isnan(values), 1.0, p)


@dataclass
class Calibration:
    null: pd.DataFrame  # channel values of the null nights
    threshold: float  # alarm when a night's smallest tail probability is at or below this
    null_rate: float  # share of null nights (leave-one-out) at or below the threshold
    centre: dict[str, float]
    scale: dict[str, float]


def calibrate(null: pd.DataFrame, target_fpr: float) -> Calibration:
    """Alarm threshold from null nights, using leave-one-out tail probabilities."""
    loo = []
    values = null[list(CHANNELS)].to_numpy(dtype=float)
    for i in range(len(values)):
        others = np.delete(values, i, axis=0)
        loo.append(min(tail_p(others[:, c], values[i : i + 1, c])[0] for c in range(len(CHANNELS))))
    loo = np.asarray(loo)
    allowed = [t for t in np.unique(loo) if np.mean(loo <= t) <= target_fpr]
    threshold = float(max(allowed)) if allowed else 0.0
    null_rate = float(np.mean(loo <= threshold))
    centre = {c: float(np.nanmedian(null[c])) for c in CHANNELS}
    scale = {c: float(1.4826 * np.nanmedian(np.abs(null[c] - centre[c]))) or 1.0 for c in CHANNELS}
    return Calibration(null[list(CHANNELS)].copy(), threshold, null_rate, centre, scale)


def evaluate(values: pd.DataFrame, calibration: Calibration) -> pd.DataFrame:
    """Tail probability per channel, the smallest, the alarm, the channel with the smallest
    probability, and a continuous score for ranking (largest robust-standardized channel)."""
    out = pd.DataFrame(index=values.index)
    for channel in CHANNELS:
        out[f"p_{channel}"] = tail_p(
            calibration.null[channel].to_numpy(dtype=float), values[channel].to_numpy()
        )
    p = out[[f"p_{c}" for c in CHANNELS]].to_numpy()
    out["min_p"] = p.min(axis=1)
    out["flagged"] = out["min_p"] <= calibration.threshold
    robust = np.column_stack(
        [
            (values[c].to_numpy(dtype=float) - calibration.centre[c]) / calibration.scale[c]
            for c in CHANNELS
        ]
    )
    robust = np.where(np.isnan(robust), -np.inf, robust)
    out["score"] = robust.max(axis=1)
    # Ties in the tail probability (common beyond the null's range) go to the larger robust value.
    top = np.argmin(p - 1e-12 * np.clip(robust, -1e4, 1e4), axis=1)
    out["top_channel"] = np.array(CHANNELS)[top]
    return out
