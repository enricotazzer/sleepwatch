"""Per-epoch features from one night's heart rate and accelerometry.

Signals are first resampled onto uniform grids (HR at ``hr_grid_hz``, motion at ``motion_hz``)
without bridging gaps, so features don't depend on the watch's sampling rate, which varies between
nights (HR every 2-5 s; motion at ~50 Hz or ~33 Hz). Labels are never used to compute features.

Scope notes, all acceptable for morning-after analysis but not for real-time use:
- night-relative HR (``hr_rel``, ``hr_z``), detrending and context windows use the whole night,
  including epochs after the current one;
- the activity band-pass filter runs forward and backward.

Columns starting with ``qc_`` describe data coverage. They are not model inputs: coverage is a
device artefact and could act as a shortcut.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.signal import butter, sosfiltfilt

from sleepwatch.constants import EPOCH_S, RECSTART_TZ, UNKNOWN
from sleepwatch.data.align import epoch_grid, epoch_index, resample
from sleepwatch.data.loader import Night

META_COLUMNS = ["subject", "night", "epoch", "t_start", "expert", "dreem", "labeled"]


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "v1"
    hr_grid_hz: int = 1
    hr_max_gap_s: float = 30.0  # don't interpolate HR across longer gaps
    motion_hz: int = 50
    motion_max_gap_s: float = 1.0
    min_coverage: float = 0.5  # epochs with less valid signal get NaN features
    activity_band_hz: tuple[float, float] = (0.25, 2.5)
    angle_block_s: int = 5  # arm-angle averaging block (van Hees et al.)
    detrend_window_epochs: int = 41  # centred rolling median for HR detrending (~20 min)
    onset_window_epochs: int = 20  # sustained stillness needed to estimate sleep onset (10 min)
    onset_angle_deg: float = 5.0  # max arm-angle change per block that still counts as still
    context_windows: list[int] = [2, 5, 10]  # +-k epochs
    context_features: list[str] = ["hr_mean", "hr_detrend", "enmo_mean", "activity", "angle_change"]

    @model_validator(mode="after")
    def _blocks_tile_epochs(self) -> FeatureConfig:
        if EPOCH_S % self.angle_block_s:
            raise ValueError(f"angle_block_s must divide the {EPOCH_S} s epoch")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> FeatureConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model-input columns: everything except metadata, labels and ``qc_`` coverage columns."""
    return [c for c in df.columns if c not in META_COLUMNS and not c.startswith("qc_")]


def _per_epoch(grid_values: np.ndarray, n_epochs: int, stat, ok: np.ndarray) -> np.ndarray:
    """Row-wise NaN-ignoring statistic of a grid signal reshaped to one row per epoch."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN epochs
        out = stat(grid_values.reshape(n_epochs, -1), axis=1)
    return np.where(ok, out, np.nan)


def _hr_features(night: Night, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    n = night.n_epochs
    grid = epoch_grid(night.rec_start, n, cfg.hr_grid_hz)
    hr = resample(night.hr["t"], night.hr["hr"], grid, cfg.hr_max_gap_s)
    coverage = np.isfinite(hr).reshape(n, -1).mean(axis=1)
    ok = coverage >= cfg.min_coverage
    hr_mean = _per_epoch(hr, n, np.nanmean, ok)

    median = np.nanmedian(hr_mean) if ok.any() else np.nan
    q75, q25 = np.nanpercentile(hr_mean, [75, 25]) if ok.any() else (np.nan, np.nan)
    robust_sd = (q75 - q25) / 1.349
    trend = (
        pd.Series(hr_mean)
        .rolling(cfg.detrend_window_epochs, center=True, min_periods=5)
        .median()
        .to_numpy()
    )
    return {
        "qc_hr_coverage": coverage,
        "qc_hr_samples": np.bincount(epoch_index(night.hr["t"], night.rec_start), minlength=n)[:n],
        "hr_mean": hr_mean,
        "hr_std": _per_epoch(hr, n, np.nanstd, ok),
        "hr_min": _per_epoch(hr, n, np.nanmin, ok),
        "hr_max": _per_epoch(hr, n, np.nanmax, ok),
        "hr_rel": hr_mean - median,
        "hr_z": (hr_mean - median) / robust_sd if robust_sd > 0 else np.full(n, np.nan),
        "hr_detrend": hr_mean - trend,
        "hr_diff": np.diff(hr_mean, prepend=np.nan),
    }


def _motion_features(night: Night, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    n, hz = night.n_epochs, cfg.motion_hz
    grid = epoch_grid(night.rec_start, n, hz)
    t = night.motion["t"].to_numpy()
    x, y, z = (resample(t, night.motion[c].to_numpy(), grid, cfg.motion_max_gap_s) for c in "xyz")
    valid = np.isfinite(x)
    coverage = valid.reshape(n, -1).mean(axis=1)
    ok = coverage >= cfg.min_coverage

    mag = np.sqrt(x**2 + y**2 + z**2)
    enmo = np.clip(mag - 1.0, 0.0, None)  # Euclidean norm minus one g

    # Activity: mean absolute band-passed magnitude. Gaps are filled with 1 g (rest) for the
    # filter only and masked again afterwards.
    sos = butter(4, cfg.activity_band_hz, btype="bandpass", fs=hz, output="sos")
    filtered = sosfiltfilt(sos, np.where(valid, mag, 1.0))
    activity = np.where(valid, np.abs(filtered), np.nan)

    # Arm angle (van Hees et al. 2015): change between consecutive block means, per epoch.
    angle = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    block = cfg.angle_block_s * hz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        block_angle = np.nanmean(angle.reshape(-1, block), axis=1)
    change = np.abs(np.diff(block_angle, prepend=np.nan))

    return {
        "qc_motion_coverage": coverage,
        "enmo_mean": _per_epoch(enmo, n, np.nanmean, ok),
        "enmo_max": _per_epoch(enmo, n, np.nanmax, ok),
        "mag_std": _per_epoch(mag, n, np.nanstd, ok),
        "activity": _per_epoch(activity, n, np.nanmean, ok),
        "angle_mean": _per_epoch(angle, n, np.nanmean, ok),
        "angle_change": _per_epoch(change, n, np.nanmean, ok),
        "angle_change_max": _per_epoch(change, n, np.nanmax, ok),
    }


def estimate_onset(angle_change_max: np.ndarray, window: int, threshold_deg: float) -> int | None:
    """First epoch that starts ``window`` consecutive still epochs (label-free sleep onset)."""
    still = pd.Series((angle_change_max < threshold_deg).astype(int))  # NaN counts as not still
    run = still.rolling(window).sum().shift(-(window - 1))
    hits = np.flatnonzero(run.to_numpy() == window)
    return int(hits[0]) if hits.size else None


def _hours_since_noon(rec_start: float, n_epochs: int) -> np.ndarray:
    local = datetime.fromtimestamp(rec_start, ZoneInfo(RECSTART_TZ))
    start_h = local.hour + local.minute / 60 + local.second / 3600
    return (start_h - 12 + np.arange(n_epochs) * EPOCH_S / 3600) % 24


def night_features(night: Night, cfg: FeatureConfig) -> pd.DataFrame:
    """One row per epoch: metadata, labels, ``qc_`` coverage and features."""
    n = night.n_epochs
    epochs = np.arange(n)
    df = pd.DataFrame(
        {
            "subject": night.subject,
            "night": night.night,
            "epoch": epochs,
            "t_start": night.rec_start + epochs * EPOCH_S,
            "expert": night.expert,
            "dreem": night.dreem,
            "labeled": night.expert != UNKNOWN,
        }
    )
    for part in (_hr_features(night, cfg), _motion_features(night, cfg)):
        for name, values in part.items():
            df[name] = values

    onset = estimate_onset(
        df["angle_change_max"].to_numpy(), cfg.onset_window_epochs, cfg.onset_angle_deg
    )
    df["hours_since_start"] = epochs * EPOCH_S / 3600
    df["hours_since_onset"] = (epochs - onset) * EPOCH_S / 3600 if onset is not None else np.nan
    df["hours_since_noon"] = _hours_since_noon(night.rec_start, n)

    for feature in cfg.context_features:
        series = df[feature]
        for k in cfg.context_windows:
            rolling = series.rolling(2 * k + 1, center=True, min_periods=1)
            df[f"{feature}_mean_w{k}"] = rolling.mean()
            df[f"{feature}_std_w{k}"] = rolling.std()
    df.attrs["onset_epoch"] = onset
    return df
