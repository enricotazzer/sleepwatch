"""Checks of when the labels happen relative to the watch signals.

The dataset documents that label epoch ``k`` covers ``[recStart + 30k, recStart + 30(k+1))``.
These functions test that empirically, without assuming it:

- :func:`wake_separation_by_lag` pairs label epoch ``k`` with signal epoch ``k + lag`` and measures
  how well the signal separates Wake from sleep. Movement marks waking sharply, so the lag with the
  largest separation estimates the label-to-signal offset. A negative best lag means the labels
  run late: they describe signals recorded ``-lag`` epochs earlier.
- :func:`label_agreement_by_lag` does the same between two label sequences (expert vs Dreem).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sleepwatch.constants import UNKNOWN, WAKE

NIGHT_KEYS = ["subject", "night"]


def wake_separation_by_lag(
    epochs: pd.DataFrame, label: str, feature: str, lags: range = range(-4, 5)
) -> pd.DataFrame:
    """Per night and lag: mean ``feature`` in Wake epochs minus mean in sleep epochs."""
    rows = []
    for key, night in epochs.groupby(NIGHT_KEYS):
        labels = night[label].to_numpy()
        wake = (labels != UNKNOWN) & (labels == WAKE)
        sleep = (labels != UNKNOWN) & (labels != WAKE)
        for lag in lags:
            signal = night[feature].shift(-lag).to_numpy()  # value at epoch k + lag
            separation = (
                np.nanmean(signal[wake]) - np.nanmean(signal[sleep])
                if wake.any() and sleep.any()
                else np.nan
            )
            rows.append(
                {**dict(zip(NIGHT_KEYS, key, strict=True)), "lag": lag, "separation": separation}
            )
    return pd.DataFrame(rows)


def label_agreement_by_lag(
    epochs: pd.DataFrame, a: str = "expert", b: str = "dreem", lags: range = range(-4, 5)
) -> pd.DataFrame:
    """Per night and lag: share of epochs where ``a`` at epoch k equals ``b`` at epoch k + lag."""
    rows = []
    for key, night in epochs.groupby(NIGHT_KEYS):
        first = night[a].to_numpy()
        for lag in lags:
            second = night[b].astype(float).shift(-lag).to_numpy()
            valid = (first != UNKNOWN) & ~np.isnan(second) & (second != UNKNOWN)
            agreement = (first[valid] == second[valid]).mean() if valid.any() else np.nan
            rows.append(
                {**dict(zip(NIGHT_KEYS, key, strict=True)), "lag": lag, "agreement": agreement}
            )
    return pd.DataFrame(rows)


def best_lags(by_lag: pd.DataFrame, value: str) -> pd.Series:
    """Lag with the largest ``value`` for each night."""
    idx = by_lag.dropna(subset=[value]).groupby(NIGHT_KEYS)[value].idxmax()
    return by_lag.loc[idx].set_index(NIGHT_KEYS)["lag"]
