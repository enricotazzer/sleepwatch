"""Model inputs: night order, evaluation nights, label shifts and personal features.

Personal features describe a person from their *first N nights* only. Two kinds:

- ``prior_norm`` (label-free, usable in real life): the current epoch relative to the person's
  signal distribution on those nights, after the estimated sleep onset (no labels needed).
- ``prior_stage`` (uses the expert labels of those nights; an upper bound, since a real user has
  no EEG labels): the current epoch relative to the person's typical value *in each stage*, plus
  their usual share of each stage.

Values are filled only for nights after the first N; earlier nights get NaN and are not used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sleepwatch.constants import STAGE_NAMES, UNKNOWN
from sleepwatch.data.label_timing import wake_separation_by_lag

NIGHT_KEYS = ["subject", "night"]
EVAL_FIRST_NIGHT = 4  # the N-nights curve is scored on nights 4+ of subjects with >= 4 nights
MOTION_FLOOR = 1e-5

# Base signals for personal features; True = log10-transform first (heavy-tailed motion).
PRIOR_NORM_BASE = {
    "hr_mean": False,
    "hr_std": False,
    "activity": True,
    "enmo_mean": True,
    "angle_change": True,
}
PRIOR_STAGE_BASE = {"hr_mean": False, "activity": True}
MIN_SCALE = {"hr_mean": 1.0, "hr_std": 0.5}  # floor for the robust scale (others: 0.05 log units)


def prepare(epochs: pd.DataFrame) -> pd.DataFrame:
    """Sort by subject/night/epoch and add ``night_rank`` (1 = first night) and ``eval_night``."""
    out = epochs.sort_values([*NIGHT_KEYS, "epoch"]).reset_index(drop=True)
    out["night_rank"] = out.groupby("subject")["night"].rank(method="dense").astype(int)
    nights_per_subject = out.groupby("subject")["night_rank"].transform("max")
    out["eval_night"] = (nights_per_subject >= EVAL_FIRST_NIGHT) & (
        out["night_rank"] >= EVAL_FIRST_NIGHT
    )
    return out


def shifted(epochs: pd.DataFrame, column: str, lag: int) -> pd.Series:
    """Labels re-paired so that signal epoch ``k`` gets the label from epoch ``k - lag``.

    ``lag`` follows :mod:`sleepwatch.data.label_timing`: a best lag of -3 means labels run 3
    epochs late, so epoch ``k`` takes the label recorded at ``k + 3``. Epochs without a partner
    become Unknown. ``epochs`` must be sorted (see :func:`prepare`).
    """
    if lag == 0:
        return epochs[column].copy()
    return (
        epochs.groupby(NIGHT_KEYS)[column].shift(lag).fillna(UNKNOWN).astype(epochs[column].dtype)
    )


def estimate_label_lag(epochs: pd.DataFrame, label: str = "expert", lags=range(-8, 5)) -> int:
    """Lag at which the mean Wake-sleep activity gap peaks (call with training subjects only)."""
    data = epochs.assign(_log_activity=np.log10(epochs["activity"].clip(lower=MOTION_FLOOR)))
    curve = wake_separation_by_lag(data, label, "_log_activity", lags)
    return int(curve.groupby("lag")["separation"].mean().idxmax())


def _transformed(epochs: pd.DataFrame, base: dict[str, bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: np.log10(epochs[name].clip(lower=MOTION_FLOOR)) if log else epochs[name]
            for name, log in base.items()
        },
        index=epochs.index,
    )


def _level(table: pd.Series | pd.DataFrame, stage: int) -> pd.Series | pd.DataFrame:
    """Rows of a (subject, stage)-indexed table for one stage, indexed by subject."""
    if stage in table.index.get_level_values(1):
        return table.xs(stage, level=1)
    return table.iloc[0:0].droplevel(1)


def personal_features(
    epochs: pd.DataFrame, n_prior: int, kind: str, label: str = "expert"
) -> pd.DataFrame:
    """Personal features from each subject's first ``n_prior`` nights (``epochs`` from prepare)."""
    if n_prior == 0:
        return pd.DataFrame(index=epochs.index)
    prior = epochs["night_rank"] <= n_prior
    target = ~prior
    if kind == "prior_norm":
        values = _transformed(epochs, PRIOR_NORM_BASE)
        asleep = epochs["hours_since_onset"].fillna(0) >= 0  # label-free: after estimated onset
        ref = values[prior & asleep].groupby(epochs.loc[prior & asleep, "subject"])
        median, q75, q25 = ref.median(), ref.quantile(0.75), ref.quantile(0.25)
        out = {}
        for name in PRIOR_NORM_BASE:
            med = epochs["subject"].map(median[name])
            scale = (
                epochs["subject"].map(q75[name] - q25[name]).clip(lower=MIN_SCALE.get(name, 0.05))
            )
            out[f"p_{name}_diff"] = values[name] - med
            out[f"p_{name}_z"] = (values[name] - med) / scale
        out["p_hr_mean_level"] = epochs["subject"].map(median["hr_mean"])
        out["p_activity_level"] = epochs["subject"].map(median["activity"])
    elif kind == "prior_stage":
        values = _transformed(epochs, PRIOR_STAGE_BASE)
        labels = epochs[label]
        known = prior & (labels != UNKNOWN)
        keys = [epochs.loc[known, "subject"], labels[known]]
        stage_median = values[known].groupby(keys).median()
        out = {}
        for stage in range(5):
            name = STAGE_NAMES[stage]
            per_subject = _level(stage_median, stage)
            for base in PRIOR_STAGE_BASE:
                out[f"q_{base}_vs_{name}"] = values[base] - epochs["subject"].map(per_subject[base])
        share = labels[known].groupby(epochs.loc[known, "subject"]).value_counts(normalize=True)
        for stage in range(5):
            per_subject = _level(share, stage)
            out[f"q_share_{STAGE_NAMES[stage]}"] = epochs["subject"].map(per_subject).fillna(0.0)
    else:
        raise ValueError(f"unknown personal feature kind {kind!r}")
    features = pd.DataFrame(out, index=epochs.index)
    features.loc[~target] = np.nan
    return features
