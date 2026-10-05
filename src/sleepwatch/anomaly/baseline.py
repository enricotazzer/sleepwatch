"""Expected heart rate and movement per person, sleep stage and time of night.

For signal ``x`` of person ``p`` in stage ``s`` at time-of-night bin ``t``::

    expected = mu_pop(s, t) + b_p + b_ps(s)

``mu_pop`` is the population mean, estimated on training subjects. ``b_p`` (the person's overall
offset) and ``b_ps`` (their extra offset in stage ``s``) are sums of per-night mean residuals
over the person's earlier nights divided by ``N + kappa``, i.e. their average shrunk toward 0.
Working with one value per night keeps hundreds of autocorrelated epochs from counting as
hundreds of observations. ``kappa`` is the ratio of between-night to between-person variance,
estimated on training subjects by method of moments: with few earlier nights a person looks like
the population, with many they look like themselves. ``z = (x - expected) / sd(s)``, with
``sd(s)`` the within-night SD of the stage's residuals.

Stages are integer codes 0-4 (NaN = unknown); a stage-free detector passes 0 for every epoch.
Signals: ``hr`` (epoch mean heart rate, bpm) and ``act`` (log10 activity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SIGNALS = ("hr", "act")
TIME_EDGES = (0.0, 1.5, 3.0, 4.5, 6.0)  # hours since the estimated onset; bin 0 = before onset
MIN_CELL = 50  # epochs a (stage, time) cell needs for its own mean; else the stage's mean
MIN_STAGE_EPOCHS = 20  # epochs of a stage a night needs to inform that stage's offset
MOTION_FLOOR = 1e-5
KAPPA_FLOOR = 1e-6  # keeps "no earlier nights" at exactly the population value
NIGHT_KEYS = ["subject", "night"]


def detector_inputs(epochs: pd.DataFrame, stage) -> pd.DataFrame:
    """The columns the detector uses, one row per epoch, with the given stage per epoch."""
    hours = epochs["hours_since_onset"].to_numpy(dtype=float)
    # No onset found: fall back to time since the recording started.
    hours = np.where(np.isnan(hours), epochs["hours_since_start"].to_numpy(dtype=float), hours)
    return pd.DataFrame(
        {
            "subject": epochs["subject"].to_numpy(),
            "night": epochs["night"].to_numpy(),
            "epoch": epochs["epoch"].to_numpy(),
            "stage": np.asarray(stage, dtype=float),
            "tbin": np.digitize(hours, TIME_EDGES),
            "hours_since_onset": epochs["hours_since_onset"].to_numpy(dtype=float),
            "hr": epochs["hr_mean"].to_numpy(dtype=float),
            "act": np.log10(np.clip(epochs["activity"].to_numpy(dtype=float), MOTION_FLOOR, None)),
        }
    )


def _kappa(values: pd.Series, groups: pd.Series) -> float:
    """Between-night / between-group variance ratio of ``values`` grouped by ``groups``."""
    frame = pd.DataFrame({"v": values.to_numpy(), "g": groups.to_numpy()}).dropna()
    sizes = frame.groupby("g")["v"].size()
    centred = frame["v"] - frame.groupby("g")["v"].transform("mean")
    dof = (sizes - 1).clip(lower=0).sum()
    within = float((centred**2).sum() / dof) if dof > 0 else np.nan
    means = frame.groupby("g")["v"].mean()
    between = float(means.var(ddof=1) - within * (1 / sizes).mean()) if len(means) > 1 else np.nan
    if not np.isfinite(within) or not np.isfinite(between):
        return 1.0
    if within <= 0:  # nights agree perfectly: no shrinkage needed
        return KAPPA_FLOOR
    return max(within / max(between, within * 1e-3), KAPPA_FLOOR)


@dataclass
class Prior:
    cell_mean: dict[str, pd.Series]  # (stage, tbin) -> mean, cells with enough epochs only
    stage_mean: dict[str, pd.Series]
    global_mean: dict[str, float]
    stage_sd: dict[str, pd.Series]
    kappa: dict[str, float]
    kappa_stage: dict[str, float]

    def population(self, rows: pd.DataFrame, signal: str) -> np.ndarray:
        """``mu_pop`` for each row (NaN where the stage is unknown)."""
        keys = pd.MultiIndex.from_arrays([rows["stage"], rows["tbin"]])
        value = self.cell_mean[signal].reindex(keys).to_numpy()
        stage_value = self.stage_mean[signal].reindex(rows["stage"]).to_numpy()
        value = np.where(np.isnan(value), stage_value, value)
        return np.where(rows["stage"].isna(), np.nan, value)

    def sd(self, rows: pd.DataFrame, signal: str) -> np.ndarray:
        return self.stage_sd[signal].reindex(rows["stage"]).to_numpy()


def night_summaries(rows: pd.DataFrame, prior: Prior) -> pd.DataFrame:
    """Per night: mean residual ``m_<signal>`` and per-stage deviation ``d_<signal>_<stage>``.

    These are the only things a night contributes to a person's baseline.
    """
    frame = rows[[*NIGHT_KEYS, "stage"]].copy()
    for signal in SIGNALS:
        frame[signal] = rows[signal].to_numpy() - prior.population(rows, signal)
    frame = frame.dropna(subset=["stage"])
    out = frame.groupby(NIGHT_KEYS)[list(SIGNALS)].mean().add_prefix("m_")
    by_stage = frame.groupby([*NIGHT_KEYS, "stage"])
    counts = by_stage[list(SIGNALS)].count()
    means = by_stage[list(SIGNALS)].mean().where(counts >= MIN_STAGE_EPOCHS)
    for signal in SIGNALS:
        table = means[signal].unstack("stage")
        for stage in table.columns:
            out[f"d_{signal}_{int(stage)}"] = table[stage] - out[f"m_{signal}"]
    return out


def fit_prior(rows: pd.DataFrame) -> Prior:
    """Population means, spreads and shrinkage strengths from (training subjects') epochs."""
    rows = rows.dropna(subset=["stage"])
    cell_mean, stage_mean, global_mean = {}, {}, {}
    for signal in SIGNALS:
        valid = rows.dropna(subset=[signal])
        cells = valid.groupby(["stage", "tbin"])[signal]
        cell_mean[signal] = cells.mean()[cells.count() >= MIN_CELL]
        stage_mean[signal] = valid.groupby("stage")[signal].mean()
        global_mean[signal] = float(valid[signal].mean())
    prior = Prior(cell_mean, stage_mean, global_mean, {}, {}, {})
    residual = {
        signal: rows[signal].to_numpy() - prior.population(rows, signal) for signal in SIGNALS
    }
    stage_sd, kappa, kappa_stage = {}, {}, {}
    summaries = night_summaries(rows, prior).reset_index()
    for signal in SIGNALS:
        night_mean = (
            pd.Series(residual[signal], index=rows.index)
            .groupby([rows["subject"], rows["night"]])
            .transform("mean")
        )
        within_night = pd.Series(residual[signal] - night_mean.to_numpy(), index=rows.index)
        stage_sd[signal] = within_night.groupby(rows["stage"]).std()
        kappa[signal] = _kappa(summaries[f"m_{signal}"], summaries["subject"])
        stacked = summaries.melt(
            id_vars=["subject"],
            value_vars=[c for c in summaries.columns if c.startswith(f"d_{signal}_")],
        )
        kappa_stage[signal] = _kappa(
            stacked["value"], stacked["subject"] + "|" + stacked["variable"]
        )
    prior.stage_sd, prior.kappa, prior.kappa_stage = stage_sd, kappa, kappa_stage
    return prior


def personal_offsets(baseline: pd.DataFrame, prior: Prior) -> dict[str, dict]:
    """Shrunk offsets ``b`` and ``b_stage`` from the night summaries of a person's earlier nights.

    An empty ``baseline`` gives zeros (the population model).
    """
    out = {}
    for signal in SIGNALS:
        values = baseline[f"m_{signal}"].dropna() if len(baseline) else pd.Series(dtype=float)
        b = float(values.sum() / (len(values) + prior.kappa[signal]))
        per_stage = {}
        for stage in range(5):
            column = f"d_{signal}_{stage}"
            vals = baseline[column].dropna() if column in baseline else pd.Series(dtype=float)
            per_stage[stage] = float(vals.sum() / (len(vals) + prior.kappa_stage[signal]))
        out[signal] = {"b": b, "stage": per_stage}
    return out


def score_epochs(rows: pd.DataFrame, prior: Prior, offsets: dict | None) -> pd.DataFrame:
    """Expected value and z-score per epoch and signal; ``offsets=None`` = population only."""
    out = {}
    stage = rows["stage"].to_numpy()
    known = ~np.isnan(stage)
    index = np.where(known, stage, 0).astype(int)
    for signal in SIGNALS:
        expected = prior.population(rows, signal)
        if offsets is not None:
            per_stage = np.array([offsets[signal]["stage"][k] for k in range(5)])
            expected = expected + offsets[signal]["b"] + np.where(known, per_stage[index], 0.0)
        out[f"expected_{signal}"] = expected
        out[f"z_{signal}"] = (rows[signal].to_numpy() - expected) / prior.sd(rows, signal)
    return pd.DataFrame(out, index=rows.index)
