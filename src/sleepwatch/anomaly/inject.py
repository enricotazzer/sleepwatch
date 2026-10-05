"""Synthetic anomalies injected into a loaded night's raw samples (Phase 3 validation).

Each function returns a new :class:`~sleepwatch.data.loader.Night` (the input is not modified)
plus the injected epoch windows. Features are then rebuilt with the unchanged pipeline, so every
downstream step (features, predicted stages, scores) sees the anomaly as it would see real data.

- :func:`elevate_hr`: heart rate raised by ``delta`` bpm over a window, with linear ramps;
- :func:`fragment`: short awakenings inserted into sleep, their heart-rate and accelerometer
  samples copied from the same night's own wake stretches; the expert labels there become Wake;
- :func:`delay_onset`: the first minutes of sleep replaced by copied wake (labels Wake).

Placement uses the expert labels (generation may; detection never does).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from sleepwatch.anomaly.scores import runs
from sleepwatch.constants import EPOCH_S, UNKNOWN, WAKE
from sleepwatch.data.loader import Night


class InjectionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hr_deltas: tuple[float, ...] = (3.0, 6.0, 10.0)
    hr_durations_min: tuple[int, ...] = (30, 120, 0)  # 0 = the whole sleep period
    ramp_s: float = 120.0
    frag_counts: tuple[int, ...] = (2, 4, 8)
    frag_bout_epochs: int = 6
    frag_min_gap_epochs: int = 30
    onset_delays_min: tuple[int, ...] = (20, 40, 60)
    persistent_epochs: int = 20
    source_min_epochs: int = 6
    source_min_coverage: float = 0.9


def expert_sleep_period(expert: np.ndarray, persistent: int) -> tuple[int, int] | None:
    """``[onset, end)`` from the expert labels: first run of ``persistent`` sleep epochs to the
    last sleep epoch."""
    asleep = (expert != WAKE) & (expert != UNKNOWN)
    long_runs = [r for r in runs(asleep) if r[1] - r[0] >= persistent]
    if not long_runs:
        return None
    return long_runs[0][0], int(np.flatnonzero(asleep)[-1]) + 1


def wake_sources(features: pd.DataFrame, settings: InjectionSettings) -> list[tuple[int, int]]:
    """Expert-Wake runs with good heart-rate and motion coverage to copy wake signals from."""
    good = (
        (features["expert"].to_numpy() == WAKE)
        & (features["qc_hr_coverage"].to_numpy() >= settings.source_min_coverage)
        & (features["qc_motion_coverage"].to_numpy() >= settings.source_min_coverage)
    )
    return [r for r in runs(good) if r[1] - r[0] >= settings.source_min_epochs]


def _t(night: Night, epoch: int) -> float:
    return night.rec_start + epoch * EPOCH_S


def elevate_hr(night: Night, start: int, end: int, delta: float, ramp_s: float) -> Night:
    """Heart rate + ``delta`` bpm over epochs ``[start, end)``, ramping in and out."""
    t = night.hr["t"].to_numpy()
    t0, t1 = _t(night, start), _t(night, end)
    weight = np.clip((t - t0) / ramp_s, 0, 1) * np.clip((t1 - t) / ramp_s, 0, 1)
    hr = night.hr.assign(hr=night.hr["hr"].to_numpy() + delta * weight)
    return replace(night, hr=hr)


def _copy_wake(night: Night, start: int, n_epochs: int, sources: list[tuple[int, int]]) -> Night:
    """Replace epochs ``[start, start + n_epochs)`` with samples copied from ``sources`` (used in
    turn, from their beginning, until the window is filled); those labels become Wake."""
    pieces_hr, pieces_motion = [], []
    filled, i = 0, 0
    while filled < n_epochs:
        src_start, src_end = sources[i % len(sources)]
        take = min(src_end - src_start, n_epochs - filled)
        s0, s1 = _t(night, src_start), _t(night, src_start + take)
        shift = _t(night, start + filled) - s0
        for frame, out in ((night.hr, pieces_hr), (night.motion, pieces_motion)):
            piece = frame[(frame["t"] >= s0) & (frame["t"] < s1)]
            out.append(piece.assign(t=piece["t"] + shift))
        filled += take
        i += 1
    d0, d1 = _t(night, start), _t(night, start + n_epochs)

    def splice(frame, pieces):
        keep = frame[(frame["t"] < d0) | (frame["t"] >= d1)]
        return pd.concat([keep, *pieces]).sort_values("t", kind="stable").reset_index(drop=True)

    expert = night.expert.copy()
    expert[start : start + n_epochs] = WAKE
    return replace(
        night,
        hr=splice(night.hr, pieces_hr),
        motion=splice(night.motion, pieces_motion),
        expert=expert,
    )


def fragment(
    night: Night,
    count: int,
    period: tuple[int, int],
    sources: list[tuple[int, int]],
    settings: InjectionSettings,
    rng: np.random.Generator,
) -> tuple[Night, list[tuple[int, int]]] | None:
    """``count`` awakenings placed at random in sleep (``None`` if they don't fit)."""
    bout, gap = settings.frag_bout_epochs, settings.frag_min_gap_epochs
    onset, end = period
    asleep = (night.expert != WAKE) & (night.expert != UNKNOWN)
    candidates = [i for i in range(onset + gap, end - bout - gap) if asleep[i : i + bout].all()]
    chosen: list[int] = []
    for i in rng.permutation(candidates):
        if all(abs(int(i) - j) >= gap for j in chosen):
            chosen.append(int(i))
            if len(chosen) == count:
                break
    if len(chosen) < count:
        return None
    for i in sorted(chosen):
        night = _copy_wake(night, i, bout, sources)
    return night, [(i, i + bout) for i in sorted(chosen)]


def delay_onset(
    night: Night, minutes: int, period: tuple[int, int], sources: list[tuple[int, int]]
) -> tuple[Night, list[tuple[int, int]]]:
    """Sleep onset pushed back by ``minutes``: the first minutes of sleep become copied wake."""
    onset, end = period
    n_epochs = min(minutes * 60 // EPOCH_S, end - onset)
    return _copy_wake(night, onset, n_epochs, sources), [(onset, onset + n_epochs)]


def versions(night: Night, features: pd.DataFrame, settings: InjectionSettings, seed: int):
    """Every injected version of ``night``: ``(version, kind, size, duration, windows, night)``.

    ``features`` are the clean night's epoch features (for coverage and labels). Fragmentation
    and delayed onset are skipped when the night has no wake stretch to copy from.
    """
    rng = np.random.default_rng(seed)
    period = expert_sleep_period(night.expert, settings.persistent_epochs)
    if period is None:
        return []
    onset, end = period
    sources = wake_sources(features, settings)
    out = []
    for delta in settings.hr_deltas:
        for minutes in settings.hr_durations_min:
            length = minutes * 60 // EPOCH_S if minutes else end - onset
            length = min(length, end - onset)
            start = int(rng.integers(onset, end - length + 1))
            injected = elevate_hr(night, start, start + length, delta, settings.ramp_s)
            name = f"hr+{delta:g}_{minutes or 'sleep'}"
            out.append((name, "hr", delta, minutes, [(start, start + length)], injected))
    if sources:
        for count in settings.frag_counts:
            result = fragment(night, count, period, sources, settings, rng)
            if result is not None:
                out.append((f"frag{count}", "frag", count, None, result[1], result[0]))
        for minutes in settings.onset_delays_min:
            injected, windows = delay_onset(night, minutes, period, sources)
            out.append((f"onset+{minutes}", "onset", minutes, None, windows, injected))
    return out
