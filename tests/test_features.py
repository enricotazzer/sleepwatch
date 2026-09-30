import numpy as np
import pandas as pd
import pytest
from synth import make_night

from sleepwatch.constants import UNKNOWN, WAKE
from sleepwatch.features.epoch_features import (
    META_COLUMNS,
    FeatureConfig,
    feature_columns,
    night_features,
)

CFG = FeatureConfig()


def sine_hr(t):
    return 60 + 8 * np.sin(2 * np.pi * t / 900)


def burst_in_epoch(epoch, amplitude=0.3, freq=1.0):
    """Still wrist except a 1 Hz shake along the gravity axis during one epoch. (A shake
    perpendicular to gravity changes the magnitude only at second order.)"""

    def fn(t):
        rel = t - t[0]
        shake = np.where((rel >= epoch * 30) & (rel < (epoch + 1) * 30), amplitude, 0.0)
        return np.zeros_like(t), np.zeros_like(t), 1 + shake * np.sin(2 * np.pi * freq * rel)

    return fn


def test_constant_signals():
    df = night_features(make_night(), CFG)
    inner = df.iloc[1:-1]
    assert np.allclose(inner["hr_mean"], 60) and np.allclose(inner["hr_std"], 0)
    assert np.allclose(inner["hr_rel"], 0)
    assert np.allclose(df["enmo_mean"], 0, atol=1e-6)
    assert np.allclose(df["activity"], 0, atol=1e-4)
    assert np.allclose(df["angle_mean"], 90) and np.allclose(df["angle_change"].iloc[1:], 0)


def test_hr_features_do_not_depend_on_sampling_rate():
    every_2s = night_features(make_night(hr_fn=sine_hr, hr_dt=2.0), CFG)
    every_5s = night_features(make_night(hr_fn=sine_hr, hr_dt=5.0, hr_phase=1.3), CFG)
    inner = slice(1, -1)
    for col in ["hr_mean", "hr_rel", "hr_detrend"]:
        diff = (every_2s[col] - every_5s[col]).iloc[inner].abs()
        assert diff.max() < 0.3, col
    # The raw sample count does depend on it, which is why it is a qc_ column, not a feature.
    assert every_2s["qc_hr_samples"].median() > 2 * every_5s["qc_hr_samples"].median()


def test_hr_gaps_give_nan_features_not_interpolation():
    night = make_night(hr_fn=sine_hr)
    rel = night.hr["t"] - night.rec_start
    night.hr = night.hr[(rel < 10 * 30) | (rel >= 20 * 30)].reset_index(drop=True)
    df = night_features(night, CFG)
    assert df.loc[11:18, "hr_mean"].isna().all()
    assert (df.loc[11:18, "qc_hr_coverage"] == 0).all()
    assert df.loc[[5, 25], "hr_mean"].notna().all()


@pytest.mark.parametrize("motion_hz", [50.0, 100 / 3])
def test_movement_shows_up_in_its_epoch_at_any_motion_rate(motion_hz):
    df = night_features(make_night(motion_fn=burst_in_epoch(5), motion_hz=motion_hz), CFG)
    others = df["activity"].drop(index=5)
    assert df.loc[5, "activity"] > 0.1
    assert df.loc[5, "activity"] > 100 * others.median()
    assert df.loc[5, "enmo_mean"] > others.max()


def test_activity_is_similar_at_50_and_33_hz():
    a50 = night_features(make_night(motion_fn=burst_in_epoch(5), motion_hz=50.0), CFG)
    a33 = night_features(make_night(motion_fn=burst_in_epoch(5), motion_hz=100 / 3), CFG)
    assert a33.loc[5, "activity"] == pytest.approx(a50.loc[5, "activity"], rel=0.05)


def swing(rel):
    """Arm tilt (rad) swinging 20-50 deg with a 37 s period, so 5 s block means keep changing."""
    return np.radians(35 + 15 * np.sin(2 * np.pi * rel / 37))


def test_sleep_onset_is_estimated_from_stillness():
    def restless_then_still(t):
        rel = t - t[0]
        theta = np.where(rel < 30 * 30, swing(rel), 0.0)
        return np.sin(theta), np.zeros_like(t), np.cos(theta)

    df = night_features(make_night(motion_fn=restless_then_still), CFG)
    # Movement stops at epoch 30, but epoch 30's first block change is measured against the last
    # moving block of epoch 29, so the estimate is one epoch late by construction.
    assert df.attrs["onset_epoch"] == 31
    assert df.loc[31, "hours_since_onset"] == 0
    assert df.loc[0, "hours_since_onset"] == pytest.approx(-31 * 30 / 3600)


def test_no_stillness_means_no_onset():
    def restless(t):
        theta = swing(t - t[0])
        return np.sin(theta), np.zeros_like(t), np.cos(theta)

    df = night_features(make_night(n_epochs=30, motion_fn=restless), CFG)
    assert df.attrs["onset_epoch"] is None
    assert df["hours_since_onset"].isna().all()


def test_time_features():
    df = night_features(make_night(), CFG)
    assert df.loc[0, "hours_since_noon"] == pytest.approx(11.0)  # 23:00 local
    assert df.loc[30, "hours_since_start"] == pytest.approx(0.25)
    assert df.loc[30, "hours_since_noon"] == pytest.approx(11.25)


def test_features_never_depend_on_labels():
    expert = np.full(60, WAKE, np.uint8)
    base = night_features(make_night(hr_fn=sine_hr, expert=expert), CFG)
    shuffled = np.random.default_rng(0).integers(0, 6, 60).astype(np.uint8)
    other = night_features(make_night(hr_fn=sine_hr, expert=shuffled), CFG)
    cols = feature_columns(base)
    pd.testing.assert_frame_equal(base[cols], other[cols])


def test_table_layout():
    expert = np.array([WAKE] * 59 + [UNKNOWN], np.uint8)
    df = night_features(make_night(expert=expert), CFG)
    assert len(df) == 60 and df["epoch"].tolist() == list(range(60))
    assert df["labeled"].tolist() == [True] * 59 + [False]
    cols = feature_columns(df)
    assert not set(cols) & set(META_COLUMNS)
    assert not any(c.startswith("qc_") for c in cols)
    assert "hr_mean_mean_w5" in cols and "activity_std_w10" in cols


def test_config_rejects_blocks_that_do_not_tile_an_epoch():
    with pytest.raises(ValueError, match="divide"):
        FeatureConfig(angle_block_s=7)
