"""Phase 3: baselines, night channels, calibration, injections and the experiment runner."""

import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from synth import REC_START, make_night

from sleepwatch.anomaly import baseline as bl
from sleepwatch.anomaly import scores as sc
from sleepwatch.anomaly.experiment import (
    AnomalyConfig,
    _inputs,
    _score,
    compare_runs,
    night_table,
    rescore,
    run,
    save_run,
)
from sleepwatch.anomaly.inject import (
    InjectionSettings,
    delay_onset,
    elevate_hr,
    expert_sleep_period,
    fragment,
    versions,
    wake_sources,
)
from sleepwatch.anomaly.screen import ScreenSettings, describe, note, suspect_epochs, suspect_night
from sleepwatch.config import PROJECT_ROOT
from sleepwatch.constants import EPOCH_S, N2, N3, REM, WAKE
from sleepwatch.features.epoch_features import FeatureConfig, night_features
from sleepwatch.models.data import prepare
from sleepwatch.models.gru import GRUConfig
from sleepwatch.models.splits import make_folds

FEATURES = FeatureConfig.from_yaml(PROJECT_ROOT / "configs/features/v1.yaml")
HR_BY_STAGE = {WAKE: 75.0, N2: 60.0, N3: 56.0, REM: 64.0}


def hypnogram(n=240):
    stage = np.full(n, N2)
    stage[:12] = WAKE
    stage[40:70] = N3
    stage[110:140] = REM
    stage[150:153] = WAKE
    stage[200:230] = REM
    return stage


def staged_night(subject="S00", night=1, n=240, hr_offset=0.0, seed=0):
    """A night whose heart rate follows its stages and whose wrist moves only while awake."""
    stage = hypnogram(n)
    rng = np.random.default_rng(seed)
    night_shift = rng.normal(0, 0.5)
    rec_start = REC_START + 86400 * night

    def epoch_of(t):
        return np.clip(((t - rec_start) // EPOCH_S).astype(int), 0, n - 1)

    def hr_fn(t):
        base = np.vectorize(HR_BY_STAGE.get)(stage[epoch_of(t)])
        return base + hr_offset + night_shift + 0.8 * np.sin(t / 37.0)

    def motion_fn(t):
        awake = stage[epoch_of(t)] == WAKE
        x = np.where(awake, 0.8 * np.sin(2 * np.pi * 0.7 * t), 0.0)
        return x, np.zeros_like(t), np.where(awake, np.cos(2 * np.pi * 0.7 * t), 1.0)

    base = make_night(n, hr_fn=hr_fn, motion_fn=motion_fn, expert=stage, rec_start=rec_start)
    return replace(base, subject=subject, night=night)


# --- baseline ----------------------------------------------------------------------------------


def inputs_table(subjects=6, nights=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(subjects):
        offset = rng.normal(0, 5)
        for night in range(1, nights + 1):
            shift = rng.normal(0, 1)
            stage = np.repeat([WAKE, N2, N3, REM, N2], 40).astype(float)
            hr = np.vectorize(HR_BY_STAGE.get)(stage.astype(int)) + offset + shift
            rows.append(
                pd.DataFrame(
                    {
                        "subject": f"S{s}",
                        "night": night,
                        "epoch": np.arange(len(stage)),
                        "stage": stage,
                        "tbin": np.repeat([0, 1, 2, 3, 4], 40),
                        "hours_since_onset": np.linspace(-0.5, 4, len(stage)),
                        "hr": hr + rng.normal(0, 1, len(stage)),
                        "act": rng.normal(-3, 0.3, len(stage)),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_prior_recovers_population_means_and_shrinkage_limits():
    rows = inputs_table()
    prior = bl.fit_prior(rows)
    assert prior.stage_mean["hr"][N3] < prior.stage_mean["hr"][N2] < prior.stage_mean["hr"][WAKE]
    assert prior.kappa["hr"] > 0
    empty = bl.personal_offsets(bl.night_summaries(rows.iloc[:0], prior), prior)
    assert empty["hr"]["b"] == 0 and all(v == 0 for v in empty["hr"]["stage"].values())
    # A person 10 bpm above everyone: few nights pull partway, many nights nearly all the way.
    person = rows[rows["subject"] == "S0"]
    many = pd.concat([person.assign(night=i, hr=person["hr"] + 10) for i in range(200)])
    summaries = bl.night_summaries(many, prior)
    few = bl.personal_offsets(summaries.iloc[:1], prior)["hr"]["b"]
    lots = bl.personal_offsets(summaries, prior)["hr"]["b"]
    single = summaries["m_hr"].mean()
    assert abs(few) < abs(lots) and abs(lots - single) < 0.05 * abs(single)


def test_scores_use_the_expected_value():
    rows = inputs_table()
    prior = bl.fit_prior(rows)
    one = rows[(rows["subject"] == "S1") & (rows["night"] == 5)]
    population = bl.score_epochs(one, prior, None)
    offsets = {
        "hr": {"b": 4.0, "stage": dict.fromkeys(range(5), 0.0)},
        "act": {"b": 0.0, "stage": dict.fromkeys(range(5), 0.0)},
    }
    personal = bl.score_epochs(one, prior, offsets)
    assert np.allclose(personal["expected_hr"], population["expected_hr"] + 4.0)


# --- night channels and calibration ------------------------------------------------------------


def test_runs_and_sleep_period():
    assert sc.runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    stage = np.array([WAKE] * 5 + [N2] * 2 + [WAKE] + [N2] * 30 + [WAKE] * 3, float)
    assert sc.sleep_period(stage, None, staged=True, persistent=20) == (8, 38)
    hours = (np.arange(10) - 4) / 120
    assert sc.sleep_period(np.zeros(10), hours, staged=False, persistent=20) == (4, 10)


def test_night_channels_known_answers():
    n = 200
    stage = np.full(n, float(N2))
    stage[:10] = WAKE
    stage[[80, 81, 150, 151]] = WAKE  # two 1-min wake bouts
    z = np.zeros(n)
    z[100:160] = 2.0  # one 30-min window two SDs high
    night = pd.DataFrame(
        {
            "stage": stage,
            "z_hr": z,
            "z_act_pop": np.zeros(n),
            "hours_since_onset": (np.arange(n) - 10) / 120,
        }
    )
    ch = sc.night_channels(night, staged=True, settings=sc.ChannelSettings())
    assert ch["window"] == (100, 160) and ch["hr_window"] == pytest.approx(2.0)
    assert ch["onset"] == 5.0 and ch["sleep_onset"] == 10
    assert ch["frag"] == pytest.approx(2 / ((n - 10) / 120))
    assert ch["bouts"] == [(80, 82), (150, 152)]


def test_night_z_shrinks_toward_the_population():
    prior = sc.NightPrior(mean={"frag": 1.0}, sd={"frag": 0.5}, kappa={"frag": 2.0})
    nothing = sc.night_z(2.0, pd.Series(dtype=float), prior, "frag", personal=True)
    assert nothing == pytest.approx((2.0 - 1.0) / 0.5)
    usual = pd.Series([2.0] * 200)  # this person always has 2 bouts per hour
    assert sc.night_z(2.0, usual, prior, "frag", personal=True) == pytest.approx(0, abs=0.02)
    assert sc.night_z(2.0, usual, prior, "frag", personal=False) == pytest.approx(2.0)


def test_calibration_keeps_null_false_alarms_at_target():
    rng = np.random.default_rng(0)
    null = pd.DataFrame(rng.normal(size=(200, 4)), columns=list(sc.CHANNELS))
    calibration = sc.calibrate(null, 0.05)
    assert calibration.null_rate <= 0.05
    test = pd.DataFrame(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 10.0, 0.0], [np.nan] * 4], columns=list(sc.CHANNELS)
    )
    out = sc.evaluate(test, calibration)
    assert out["flagged"].tolist() == [False, True, False]
    assert out.loc[1, "top_channel"] == "frag"
    assert sc.tail_p(np.array([1.0, 2.0, 3.0]), np.array([2.5, np.nan])).tolist() == [0.5, 1.0]


# --- injections --------------------------------------------------------------------------------


def test_hr_elevation_changes_only_its_window():
    night = staged_night()
    t = night.hr["t"].to_numpy()
    out = elevate_hr(night, 60, 120, 6.0, ramp_s=120.0)
    diff = out.hr["hr"].to_numpy() - night.hr["hr"].to_numpy()
    t0, t1 = REC_START + 86400 + 60 * EPOCH_S, REC_START + 86400 + 120 * EPOCH_S
    assert np.allclose(diff[(t < t0) | (t >= t1)], 0)
    assert np.allclose(diff[(t >= t0 + 120) & (t <= t1 - 120)], 6.0)
    assert ((diff >= 0) & (diff <= 6.0 + 1e-9)).all()
    assert night.hr["hr"].equals(staged_night().hr["hr"])  # input untouched


def test_fragmentation_copies_wake_into_sleep_only_inside_its_windows():
    night = staged_night()
    features = night_features(night, FEATURES)
    settings = InjectionSettings()
    period = expert_sleep_period(night.expert, settings.persistent_epochs)
    sources = wake_sources(features, settings)
    assert sources == [(0, 12)]
    out, windows = fragment(night, 2, period, sources, settings, np.random.default_rng(0))
    assert len(windows) == 2 and windows[1][0] - windows[0][0] >= settings.frag_min_gap_epochs
    changed = np.flatnonzero(out.expert != night.expert)
    inside = np.concatenate([np.arange(a, b) for a, b in windows])
    assert set(changed) <= set(inside) and (out.expert[inside] == WAKE).all()
    a, b = windows[0]
    t0, t1 = night.rec_start + a * EPOCH_S, night.rec_start + b * EPOCH_S
    copied = out.hr[(out.hr["t"] >= t0) & (out.hr["t"] < t1)]["hr"].to_numpy()
    source = night.hr[night.hr["t"] < night.rec_start + 6 * EPOCH_S]["hr"].to_numpy()
    assert np.allclose(copied, source)
    keep = (night.hr["t"] < night.rec_start + windows[0][0] * EPOCH_S).to_numpy()
    assert np.allclose(out.hr["hr"].to_numpy()[keep], night.hr["hr"].to_numpy()[keep])


def test_delayed_onset_and_all_versions():
    night = staged_night()
    features = night_features(night, FEATURES)
    settings = InjectionSettings()
    period = expert_sleep_period(night.expert, settings.persistent_epochs)
    out, windows = delay_onset(night, 20, period, wake_sources(features, settings))
    assert windows == [(12, 52)] and (out.expert[12:52] == WAKE).all()
    assert (out.expert[52:] == night.expert[52:]).all()
    made = versions(night, features, settings, seed=1)
    kinds = [v[1] for v in made]
    assert kinds.count("hr") == 9 and kinds.count("onset") == 3
    assert 1 <= kinds.count("frag") <= 3  # 8 awakenings 15 min apart don't fit in 2 h
    no_wake = features.assign(qc_motion_coverage=0.0)
    assert [v[1] for v in versions(night, no_wake, settings, seed=1)] == ["hr"] * 9


# --- runner ------------------------------------------------------------------------------------


def synthetic_dataset(subjects=10, nights=4):
    store, frames = {}, []
    for s in range(subjects):
        for night in range(1, nights + 1):
            item = staged_night(f"S{s:02d}", night, hr_offset=1.5 * s, seed=100 * s + night)
            store[(item.subject, night)] = item
            frames.append(night_features(item, FEATURES))
    return store, pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def dataset():
    return synthetic_dataset()


def test_night_table_excludes_and_counts_earlier_usable_nights(dataset):
    _, epochs = dataset
    cfg = AnomalyConfig(name="t", exclude=[("S00", 2)])
    nights = night_table(prepare(epochs), cfg).set_index(["subject", "night"])
    assert not nights.loc[("S00", 2), "usable"]
    assert nights.loc[("S00", 3), "n_baseline"] == 1 and not nights.loc[("S00", 3), "scored"]
    assert nights.loc[("S00", 4), "n_baseline"] == 2 and nights.loc[("S00", 4), "scored"]
    assert nights.loc[("S01", 3), "scored"]


def test_scores_depend_only_on_earlier_nights(dataset):
    _, epochs = dataset
    data = prepare(epochs[epochs["subject"].isin(["S00", "S01", "S02"])])
    rows = bl.detector_inputs(data, data["expert"].astype(float)).assign(version="clean")
    nights = night_table(data, AnomalyConfig(name="t", exclude=[]))
    prior = bl.fit_prior(rows)
    night_prior = sc.fit_night_prior(
        pd.DataFrame(
            {
                "subject": ["S00", "S00", "S01", "S01"],
                "frag": [1, 2, 1, 3.0],
                "onset": [5, 6, 5, 9.0],
            }
        )
    )
    settings = sc.ChannelSettings()
    target = rows[(rows["subject"] == "S00") & (rows["night"] == 3)]

    def score(history):
        out = _score(target, history, nights, prior, night_prior, True, settings)
        return out[out["baseline"] == "personal"][["hr_window", "hr_night", "frag", "onset"]]

    before = score(rows)
    later = rows.copy()
    later.loc[(later["subject"] == "S00") & (later["night"] == 4), "hr"] += 30
    pd.testing.assert_frame_equal(before, score(later))
    earlier = rows.copy()
    earlier.loc[(earlier["subject"] == "S00") & (earlier["night"] == 1), "hr"] += 30
    assert not before.equals(score(earlier))


@pytest.fixture(scope="module")
def smoke(dataset, tmp_path_factory):
    """One small saved run (fold 0 only), shared by the runner and rescoring tests."""
    store, epochs = dataset
    nights = [tuple(k) for k in epochs[["subject", "night"]].drop_duplicates().to_numpy()]
    folds = make_folds(nights, 5, seed=0)
    cfg = AnomalyConfig(
        name="smoke",
        folds=[0],
        inner_folds=2,
        n_boot=20,
        exclude=[("S03", 2)],
        gru=GRUConfig(hidden=8, lr=3e-2, max_epochs=3, patience=3),
    )
    output = run(cfg, epochs, folds, lambda s, n: store[(s, n)], FEATURES, log=lambda *_: None)
    quality = pd.DataFrame(
        {
            "subject": [k[0] for k in store],
            "night": [k[1] for k in store],
            "rec_start_local": "2022-01-10 23:00:00",
        }
    )
    results_dir = tmp_path_factory.mktemp("results")
    run_dir = save_run(cfg, output, quality, results_dir, extra={})
    return {"cfg": cfg, "folds": folds, "output": output, "run_dir": run_dir, "dir": results_dir}


def test_runner_end_to_end(smoke):
    cfg, folds, output, run_dir = smoke["cfg"], smoke["folds"], smoke["output"], smoke["run_dir"]
    checks = output["checks"]
    assert checks["features_equal"].all() and checks["stages_equal"].all()
    results = output["results"]
    test_subjects = {s for s, f in folds.items() if f == 0}
    assert set(results["subject"]) <= test_subjects
    assert not set(output["nulls"]["subject"]) & test_subjects  # thresholds from other people
    assert (results["null_rate"] <= cfg.target_fpr).all()
    assert set(results["source"]) == {"gru", "expert", "none"}
    assert set(results["baseline"]) == {"personal", "population"}
    assert (results.groupby(["source", "baseline", "version"]).size() == 4).all()  # 2 x 2 nights
    metrics = output["metrics"]
    assert "hr+10_sleep" in metrics["variants"]["gru/personal"]["conditions"]
    for name in [
        "night_scores.parquet",
        "injections.parquet",
        "metrics.json",
        "flagged_nights.json",
    ]:
        assert (run_dir / name).is_file()
    assert pd.read_csv(smoke["dir"] / "anomaly_index.csv").loc[0, "partial"]


# --- Phase 3b: screen and rescoring -------------------------------------------------------------


def test_screen_marks_only_sustained_plateaus_far_above_the_night():
    settings = ScreenSettings()
    hr = np.full(300, 60.0)
    hr[50:60] = 101.0  # 10 epochs at +41: marked
    hr[100:109] = 130.0  # 9 epochs: too short
    hr[150:180] = 99.0  # +39: not far enough
    mask = suspect_night(hr, settings)
    assert mask[50:60].all() and mask.sum() == 10
    np.testing.assert_array_equal(suspect_night(hr + 10, settings), mask)  # whole-night offset
    gap = hr.copy()
    gap[55] = np.nan  # a missing epoch breaks the run into 5 + 4
    assert not suspect_night(gap, settings).any()
    assert not suspect_night(np.full(20, np.nan), settings).any()
    found = describe(hr, settings)
    assert found == {"minutes": 5.0, "peak_bpm": 101.0}
    assert "5 min at up to 101 bpm" in note(found)


def test_screen_works_per_night_version_and_masks_only_heart_rate():
    night = night_features(staged_night("S00", 3), FEATURES)
    hr = night["hr_mean"].to_numpy().copy()
    hr[100:120] = np.nanmedian(hr) + 60
    rows = pd.concat(
        [night.assign(version="clean"), night.assign(version="hr+60", hr_mean=hr)],
        ignore_index=True,
    ).sample(frac=1, random_state=0)
    suspect = suspect_epochs(rows, ScreenSettings())
    marked = rows[suspect]
    assert set(marked["version"]) == {"hr+60"}
    assert sorted(marked["epoch"]) == list(range(100, 120))
    plain = _inputs(rows, "expert")
    screened = _inputs(rows, "expert", screen=ScreenSettings())
    assert screened.loc[suspect, "hr"].isna().all()
    pd.testing.assert_frame_equal(screened[~suspect], plain[~suspect])
    pd.testing.assert_frame_equal(screened.drop(columns="hr"), plain.drop(columns="hr"))


def as_complete_base(smoke, tmp_path):
    """A copy of the smoke run marked complete and clean (the guard is tested separately)."""
    base = tmp_path / "base"
    shutil.copytree(smoke["run_dir"], base)
    info = json.loads((base / "run.json").read_text())
    info["partial"], info["git"]["dirty"] = False, False
    (base / "run.json").write_text(json.dumps(info))
    return base


def test_rescore_without_a_screen_reproduces_the_run(dataset, smoke, tmp_path):
    _, epochs = dataset
    base = as_complete_base(smoke, tmp_path)
    cfg = smoke["cfg"].model_copy(update={"name": "again"})
    again = rescore(cfg, base, epochs, log=lambda *_: None)
    pd.testing.assert_frame_equal(
        again["results"], smoke["output"]["results"], check_dtype=False, check_exact=True
    )
    comparison = compare_runs(smoke["output"]["results"], again["results"], cfg)
    for entry in comparison["variants"].values():
        assert all(c["difference"] == 0 and c["ci95"] == [0, 0] for c in entry.values())
    screened = rescore(
        cfg.model_copy(update={"screen": ScreenSettings()}), base, epochs, log=lambda *_: None
    )
    table = screened["screened"]  # synthetic nights have no plateaus: nothing marked
    assert (table["suspect_epochs"] == 0).all() and table["outside_matches_clean"].all()
    pd.testing.assert_frame_equal(screened["results"], again["results"], check_exact=True)


def test_rescore_refuses_partial_bases_and_other_settings(dataset, smoke, tmp_path):
    _, epochs = dataset
    with pytest.raises(ValueError, match="complete"):
        rescore(smoke["cfg"], smoke["run_dir"], epochs)
    base = as_complete_base(smoke, tmp_path)
    with pytest.raises(ValueError, match="target_fpr"):
        rescore(smoke["cfg"].model_copy(update={"target_fpr": 0.1}), base, epochs)


def test_compare_runs_pairs_night_versions():
    rows = pd.DataFrame(
        {
            "source": "gru",
            "baseline": "personal",
            "subject": ["A", "A", "B", "B"],
            "night": [3, 4, 3, 4],
            "version": "clean",
            "flagged": False,
            "fold": 0,
            "threshold": 1 / 120,
            "null_nights": 120,
        }
    )
    new = rows.assign(flagged=[True, False, False, False])
    out = compare_runs(rows, new, AnomalyConfig(name="t", n_boot=200))
    entry = out["variants"]["gru/personal"]["clean"]
    assert entry["difference"] == 0.25 and entry["changed_up"] == 1 and entry["changed_down"] == 0
    assert 0 <= entry["ci95"][0] <= 0.25 <= entry["ci95"][1] <= 0.5
    assert out["edge"]["base"]["gru/personal/0"] == "0 of 120"
    with pytest.raises(ValueError):
        compare_runs(rows, new.iloc[:3], AnomalyConfig(name="t"))
