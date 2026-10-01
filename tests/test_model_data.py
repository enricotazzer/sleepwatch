import numpy as np
import pandas as pd
import pytest

from sleepwatch.constants import N2, UNKNOWN, WAKE
from sleepwatch.models.data import (
    estimate_label_lag,
    expanding_features,
    personal_features,
    prepare,
    shifted,
)
from sleepwatch.models.splits import inner_split, make_folds, outer_splits


def epoch_table(subjects=("A", "B"), nights=5, n=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s_i, subject in enumerate(subjects):
        for night in range(1, nights + 1):
            labels = np.where(rng.random(n) < 0.2, WAKE, N2)
            rows.append(
                pd.DataFrame(
                    {
                        "subject": subject,
                        "night": night,
                        "epoch": np.arange(n),
                        "expert": labels,
                        "dreem": labels,
                        "hr_mean": 60 + 5 * s_i + 8 * (labels == WAKE) + rng.normal(0, 1, n),
                        "hr_std": rng.uniform(0.5, 2, n),
                        "activity": np.where(labels == WAKE, 1e-2, 2e-4) * rng.uniform(0.5, 2, n),
                        "enmo_mean": rng.uniform(1e-3, 3e-3, n),
                        "angle_change": rng.uniform(0.01, 0.1, n),
                        "hours_since_onset": (np.arange(n) - 5) * 30 / 3600,
                    }
                )
            )
    return prepare(pd.concat(rows, ignore_index=True))


@pytest.mark.parametrize("kind", ["prior_norm", "prior_stage"])
def test_personal_features_use_only_the_first_n_nights_of_the_same_subject(kind):
    base = epoch_table()
    feats = personal_features(base, 2, kind)
    changed = base.copy()
    later = (changed["subject"] == "A") & (changed["night"] == 3)
    changed.loc[later, ["hr_mean", "activity"]] *= 3  # a later night of A
    changed.loc[later, "expert"] = WAKE
    changed.loc[changed["subject"] == "B", ["hr_mean", "activity"]] *= 3  # another subject
    after = personal_features(changed, 2, kind)
    night4 = (base["subject"] == "A") & (base["night"] == 4)
    pd.testing.assert_frame_equal(feats[night4], after[night4])


def test_prior_norm_never_reads_labels():
    base = epoch_table()
    relabelled = base.assign(expert=np.random.default_rng(1).permutation(base["expert"].to_numpy()))
    pd.testing.assert_frame_equal(
        personal_features(base, 2, "prior_norm"), personal_features(relabelled, 2, "prior_norm")
    )


def test_prior_stage_does_read_labels():  # it is the label-using upper bound by design
    base = epoch_table()
    relabelled = base.assign(expert=np.where(base["expert"] == WAKE, N2, WAKE))
    assert not personal_features(base, 2, "prior_stage").equals(
        personal_features(relabelled, 2, "prior_stage")
    )


def test_features_only_on_nights_after_the_first_n():
    base = epoch_table()
    feats = personal_features(base, 3, "prior_norm")
    assert feats[base["night_rank"] <= 3].isna().all().all()
    assert feats[base["night_rank"] > 3].notna().all().all()
    assert personal_features(base, 0, "prior_norm").shape[1] == 0


def test_prior_norm_values():
    base = epoch_table()
    feats = personal_features(base, 1, "prior_norm")
    first = base[(base["subject"] == "A") & (base["night"] == 1) & (base["hours_since_onset"] >= 0)]
    row = (base["subject"] == "A") & (base["night"] == 2)
    expected = base.loc[row, "hr_mean"] - first["hr_mean"].median()
    np.testing.assert_allclose(feats.loc[row, "p_hr_mean_diff"], expected)


def test_prepare_marks_evaluation_nights():
    base = epoch_table(subjects=("A", "B"), nights=5)
    short = epoch_table(subjects=("C",), nights=3)
    both = prepare(pd.concat([base, short], ignore_index=True).drop(columns=["night_rank"]))
    per_night = both.groupby(["subject", "night"])["eval_night"].first()
    assert per_night.loc["A"].tolist() == [False, False, False, True, True]
    assert not per_night.loc["C"].any()  # only 3 nights: never evaluated in the N-curve


def test_shift_pairs_each_epoch_with_a_later_label():
    base = epoch_table(subjects=("A",), nights=1, n=10)
    moved = shifted(base, "expert", -3)  # labels run 3 epochs late
    assert moved.iloc[:7].tolist() == base["expert"].iloc[3:].tolist()
    assert (moved.iloc[7:] == UNKNOWN).all()


def test_estimated_lag_recovers_a_known_delay():
    base = epoch_table(subjects=("A", "B", "C"), nights=4, n=200)
    late = base.assign(expert=shifted(base, "expert", 2))  # labels now 2 epochs late
    assert estimate_label_lag(base) == 0
    assert estimate_label_lag(late) == -2


def test_folds_are_disjoint_complete_and_reproducible():
    nights = [(f"S{i:02d}", n) for i in range(23) for n in range(1, 3 + i % 5)]
    folds = make_folds(nights, 5, seed=42)
    assert folds == make_folds(nights, 5, seed=42)
    assert sorted(folds) == sorted({s for s, _ in nights}) and set(folds.values()) == set(range(5))
    for train, test in outer_splits(folds):
        assert not set(train) & set(test) and len(train) + len(test) == 23
    fit, val = inner_split(sorted(folds)[:18], 0.2, seed=1)
    assert not set(fit) & set(val) and len(val) == 4


def test_expanding_baseline_uses_only_earlier_nights_of_the_same_subject():
    base = epoch_table()
    feats = expanding_features(base)
    changed = base.copy()
    night3 = (changed["subject"] == "A") & (changed["night"] == 3)
    changed.loc[(changed["subject"] == "A") & (changed["night"] >= 3), "hr_mean"] += 20
    changed.loc[changed["subject"] == "B", ["hr_mean", "activity"]] *= 3
    after = expanding_features(changed)
    early = (base["subject"] == "A") & (base["night"] < 3)
    pd.testing.assert_frame_equal(feats[early], after[early])  # later nights don't matter
    levels = ["p_hr_mean_level", "p_activity_level"]
    pd.testing.assert_frame_equal(feats.loc[night3, levels], after.loc[night3, levels])
    shift = after.loc[night3, "p_hr_mean_diff"] - feats.loc[night3, "p_hr_mean_diff"]
    assert np.allclose(shift, 20)  # the night itself is compared against the unchanged baseline


def test_expanding_baseline_matches_prior_norm_over_the_same_nights():
    base = epoch_table()
    night4 = base["night_rank"] == 4
    pd.testing.assert_frame_equal(
        expanding_features(base)[night4], personal_features(base, 3, "prior_norm")[night4]
    )
    assert expanding_features(base)[base["night_rank"] == 1].isna().all().all()


def test_expanding_baseline_never_reads_labels():
    base = epoch_table()
    relabelled = base.assign(expert=np.random.default_rng(1).permutation(base["expert"].to_numpy()))
    pd.testing.assert_frame_equal(expanding_features(base), expanding_features(relabelled))
