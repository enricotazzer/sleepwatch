import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from sleepwatch.models.metrics import (
    PROBA_COLUMNS,
    calibration_bins,
    confusion,
    labels_and_proba,
    paired_difference,
    per_group_scores,
    scores_from_tables,
    summarize,
)


def random_predictions(n=3000, subjects=6, seed=0, skill=0.6):
    rng = np.random.default_rng(seed)
    y = rng.choice(5, n, p=[0.1, 0.07, 0.4, 0.18, 0.25])
    logits = rng.normal(0, 1, (n, 5))
    logits[np.arange(n), y] += skill * 3
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    pred = pd.DataFrame(proba, columns=PROBA_COLUMNS)
    pred["y_true"] = y
    pred["subject"] = rng.choice([f"S{i}" for i in range(subjects)], n)
    pred["night"] = rng.integers(1, 4, n)
    return pred


@pytest.mark.parametrize("scheme", ["5", "4"])
def test_scores_match_sklearn(scheme):
    pred = random_predictions()
    y, proba = labels_and_proba(pred, scheme)
    y_hat = proba.argmax(axis=1)
    k = proba.shape[1]
    s = scores_from_tables(confusion(y, y_hat, k), calibration_bins(y, proba), ["a"] * k)
    assert s["accuracy"] == pytest.approx(accuracy_score(y, y_hat))
    assert s["kappa"] == pytest.approx(cohen_kappa_score(y, y_hat))
    assert s["macro_f1"] == pytest.approx(f1_score(y, y_hat, average="macro"))


def test_macro_f1_ignores_stages_absent_from_the_truth():
    y = np.array([0, 0, 2, 2, 2])
    y_hat = np.array([0, 2, 2, 2, 3])  # predicts N3, which never occurs
    proba = np.eye(5)[y_hat]
    s = scores_from_tables(confusion(y, y_hat, 5), calibration_bins(y, proba), list("abcde"))
    expected = f1_score(y, y_hat, labels=[0, 2], average="macro")
    assert s["macro_f1"] == pytest.approx(expected)
    assert np.isnan(s["f1_d"])  # N3 absent from the truth: undefined, not 0


def test_four_class_merges_light_sleep_probabilities():
    pred = pd.DataFrame([[0.3, 0.2, 0.25, 0.15, 0.1]], columns=PROBA_COLUMNS).assign(y_true=1)
    y, proba = labels_and_proba(pred, "4")
    assert y.tolist() == [1]  # N1 -> Light
    assert proba.argmax() == 1  # Light (0.45) beats Wake (0.3), though Wake is the 5-class argmax


def test_ece_is_zero_when_confidence_matches_accuracy():
    # Confidence 0.8 everywhere and exactly 80% correct.
    y = np.array([0] * 8 + [1] * 2)
    proba = np.tile([0.8, 0.05, 0.05, 0.05, 0.05], (10, 1))
    s = scores_from_tables(
        confusion(y, proba.argmax(1), 5), calibration_bins(y, proba), list("abcde")
    )
    assert s["ece"] == pytest.approx(0.0)


def test_ece_of_an_overconfident_model():
    y = np.array([0] * 5 + [1] * 5)  # 50% correct at confidence 0.9
    proba = np.tile([0.9, 0.025, 0.025, 0.025, 0.025], (10, 1))
    s = scores_from_tables(
        confusion(y, proba.argmax(1), 5), calibration_bins(y, proba), list("abcde")
    )
    assert s["ece"] == pytest.approx(0.4)


def test_summary_and_bootstrap():
    pred = random_predictions()
    out = summarize(pred, n_boot=200, seed=1)
    assert out["subjects"] == 6
    low, high = out["ci95"]["kappa"]
    assert low <= out["pooled"]["kappa"] <= high
    assert summarize(pred, n_boot=200, seed=1)["ci95"] == out["ci95"]  # seeded
    per_subject = per_group_scores(pred, ["subject"])
    assert out["subject_mean"]["kappa"] == pytest.approx(per_subject["kappa"].mean())


def test_paired_difference():
    a = random_predictions(seed=0, skill=0.4)
    b = a.copy()
    b[PROBA_COLUMNS] = random_predictions(seed=0, skill=0.9)[PROBA_COLUMNS].to_numpy()
    out = paired_difference(a, b, n_boot=200)
    assert out["kappa"]["diff"] > 0 and out["kappa"]["ci95"][0] > 0  # clearly better
    same = paired_difference(a, a, n_boot=200)
    assert same["kappa"]["diff"] == 0 and same["kappa"]["ci95"] == [0.0, 0.0]
