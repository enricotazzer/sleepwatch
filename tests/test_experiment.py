import numpy as np
import pandas as pd
import pytest

from sleepwatch.constants import N2, N3, REM, UNKNOWN, WAKE
from sleepwatch.models.data import ALL_PRIOR
from sleepwatch.models.experiment import StagingConfig, compute_metrics, run_folds, save_run
from sleepwatch.models.finetune import FinetuneConfig, FinetuneSetting
from sleepwatch.models.gru import GRUConfig
from sleepwatch.models.hgb import HGBConfig
from sleepwatch.models.splits import make_folds


def synthetic_epochs(subjects=10, nights=5, n=60, seed=0) -> pd.DataFrame:
    """Epoch table whose stages are learnable from heart rate and activity."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(subjects):
        for night in range(1, nights + 1):
            stage = rng.choice([WAKE, N2, N3, REM], n, p=[0.2, 0.4, 0.2, 0.2])
            stage[rng.random(n) < 0.03] = UNKNOWN
            hr = (
                60
                + 0.5 * s
                + np.select([stage == WAKE, stage == REM, stage == N3], [10, 5, -4], 0)
                + rng.normal(0, 1, n)
            )
            act = np.where(stage == WAKE, 1e-2, 2e-4) * rng.uniform(0.5, 2, n)
            rows.append(
                pd.DataFrame(
                    {
                        "subject": f"S{s:02d}",
                        "night": night,
                        "epoch": np.arange(n),
                        "t_start": np.arange(n) * 30.0,
                        "expert": stage,
                        "dreem": stage,
                        "labeled": stage != UNKNOWN,
                        "qc_hr_coverage": 1.0,
                        "qc_motion_coverage": 1.0,
                        "hr_mean": hr,
                        "hr_std": np.where(stage == N3, 0.3, 1.5) * rng.uniform(0.8, 1.2, n),
                        "activity": act,
                        "enmo_mean": act * 3,
                        "angle_change": np.where(stage == WAKE, 5.0, 0.05),
                        "hours_since_onset": (np.arange(n) - 5) * 30 / 3600,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def epochs():
    return synthetic_epochs()


@pytest.fixture(scope="module")
def folds(epochs):
    nights = list(epochs[["subject", "night"]].drop_duplicates().itertuples(index=False))
    return make_folds([tuple(x) for x in nights], 5, seed=0)


QUICK_HGB = HGBConfig(max_iter=30, grid=[{"max_leaf_nodes": 7, "class_weight": None}])
QUICK_GRU = GRUConfig(hidden=8, layers=1, lr=3e-2, max_epochs=15, patience=15, threads=1)


@pytest.mark.parametrize("model", ["hgb", "gru"])
def test_cross_validated_predictions_are_complete_and_leak_free(epochs, folds, model):
    cfg = StagingConfig(
        name="smoke",
        model=model,
        personalization=["prior_norm", "prior_stage"],
        n_prior=[1, 2],
        n_boot=20,
        hgb=QUICK_HGB,
        gru=QUICK_GRU,
    )
    pred, fold_log = run_folds(cfg, epochs, folds, log=lambda *_: None)

    population = pred[pred["method"] == "population"]
    labelled = epochs[epochs["expert"] != UNKNOWN]
    key = ["subject", "night", "epoch"]
    assert len(population) == len(labelled)  # every labelled epoch scored exactly once
    assert not population.duplicated(key).any()
    for entry in fold_log:
        assert not set(entry["train"]) & set(entry["test"])
        assert set(entry["val"]) <= set(entry["train"])
        fold_rows = pred[pred["fold"] == entry["fold"]]
        assert set(fold_rows["subject"]) == set(entry["test"])
    personalized = pred[pred["method"] != "population"]
    assert personalized["eval_night"].all()  # N-curve variants score evaluation nights only
    assert set(personalized["method"]) == {"matched", "prior_norm", "prior_stage"}
    for (method, n), rows in personalized.groupby(["method", "n_prior"]):
        assert len(rows) == len(population[population["eval_night"]]), (method, n)

    metrics = compute_metrics(pred, n_boot=20, seed=0)
    assert metrics["population"]["all"]["5"]["pooled"]["kappa"] > 0.3  # learnable signal
    assert set(metrics["n_curve"]) == {"matched", "prior_norm", "prior_stage"}
    assert "vs_matched" in metrics["n_curve"]["prior_norm"][1]


def test_label_shift_is_estimated_on_training_subjects(epochs, folds):
    cfg = StagingConfig(name="shift", model="hgb", label_shift="estimate", n_boot=20, hgb=QUICK_HGB)
    late = epochs.copy()
    late["expert"] = (
        late.groupby(["subject", "night"])["expert"].shift(2).fillna(UNKNOWN).astype(int)
    )
    _, fold_log = run_folds(cfg, late, folds, log=lambda *_: None)
    assert {entry["label_lag"] for entry in fold_log} == {-2}


def test_saved_run(tmp_path, epochs, folds):
    cfg = StagingConfig(name="smoke", model="hgb", folds=[0], n_boot=20, hgb=QUICK_HGB)
    pred, fold_log = run_folds(cfg, epochs, folds, log=lambda *_: None)
    run_dir = save_run(cfg, pred, fold_log, compute_metrics(pred, 20, 0), tmp_path, extra={})
    for name in ["config.yaml", "run.json", "predictions.parquet", "metrics.json", "per_night.csv"]:
        assert (run_dir / name).is_file()
    index = pd.read_csv(tmp_path / "index.csv")
    assert index.loc[0, "partial"] and index.loc[0, "name"] == "smoke"


@pytest.mark.parametrize("model", ["hgb", "gru"])
def test_parallel_folds_give_identical_predictions(epochs, folds, model):
    cfg = StagingConfig(
        name="jobs", model=model, folds=[0, 1], n_boot=20, hgb=QUICK_HGB, gru=QUICK_GRU
    )
    serial, log_serial = run_folds(cfg, epochs, folds, log=lambda *_: None, jobs=1)
    parallel, log_parallel = run_folds(cfg, epochs, folds, log=lambda *_: None, jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)
    assert [e["fold"] for e in log_parallel] == [e["fold"] for e in log_serial] == [0, 1]


QUICK_FINETUNE = FinetuneConfig(
    grid=[
        FinetuneSetting(lr=1e-3, epochs=2, layers="output"),
        FinetuneSetting(lr=1e-2, epochs=3, layers="all"),
    ]
)


def test_phase2b_methods_score_exactly_the_evaluation_nights(epochs, folds):
    cfg = StagingConfig(
        name="2b",
        model="gru",
        personalization=["expanding_norm", "context", "finetune"],
        n_prior=[1, 2],
        folds=[0, 1],
        n_boot=20,
        gru=QUICK_GRU,
        finetune=QUICK_FINETUNE,
    )
    pred, fold_log = run_folds(cfg, epochs, folds, log=lambda *_: None)
    population = pred[pred["method"] == "population"]
    eval_rows = population[population["eval_night"]]
    expected = {("expanding_norm", ALL_PRIOR), ("context", ALL_PRIOR)}
    expected |= {("context_shuffled", ALL_PRIOR)}
    expected |= {(m, n) for m in ("finetune", "finetune_other") for n in (1, 2, ALL_PRIOR)}
    variants = pred[pred["method"] != "population"].groupby(["method", "n_prior"])
    assert set(variants.groups) == expected
    key = ["subject", "night", "epoch", "y_true"]
    for name, rows in variants:
        assert rows[key].reset_index(drop=True).equals(eval_rows[key].reset_index(drop=True)), name
    for entry in fold_log:
        chosen = entry["finetune"]["setting"]
        assert chosen in [g.model_dump() for g in QUICK_FINETUNE.grid]
        assert set(entry["finetune"]["validation_subjects"]) <= set(entry["val"])
        donors = entry["finetune"]["donors"]
        assert set(donors) <= set(entry["test"]) and all(s != d for s, d in donors.items())
    metrics = compute_metrics(pred, n_boot=20, seed=0)
    assert "vs_context_shuffled" in metrics["n_curve"]["context"]["all"]
    assert "vs_finetune_other" in metrics["n_curve"]["finetune"][1]
    assert "vs_population" in metrics["n_curve"]["expanding_norm"]["all"]


def test_expanding_baseline_runs_with_trees(epochs, folds):
    cfg = StagingConfig(
        name="2b", model="hgb", personalization=["expanding_norm"], folds=[0], hgb=QUICK_HGB
    )
    pred, _ = run_folds(cfg, epochs, folds, log=lambda *_: None)
    assert set(pred["method"]) == {"population", "expanding_norm"}


def test_learned_and_finetuned_personalization_need_the_gru():
    with pytest.raises(ValueError, match="need model: gru"):
        StagingConfig(name="x", model="hgb", personalization=["context"])
