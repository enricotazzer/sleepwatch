"""Config-driven staging experiments with subject-level cross-validation.

For every outer fold (fixed fold file), models are trained on the training subjects only and
predict the test subjects:

- ``population``: one model on all training nights, predicting every test night;
- each personalization method x N prior nights: a model trained on training subjects' nights
  after their first N, predicting the fixed evaluation nights (night 4+) of test subjects.

Tuning, early stopping and the label-shift estimate use training subjects only. Every run writes
its resolved config, provenance, predictions and metrics to ``results/<name>/<timestamp>/`` and a
row to ``results/index.csv``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict

from sleepwatch.constants import UNKNOWN
from sleepwatch.features.epoch_features import feature_columns
from sleepwatch.models import gru as gru_model
from sleepwatch.models import hgb as hgb_model
from sleepwatch.models.data import estimate_label_lag, personal_features, prepare, shifted
from sleepwatch.models.gru import GRUConfig
from sleepwatch.models.hgb import HGBConfig
from sleepwatch.models.metrics import PROBA_COLUMNS, paired_difference, per_group_scores, summarize
from sleepwatch.models.splits import inner_split, outer_splits
from sleepwatch.provenance import git_revision, timestamp

MIN_COVERAGE = 0.5


class StagingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: Literal["hgb", "gru"]
    features: str = "v1"
    train_labels: Literal["expert", "dreem"] = "expert"  # evaluation is always against expert
    label_shift: Literal["none", "estimate"] = "none"
    personalization: list[Literal["prior_norm", "prior_stage"]] = []
    n_prior: list[int] = [1, 2, 3]
    val_fraction: float = 0.2
    seed: int = 42
    folds: list[int] | None = None  # a subset only for quick checks; recorded as partial
    n_boot: int = 1000
    hgb: HGBConfig = HGBConfig()
    gru: GRUConfig = GRUConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> StagingConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


def _fit_predict(cfg, train_rows, test_rows, cols, fit_s, val_s, seed, tuned):
    """Train on ``train_rows`` (target ``_target``) and return test probabilities + a log."""
    train_rows = train_rows[train_rows["_target"] != UNKNOWN]
    fit = train_rows[train_rows["subject"].isin(fit_s)]
    val = train_rows[train_rows["subject"].isin(val_s)]
    if cfg.model == "hgb":
        if tuned is None:
            params, n_iter, _, grid = hgb_model.tune(
                cfg.hgb, fit[cols], fit["_target"], val[cols], val["_target"], seed
            )
        else:
            params, grid = tuned, None
            n_iter = hgb_model.fit_with_early_stopping(
                cfg.hgb, params, fit[cols], fit["_target"], val[cols], val["_target"], seed
            ).n_iter_
        model = hgb_model.refit(
            cfg.hgb, params, n_iter, train_rows[cols], train_rows["_target"], seed
        )
        proba = hgb_model.predict_proba(model, test_rows[cols])
        return proba, {"params": params, "n_iter": n_iter, "grid": grid}
    prep = gru_model.Preprocessor(cols).fit(train_rows)
    model, history = gru_model.train(
        gru_model.night_arrays(fit, prep, "_target"),
        gru_model.night_arrays(val, prep, "_target"),
        prep.n_inputs,
        cfg.gru,
        seed,
    )
    proba = np.concatenate(gru_model.predict(model, gru_model.night_arrays(test_rows, prep, None)))
    return proba, history


def run_folds(
    cfg: StagingConfig, epochs: pd.DataFrame, folds: dict[str, int], log=print, jobs: int = 1
):
    """Cross-validated predictions (one row per test epoch per variant) and a per-fold log.

    With ``jobs > 1`` folds train in parallel processes. Every fold is seeded on its own, so the
    output doesn't depend on ``jobs``; progress is then logged as each fold finishes.
    """
    base_cols = feature_columns(epochs)
    epochs = prepare(epochs)
    epochs["covered"] = (epochs["qc_hr_coverage"] >= MIN_COVERAGE) & (
        epochs["qc_motion_coverage"] >= MIN_COVERAGE
    )
    tasks = [
        (fold, train_s, test_s)
        for fold, (train_s, test_s) in enumerate(outer_splits(folds))
        if cfg.folds is None or fold in cfg.folds
    ]
    if jobs == 1:
        results = (_run_fold(cfg, epochs, base_cols, *task, log) for task in tasks)
    else:
        results = Parallel(n_jobs=jobs, return_as="generator")(
            delayed(_run_fold)(cfg, epochs, base_cols, *task, None) for task in tasks
        )
    predictions, fold_log = [], []
    for fold_predictions, entry, messages in results:
        for message in messages:
            log(message)
        predictions.extend(fold_predictions)
        fold_log.append(entry)
    return pd.concat(predictions, ignore_index=True), fold_log


def _run_fold(cfg, epochs, base_cols, fold, train_s, test_s, log):
    """Every variant for one outer fold; messages are returned when ``log`` is None."""
    messages = []
    emit = log or messages.append
    # "matched" is the control for personalization: trained on exactly the same nights as the
    # personalized models (each subject's nights after the first N) but with no personal
    # features, so a personalization gain isn't confounded with the smaller training set.
    personalized = [m for m in ("matched", *cfg.personalization) if cfg.personalization]
    variants = [("population", 0)] + [
        (method, n) for method in personalized for n in cfg.n_prior if n > 0
    ]
    if set(train_s) & set(test_s):
        raise RuntimeError(f"fold {fold}: a subject is in both training and test")
    seed = cfg.seed + fold
    in_train = epochs["subject"].isin(train_s)
    lag = estimate_label_lag(epochs[in_train]) if cfg.label_shift == "estimate" else 0
    data = epochs.assign(
        _target=shifted(epochs, cfg.train_labels, lag), _truth=shifted(epochs, "expert", lag)
    )
    fit_s, val_s = inner_split(train_s, cfg.val_fraction, seed)
    entry = {"fold": fold, "train": train_s, "test": test_s, "val": val_s, "label_lag": lag}
    tuned, predictions = None, []
    for method, n in variants:
        frame, cols, usable = data, base_cols, np.ones(len(data), bool)
        if method != "population":
            usable = (data["night_rank"] > n).to_numpy()
        if method in ("prior_norm", "prior_stage"):
            extra = personal_features(data, n, method, label="_target")
            frame = pd.concat([data, extra], axis=1)
            cols = base_cols + list(extra.columns)
        test_mask = data["subject"].isin(test_s).to_numpy()
        if method != "population":
            test_mask = test_mask & data["eval_night"].to_numpy()
        train_rows = frame[in_train.to_numpy() & usable]
        test_rows = frame[test_mask]
        # A personal feature can be missing for every training row (e.g. no one had N1 on
        # their first night); it carries no information and breaks tree binning.
        cols = [c for c in cols if train_rows[c].notna().any()]
        proba, info = _fit_predict(cfg, train_rows, test_rows, cols, fit_s, val_s, seed, tuned)
        if method == "population" and cfg.model == "hgb":
            tuned = info["params"]  # personalized variants reuse the tuned hyperparameters
        entry[f"{method}_{n}"] = info
        out = test_rows[["subject", "night", "epoch", "eval_night", "covered", "_truth"]]
        out = out.rename(columns={"_truth": "y_true"}).assign(fold=fold, method=method, n_prior=n)
        out[PROBA_COLUMNS] = proba
        predictions.append(out[out["y_true"] != UNKNOWN])
        emit(f"fold {fold} {method} N={n}: {len(test_rows)} test epochs")
    return predictions, entry, messages


def compute_metrics(pred: pd.DataFrame, n_boot: int, seed: int) -> dict:
    population = pred[pred["method"] == "population"]
    out = {
        "population": {
            subset: {scheme: summarize(rows, scheme, n_boot, seed) for scheme in ("5", "4")}
            for subset, rows in (
                ("all", population),
                ("covered", population[population["covered"]]),
            )
        },
        "n_curve": {},
    }
    baseline = population[population["eval_night"]]
    for method in sorted(set(pred["method"]) - {"population"}):
        curve = {0: {s: summarize(baseline, s, n_boot, seed) for s in ("5", "4")}}
        for n in sorted(pred.loc[pred["method"] == method, "n_prior"].unique()):
            rows = pred[(pred["method"] == method) & (pred["n_prior"] == n)]
            curve[int(n)] = {s: summarize(rows, s, n_boot, seed) for s in ("5", "4")}
            curve[int(n)]["vs_population"] = {
                s: paired_difference(baseline, rows, s, n_boot, seed) for s in ("5", "4")
            }
            control = pred[(pred["method"] == "matched") & (pred["n_prior"] == n)]
            if method != "matched" and len(control):
                curve[int(n)]["vs_matched"] = {
                    s: paired_difference(control, rows, s, n_boot, seed) for s in ("5", "4")
                }
        out["n_curve"][method] = curve
    return out


def save_run(cfg, pred, fold_log, metrics, results_dir: Path, extra: dict) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_dir / cfg.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    git = git_revision()
    partial = cfg.folds is not None
    info = {"created": timestamp(), "git": git, "partial": partial, "folds": fold_log, **extra}
    (run_dir / "run.json").write_text(json.dumps(info, indent=2, default=str))
    pred.to_parquet(run_dir / "predictions.parquet", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    population = pred[pred["method"] == "population"]
    per_group_scores(population, ["subject"]).to_csv(run_dir / "per_subject.csv")
    per_group_scores(population, ["subject", "night"]).to_csv(run_dir / "per_night.csv")

    pooled5 = metrics["population"]["all"]["5"]["pooled"]
    pooled4 = metrics["population"]["all"]["4"]["pooled"]
    row = {
        "name": cfg.name,
        "run": run_id,
        "model": cfg.model,
        "commit": (git["commit"] or "")[:8],
        "dirty": git["dirty"],
        "partial": partial,
        "kappa_5": round(pooled5["kappa"], 4),
        "macro_f1_5": round(pooled5["macro_f1"], 4),
        "kappa_4": round(pooled4["kappa"], 4),
        "macro_f1_4": round(pooled4["macro_f1"], 4),
        "ece_5": round(pooled5["ece"], 4),
    }
    index = results_dir / "index.csv"
    pd.DataFrame([row]).to_csv(index, mode="a", header=not index.exists(), index=False)
    return run_dir
