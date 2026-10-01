"""Config-driven staging experiments with subject-level cross-validation.

For every outer fold (fixed fold file), models are trained on the training subjects only and
predict the test subjects:

- ``population``: one model on all training nights, predicting every test night;
- each personalization method x N prior nights: a model trained on training subjects' nights
  after their first N, predicting the fixed evaluation nights (night 4+) of test subjects;
- Phase 2b methods (``expanding_norm``, ``context``, ``finetune``), each with its own control,
  predicting the same evaluation nights. ``n_prior = ALL_PRIOR`` marks "every earlier night".

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
from pydantic import BaseModel, ConfigDict, model_validator
from sklearn.metrics import cohen_kappa_score

from sleepwatch.constants import UNKNOWN
from sleepwatch.features.epoch_features import feature_columns
from sleepwatch.models import context as context_model
from sleepwatch.models import finetune as finetune_model
from sleepwatch.models import gru as gru_model
from sleepwatch.models import hgb as hgb_model
from sleepwatch.models.context import ContextConfig
from sleepwatch.models.data import (
    ALL_PRIOR,
    estimate_label_lag,
    expanding_features,
    personal_features,
    prepare,
    shifted,
)
from sleepwatch.models.finetune import FinetuneConfig
from sleepwatch.models.gru import GRUConfig
from sleepwatch.models.hgb import HGBConfig
from sleepwatch.models.metrics import PROBA_COLUMNS, paired_difference, per_group_scores, summarize
from sleepwatch.models.splits import inner_split, outer_splits
from sleepwatch.provenance import git_revision, timestamp

MIN_COVERAGE = 0.5
FIXED_N = ("prior_norm", "prior_stage")  # personal features from the first N nights
# Each personalized method's control (the population model is always compared as well).
CONTROLS = {
    "prior_norm": "matched",
    "prior_stage": "matched",
    "context": "context_shuffled",
    "finetune": "finetune_other",
}


class StagingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: Literal["hgb", "gru"]
    features: str = "v1"
    train_labels: Literal["expert", "dreem"] = "expert"  # evaluation is always against expert
    label_shift: Literal["none", "estimate"] = "none"
    personalization: list[
        Literal["prior_norm", "prior_stage", "expanding_norm", "context", "finetune"]
    ] = []
    n_prior: list[int] = [1, 2, 3]
    val_fraction: float = 0.2
    seed: int = 42
    folds: list[int] | None = None  # a subset only for quick checks; recorded as partial
    n_boot: int = 1000
    hgb: HGBConfig = HGBConfig()
    gru: GRUConfig = GRUConfig()
    context: ContextConfig = ContextConfig()
    finetune: FinetuneConfig = FinetuneConfig()

    @model_validator(mode="after")
    def _gru_only_methods(self) -> StagingConfig:
        if self.model != "gru" and {"context", "finetune"} & set(self.personalization):
            raise ValueError("'context' and 'finetune' need model: gru")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> StagingConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


def _fit_predict(cfg, train_rows, test_rows, cols, fit_s, val_s, seed, tuned, flags=()):
    """Train on ``train_rows`` (target ``_target``): test probabilities, a log, the model.

    The GRU's model is returned as ``(net, preprocessor)``; ``flags`` adds availability inputs.
    """
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
        return proba, {"params": params, "n_iter": n_iter, "grid": grid}, model
    prep = gru_model.Preprocessor(cols, ("hr_mean", "activity", *flags)).fit(train_rows)
    model, history = gru_model.train(
        gru_model.night_arrays(fit, prep, "_target"),
        gru_model.night_arrays(val, prep, "_target"),
        prep.n_inputs,
        cfg.gru,
        seed,
    )
    proba = np.concatenate(gru_model.predict(model, gru_model.night_arrays(test_rows, prep, None)))
    return proba, history, (model, prep)


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


def _variants(cfg: StagingConfig) -> list[tuple[str, int]]:
    """Model variants trained in every fold (fine-tuning is handled separately)."""
    variants = [("population", 0)]
    # "matched" is the control for the first-N methods: trained on exactly the same nights as
    # the personalized models (each subject's nights after the first N) but with no personal
    # features, so a personalization gain isn't confounded with the smaller training set.
    fixed = [m for m in cfg.personalization if m in FIXED_N]
    if fixed:
        variants += [(m, n) for m in ("matched", *fixed) for n in cfg.n_prior if n > 0]
    if "expanding_norm" in cfg.personalization:
        variants.append(("expanding_norm", ALL_PRIOR))
    if "context" in cfg.personalization:
        variants += [("context", ALL_PRIOR), ("context_shuffled", ALL_PRIOR)]
    return variants


def _output(rows, proba, fold, method, n):
    out = rows[["subject", "night", "epoch", "eval_night", "covered", "_truth"]]
    out = out.rename(columns={"_truth": "y_true"}).assign(fold=fold, method=method, n_prior=n)
    out[PROBA_COLUMNS] = proba
    return out[out["y_true"] != UNKNOWN]


def _run_fold(cfg, epochs, base_cols, fold, train_s, test_s, log):
    """Every variant for one outer fold; messages are returned when ``log`` is None."""
    messages = []
    emit = log or messages.append
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
    in_test = data["subject"].isin(test_s).to_numpy()
    eval_test = in_test & data["eval_night"].to_numpy()
    tuned, population, predictions = None, None, []
    for method, n in _variants(cfg):
        if method in ("context", "context_shuffled"):
            test_rows = data[eval_test]
            proba, info = _context_fit_predict(
                cfg, data, base_cols, train_s, test_s, fit_s, val_s, seed, method
            )
        else:
            frame, cols, flags = data, base_cols, ()
            usable = np.ones(len(data), bool)
            if method in ("matched", *FIXED_N):
                usable = (data["night_rank"] > n).to_numpy()
            if method in FIXED_N:
                extra = personal_features(data, n, method, label="_target")
            elif method == "expanding_norm":
                extra = expanding_features(data)
                flags = ("p_hr_mean_level",)  # GRU: whether the night has a baseline at all
            if method in (*FIXED_N, "expanding_norm"):
                frame = pd.concat([data, extra], axis=1)
                cols = base_cols + list(extra.columns)
            test_rows = frame[eval_test if method != "population" else in_test]
            train_rows = frame[in_train.to_numpy() & usable]
            # A personal feature can be missing for every training row (e.g. no one had N1 on
            # their first night); it carries no information and breaks tree binning.
            cols = [c for c in cols if train_rows[c].notna().any()]
            proba, info, fitted = _fit_predict(
                cfg, train_rows, test_rows, cols, fit_s, val_s, seed, tuned, flags
            )
            if method == "population":
                population = fitted
                if cfg.model == "hgb":
                    tuned = info["params"]  # personalized variants reuse the tuned grid point
        entry[f"{method}_{n}"] = info
        predictions.append(_output(test_rows, proba, fold, method, n))
        emit(f"fold {fold} {method} N={n}: {len(test_rows)} test epochs")
    if "finetune" in cfg.personalization:
        results, entry["finetune"] = _finetune_variants(cfg, data, population, test_s, val_s, seed)
        test_rows = data[eval_test]
        for (method, n), proba in results.items():
            predictions.append(_output(test_rows, proba, fold, method, n))
            emit(f"fold {fold} {method} N={n}: {len(test_rows)} test epochs")
    return predictions, entry, messages


def _night_order(data: pd.DataFrame) -> dict[str, list[tuple[str, int]]]:
    """Each subject's nights as ``(subject, night)`` keys, in night order."""
    nights = data[["subject", "night", "night_rank"]].drop_duplicates()
    nights = nights.sort_values(["subject", "night_rank"])
    return {
        s: list(zip(g["subject"], g["night"], strict=True)) for s, g in nights.groupby("subject")
    }


def _eval_subjects(data: pd.DataFrame, subjects) -> list[str]:
    rows = data[data["subject"].isin(subjects) & data["eval_night"]]
    return sorted(rows["subject"].unique())


def _context_fit_predict(cfg, data, base_cols, train_s, test_s, fit_s, val_s, seed, method):
    """Method B: a GRU given the person's (or, for the control, a donor's) earlier nights."""
    labelled = data[data["_target"] != UNKNOWN]
    train_rows = labelled[labelled["subject"].isin(train_s)]
    prep = gru_model.Preprocessor(base_cols).fit(train_rows)
    # Context is label-free, so it uses every epoch of every night (test subjects' included).
    nights = {item[0]: item[1] for item in gru_model.night_arrays(data, prep, None)}
    ranks = data.groupby(["subject", "night"])["night_rank"].first().to_dict()
    shuffled = method == "context_shuffled"
    train_donor = context_model.donors(train_s, seed) if shuffled else None
    test_donor = context_model.donors(_eval_subjects(data, test_s), seed) if shuffled else None

    def items(rows, target, donor):
        arrays = gru_model.night_arrays(rows, prep, target)
        return context_model.with_context(arrays, nights, ranks, donor)

    fit = items(train_rows[train_rows["subject"].isin(fit_s)], "_target", train_donor)
    val = items(train_rows[train_rows["subject"].isin(val_s)], "_target", train_donor)
    test = items(data[data["subject"].isin(test_s) & data["eval_night"]], None, test_donor)
    model, history = gru_model.train(
        fit,
        val,
        prep.n_inputs,
        cfg.gru,
        seed,
        build=lambda n_inputs, gru: context_model.ContextNet(n_inputs, gru, cfg.context),
        collate_fn=context_model.collate,
    )
    proba = gru_model.predict(model, test, collate_fn=context_model.collate)
    return np.concatenate(proba), {**history, "donors": test_donor}


def _finetune_jobs(order, subjects, eval_keys, n_prior, donor=None):
    """``(n, training nights, target nights)`` per fine-tuning run for ``subjects``.

    First-N runs train on the first N nights and predict every evaluation night; ``ALL_PRIOR``
    runs train on the nights before one evaluation night and predict it. With ``donor``, the
    training nights are the donor's (the same count, or as many as the donor has).
    """
    jobs = []
    for subject in subjects:
        nights = order[subject]
        targets = [k for k in nights if k in eval_keys]
        source = order[donor[subject]] if donor else nights
        jobs += [(n, source[:n], targets) for n in n_prior]
        jobs += [(ALL_PRIOR, source[: nights.index(k)], [k]) for k in targets]
    return jobs


def _run_finetune_jobs(cfg, model, items, jobs, setting, seed):
    """Probabilities per ``(n, target night)``."""
    out = {}
    for j, (n, train_keys, targets) in enumerate(jobs):
        tuned = finetune_model.finetune(
            model, [items[k] for k in train_keys], setting, cfg.finetune, cfg.gru, seed + j
        )
        probas = gru_model.predict(tuned, [items[k] for k in targets])
        out.update({(n, k): p for k, p in zip(targets, probas, strict=True)})
    return out


def _finetune_variants(cfg, data, population, test_s, val_s, seed):
    """Method C: fine-tune the population GRU per test subject; grid point from validation."""
    if cfg.model != "gru":
        raise ValueError("fine-tuning is implemented for the GRU only")
    model, prep = population
    order = _night_order(data)
    eval_keys = set(map(tuple, data.loc[data["eval_night"], ["subject", "night"]].to_numpy()))
    eval_keys = {(s, int(n)) for s, n in eval_keys}
    n_prior = [n for n in cfg.n_prior if n > 0]
    keep = data["subject"].isin([*val_s, *test_s])
    items = {i[0]: i for i in gru_model.night_arrays(data[keep], prep, "_target")}

    val_subjects = _eval_subjects(data, val_s)
    val_jobs = _finetune_jobs(order, val_subjects, eval_keys, n_prior)
    scores = []
    for setting in cfg.finetune.grid:
        probas = _run_finetune_jobs(cfg, model, items, val_jobs, setting, seed)
        y = np.concatenate([items[k][2] for _, k in probas])
        y_hat = np.concatenate([p.argmax(axis=1) for p in probas.values()])
        known = y != gru_model.IGNORE
        scores.append(cohen_kappa_score(y[known], y_hat[known]) if known.any() else np.nan)
    best = cfg.finetune.grid[int(np.nanargmax(scores)) if np.isfinite(scores).any() else 0]

    test_subjects = _eval_subjects(data, test_s)
    donor = context_model.donors(test_subjects, seed)
    eval_rows = data[data["subject"].isin(test_s) & data["eval_night"]]
    targets = list(dict.fromkeys(zip(eval_rows["subject"], eval_rows["night"], strict=True)))
    results = {}
    for method, pairing in (("finetune", None), ("finetune_other", donor)):
        jobs = _finetune_jobs(order, test_subjects, eval_keys, n_prior, pairing)
        probas = _run_finetune_jobs(cfg, model, items, jobs, best, seed)
        for n in [*n_prior, ALL_PRIOR]:
            results[(method, n)] = np.concatenate([probas[(n, k)] for k in targets])
    info = {
        "setting": best.model_dump(),
        "validation_kappa": [
            {**g.model_dump(), "kappa": k} for g, k in zip(cfg.finetune.grid, scores, strict=True)
        ],
        "validation_subjects": val_subjects,
        "donors": donor,
    }
    return results, info


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
            key = "all" if n == ALL_PRIOR else int(n)
            rows = pred[(pred["method"] == method) & (pred["n_prior"] == n)]
            curve[key] = {s: summarize(rows, s, n_boot, seed) for s in ("5", "4")}
            curve[key]["vs_population"] = {
                s: paired_difference(baseline, rows, s, n_boot, seed) for s in ("5", "4")
            }
            reference = CONTROLS.get(method)
            control = pred[(pred["method"] == reference) & (pred["n_prior"] == n)]
            if reference and len(control):
                curve[key][f"vs_{reference}"] = {
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
