"""Gradient-boosted trees (scikit-learn ``HistGradientBoostingClassifier``) on epoch features.

Each epoch is classified independently from its features (which already include context
windows). Hyperparameters are chosen on held-out *training* subjects; early stopping uses the
same held-out subjects; the final model is refitted on all training subjects with the chosen
number of boosting rounds. Missing values are handled natively by the trees.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

N_CLASSES = 5


class HGBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 0.1
    max_iter: int = 1000
    n_iter_no_change: int = 20
    min_samples_leaf: int = 100
    l2_regularization: float = 1.0
    grid: list[dict] = [
        {"max_leaf_nodes": leaves, "class_weight": weight}
        for leaves in (15, 31, 63)
        for weight in (None, "balanced")
    ]


def _estimator(
    cfg: HGBConfig, params: dict, seed: int, **overrides
) -> HistGradientBoostingClassifier:
    settings = {
        "learning_rate": cfg.learning_rate,
        "max_iter": cfg.max_iter,
        "min_samples_leaf": cfg.min_samples_leaf,
        "l2_regularization": cfg.l2_regularization,
        "early_stopping": True,
        "n_iter_no_change": cfg.n_iter_no_change,
        "scoring": "loss",
        "random_state": seed,
        **params,
        **overrides,
    }
    return HistGradientBoostingClassifier(**settings)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the stages present in ``y_true`` (same convention as metrics.py)."""
    return f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro")


def predict_proba(model: HistGradientBoostingClassifier, X: np.ndarray) -> np.ndarray:
    """Probabilities for all five stages, even if a stage was absent from training."""
    out = np.zeros((len(X), N_CLASSES))
    out[:, model.classes_.astype(int)] = model.predict_proba(X)
    return out


def tune(cfg, X_fit, y_fit, X_val, y_val, seed):
    """Fit every grid point on fit subjects with early stopping on validation subjects.

    Returns ``(best_params, best_n_iter, best_validation_model, table)``; selection is by
    validation macro-F1.
    """
    rows, best = [], None
    for params in cfg.grid:
        model = _estimator(cfg, params, seed).fit(X_fit, y_fit, X_val=X_val, y_val=y_val)
        score = macro_f1(y_val, predict_proba(model, X_val).argmax(axis=1))
        rows.append({**params, "n_iter": model.n_iter_, "val_macro_f1": score})
        if best is None or score > best[0]:
            best = (score, params, model.n_iter_, model)
    _, params, n_iter, model = best
    return params, n_iter, model, rows


def fit_with_early_stopping(cfg, params, X_fit, y_fit, X_val, y_val, seed):
    return _estimator(cfg, params, seed).fit(X_fit, y_fit, X_val=X_val, y_val=y_val)


def refit(cfg, params, n_iter, X, y, seed):
    """Final model on all training subjects with a fixed number of rounds (no early stopping)."""
    return _estimator(cfg, params, seed, max_iter=n_iter, early_stopping=False).fit(X, y)
