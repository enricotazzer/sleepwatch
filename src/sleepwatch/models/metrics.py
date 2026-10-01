"""Staging metrics: accuracy, macro-F1, Cohen's kappa, per-stage F1 and calibration error.

Everything is computed from confusion matrices and calibration-bin counts, so a subject-level
bootstrap only needs to add up per-subject tables. Conventions:

- 5-class scores use Wake/N1/N2/N3/REM. 4-class scores merge N1 and N2 into Light by *summing
  their probabilities* and taking the argmax, rather than relabelling 5-class predictions.
- Macro-F1 averages over the stages present in the true labels of the group being scored, so a
  subject without N3 isn't scored 0 for a stage it never had.
- ECE is the top-label expected calibration error with equal-width confidence bins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sleepwatch.constants import STAGE4_NAMES, STAGE_NAMES, TO_4CLASS

PROBA_COLUMNS = [f"proba_{k}" for k in range(5)]
CLASS_NAMES = {
    "5": [STAGE_NAMES[k] for k in range(5)],
    "4": [STAGE4_NAMES[k] for k in range(4)],
}
N_BINS = 15


def labels_and_proba(pred: pd.DataFrame, scheme: str) -> tuple[np.ndarray, np.ndarray]:
    """True labels and class probabilities for the 5- or 4-class scheme."""
    y = pred["y_true"].to_numpy().astype(int)
    proba = pred[PROBA_COLUMNS].to_numpy(dtype=float)
    if scheme == "4":
        y = np.array([TO_4CLASS[v] for v in range(5)])[y]
        proba = np.column_stack([proba[:, 0], proba[:, 1] + proba[:, 2], proba[:, 3], proba[:, 4]])
    elif scheme != "5":
        raise ValueError(f"unknown scheme {scheme!r}")
    return y, proba


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """``k x k`` counts; rows are true classes, columns predictions."""
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def calibration_bins(y_true: np.ndarray, proba: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Per confidence bin: ``[count, sum of confidence, sum of correct]``."""
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    out = np.zeros((n_bins, 3))
    np.add.at(out, bins, np.column_stack([np.ones_like(confidence), confidence, correct]))
    return out


def scores_from_tables(cm: np.ndarray, bins: np.ndarray, names: list[str]) -> dict[str, float]:
    total = cm.sum()
    if total == 0:
        return {}
    tp = np.diag(cm).astype(float)
    true_counts, pred_counts = cm.sum(axis=1), cm.sum(axis=0)
    denom = true_counts + pred_counts
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = true_counts > 0
    observed = tp.sum() / total
    expected = (true_counts * pred_counts).sum() / total**2
    scores = {
        "accuracy": observed,
        "macro_f1": f1[present].mean(),
        "kappa": (observed - expected) / (1 - expected) if expected < 1 else np.nan,
        "ece": np.abs(bins[:, 2] - bins[:, 1]).sum() / bins[:, 0].sum(),
        "epochs": int(total),
    }
    scores.update({f"f1_{name}": (f1[i] if present[i] else np.nan) for i, name in enumerate(names)})
    return scores


def _group_tables(pred: pd.DataFrame, scheme: str, by: list[str]):
    k = len(CLASS_NAMES[scheme])
    keys, cms, bins = [], [], []
    for key, group in pred.groupby(by, sort=True):
        y, proba = labels_and_proba(group, scheme)
        keys.append(key)
        cms.append(confusion(y, proba.argmax(axis=1), k))
        bins.append(calibration_bins(y, proba))
    return keys, np.array(cms), np.array(bins)


def per_group_scores(pred: pd.DataFrame, by: list[str], scheme: str = "5") -> pd.DataFrame:
    """One row of scores per group, e.g. per subject or per night."""
    keys, cms, bins = _group_tables(pred, scheme, by)
    rows = [scores_from_tables(cm, b, CLASS_NAMES[scheme]) for cm, b in zip(cms, bins, strict=True)]
    index = (
        pd.MultiIndex.from_tuples(keys, names=by)
        if len(by) > 1
        else pd.Index([k[0] if isinstance(k, tuple) else k for k in keys], name=by[0])
    )
    return pd.DataFrame(rows, index=index)


def summarize(pred: pd.DataFrame, scheme: str = "5", n_boot: int = 1000, seed: int = 0) -> dict:
    """Pooled scores, per-subject mean and SD, and subject-bootstrap 95% CIs of pooled scores."""
    names = CLASS_NAMES[scheme]
    _, cms, bins = _group_tables(pred, scheme, ["subject"])
    pooled = scores_from_tables(cms.sum(axis=0), bins.sum(axis=0), names)
    per_subject = pd.DataFrame(
        [scores_from_tables(cm, b, names) for cm, b in zip(cms, bins, strict=True)]
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cms), size=(n_boot, len(cms)))
    boot = pd.DataFrame(
        [scores_from_tables(cms[d].sum(axis=0), bins[d].sum(axis=0), names) for d in draws]
    )
    headline = ["accuracy", "macro_f1", "kappa", "ece"]
    return {
        "subjects": len(cms),
        "nights": int(pred[["subject", "night"]].drop_duplicates().shape[0]),
        "pooled": pooled,
        "subject_mean": per_subject[headline].mean().to_dict(),
        "subject_sd": per_subject[headline].std().to_dict(),
        "ci95": {m: boot[m].quantile([0.025, 0.975]).tolist() for m in headline},
    }


def paired_difference(
    pred_a: pd.DataFrame, pred_b: pd.DataFrame, scheme: str = "5", n_boot: int = 1000, seed: int = 0
) -> dict:
    """Pooled score of ``b`` minus ``a`` on the same subjects, with a paired subject bootstrap.

    Both frames must cover the same subjects (e.g. the same evaluation nights scored by two
    models); each bootstrap draw resamples subjects once and scores both models on the draw.
    """
    names = CLASS_NAMES[scheme]
    keys_a, cms_a, bins_a = _group_tables(pred_a, scheme, ["subject"])
    keys_b, cms_b, bins_b = _group_tables(pred_b, scheme, ["subject"])
    if keys_a != keys_b:
        raise ValueError("paired comparison needs the same subjects in both predictions")
    metrics = ["accuracy", "macro_f1", "kappa"]

    def diff(idx):
        a = scores_from_tables(cms_a[idx].sum(axis=0), bins_a[idx].sum(axis=0), names)
        b = scores_from_tables(cms_b[idx].sum(axis=0), bins_b[idx].sum(axis=0), names)
        return {m: b[m] - a[m] for m in metrics}

    point = diff(np.arange(len(cms_a)))
    draws = np.random.default_rng(seed).integers(0, len(cms_a), size=(n_boot, len(cms_a)))
    boot = pd.DataFrame([diff(d) for d in draws])
    return {
        m: {"diff": point[m], "ci95": boot[m].quantile([0.025, 0.975]).tolist()} for m in metrics
    }
