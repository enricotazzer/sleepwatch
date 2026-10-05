"""Phase 3 experiment: personal baselines, injected anomalies and calibrated alarms.

Per outer fold (the fixed subject folds of Phase 2):

1. **Stagers.** The population GRU is trained on the training subjects exactly as in Phase 2
   and predicts the test subjects. Five inner GRUs, each trained on four fifths of the training
   subjects, give every training subject out-of-sample predicted stages too.
2. **Injections.** Each scored test night is reloaded from the raw files; every injected
   version is rebuilt with the unchanged feature pipeline and staged by the fold's GRU.
3. **Scoring.** For each stage source (GRU, expert, none) and baseline (personal, population
   only), the training subjects' clean nights are scored out of sample (inner folds) to give
   the null; alarm thresholds come from that null. Test nights (clean and injected) are then
   scored against priors from all training subjects and baselines from their own earlier nights.

Nothing is fitted on test subjects. Results go to ``results/<name>/<timestamp>/``.
"""

from __future__ import annotations

import json
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict

from sleepwatch.anomaly import baseline as bl
from sleepwatch.anomaly import scores as sc
from sleepwatch.anomaly.inject import InjectionSettings, versions
from sleepwatch.anomaly.summary import AnomalySummary, ChannelResult, clock
from sleepwatch.constants import STAGE_NAMES, UNKNOWN
from sleepwatch.features.epoch_features import FeatureConfig, feature_columns, night_features
from sleepwatch.models import gru as gru_model
from sleepwatch.models.data import prepare
from sleepwatch.models.experiment import StagingConfig, _fit_predict
from sleepwatch.models.gru import GRUConfig
from sleepwatch.models.splits import inner_split, outer_splits
from sleepwatch.provenance import git_revision, timestamp

STAGE_SOURCES = ("gru", "expert", "none")
BASELINES = ("personal", "population")
PRIMARY = ("gru", "personal")
KEEP = [
    "subject",
    "night",
    "epoch",
    "t_start",
    "expert",
    "hr_mean",
    "activity",
    "hours_since_onset",
    "hours_since_start",
    "qc_hr_coverage",
    "qc_motion_coverage",
]


class AnomalyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    features: str = "v1"
    seed: int = 42
    folds: list[int] | None = None  # a subset only for quick checks; recorded as partial
    exclude: list[tuple[str, int]] = [("Bidslab42", 1), ("Bidslab68", 2)]
    min_baseline_nights: int = 2
    inner_folds: int = 5
    target_fpr: float = 0.05
    prevalence: float = 0.10  # assumed share of anomalous nights for the derived precision
    n_boot: int = 1000
    val_fraction: float = 0.2
    channels: sc.ChannelSettings = sc.ChannelSettings()
    injections: InjectionSettings = InjectionSettings()
    gru: GRUConfig = GRUConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> AnomalyConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


def night_table(epochs: pd.DataFrame, cfg: AnomalyConfig) -> pd.DataFrame:
    """One row per night: rank by date, usable, number of earlier usable nights, scored."""
    nights = epochs.groupby(["subject", "night"], as_index=False)["night_rank"].first()
    excluded = set(map(tuple, cfg.exclude))
    nights["usable"] = [
        (s, n) not in excluded for s, n in zip(nights["subject"], nights["night"], strict=True)
    ]
    nights = nights.sort_values(["subject", "night_rank"]).reset_index(drop=True)
    earlier = nights.groupby("subject")["usable"].cumsum() - nights["usable"].astype(int)
    nights["n_baseline"] = earlier.astype(int)
    nights["scored"] = nights["usable"] & (nights["n_baseline"] >= cfg.min_baseline_nights)
    return nights


# --- 1. stagers --------------------------------------------------------------------------------


def _inner_groups(train_s: list[str], k: int, seed: int) -> list[list[str]]:
    order = np.random.default_rng(seed).permutation(sorted(train_s))
    return [sorted(g.tolist()) for g in np.array_split(order, k)]


def _stagers(cfg: AnomalyConfig, data: pd.DataFrame, base_cols, fold, train_s, test_s):
    """Outer GRU (as Phase 2's ``gru_main`` population model) and inner out-of-sample stages."""
    staging = StagingConfig(name="phase3_stager", model="gru", gru=cfg.gru, seed=cfg.seed)

    def train_predict(train_subjects, target_subjects, seed):
        fit_s, val_s = inner_split(train_subjects, cfg.val_fraction, seed)
        train_rows = data[data["subject"].isin(train_subjects)]
        target_rows = data[data["subject"].isin(target_subjects)]
        proba, _, fitted = _fit_predict(
            staging, train_rows, target_rows, base_cols, fit_s, val_s, seed, None
        )
        return pd.Series(proba.argmax(axis=1), index=target_rows.index), fitted

    seed = cfg.seed + fold
    test_stage, outer = train_predict(train_s, test_s, seed)
    groups = _inner_groups(train_s, cfg.inner_folds, seed)
    inner = [
        train_predict(sorted(set(train_s) - set(group)), group, seed * 100 + k)[0]
        for k, group in enumerate(groups)
    ]
    return {
        "fold": fold,
        "test_stage": test_stage,
        "train_stage": pd.concat(inner),
        "groups": groups,
        "model": outer,
    }


# --- 2. injections -----------------------------------------------------------------------------


def night_seed(seed: int, subject: str, night: int) -> int:
    return seed * 1_000_003 + zlib.crc32(f"{subject}/{night}".encode()) % 1_000_000


def _inject_night(cfg, feature_cfg, loader, key, model, stored: pd.DataFrame):
    """Clean and injected versions of one night: detector inputs, GRU stages and ground truth."""
    net, prep = model
    night = loader(*key)
    clean = night_features(night, feature_cfg)
    check = {
        "subject": key[0],
        "night": key[1],
        "features_equal": bool(
            np.allclose(clean["hr_mean"], stored["hr_mean"], equal_nan=True)
            and np.allclose(clean["activity"], stored["activity"], equal_nan=True)
        ),
    }
    frames, truth = [], []
    todo = [("clean", None, None, None, [], night)]
    todo += versions(night, clean, cfg.injections, night_seed(cfg.seed, *key))
    for version, kind, size, duration, windows, injected in todo:
        features = clean if version == "clean" else night_features(injected, feature_cfg)
        proba = gru_model.predict(net, gru_model.night_arrays(features, prep, None))[0]
        frames.append(features[KEEP].assign(version=version, gru=proba.argmax(axis=1)))
        if kind is not None:
            truth.append(
                {
                    "subject": key[0],
                    "night": key[1],
                    "version": version,
                    "kind": kind,
                    "size": size,
                    "duration_min": duration,
                    "windows": windows,
                }
            )
    return frames, truth, check


# --- 3. scoring --------------------------------------------------------------------------------


def _inputs(rows: pd.DataFrame, source: str, stage_column: str = "gru") -> pd.DataFrame:
    if source == "gru":
        stage = rows[stage_column].to_numpy(dtype=float)
    elif source == "expert":
        stage = np.where(rows["expert"] == UNKNOWN, np.nan, rows["expert"]).astype(float)
    else:
        stage = np.zeros(len(rows))
    out = bl.detector_inputs(rows, stage)
    out["version"] = rows["version"].to_numpy() if "version" in rows else "clean"
    out.index = rows.index
    return out


def _night_values(rows: pd.DataFrame, prior: bl.Prior, staged: bool, settings) -> pd.DataFrame:
    """Raw frag and onset values per clean night (they don't depend on personal offsets)."""
    out = []
    for (subject, night), group in rows.groupby(["subject", "night"], sort=False):
        z = bl.score_epochs(group, prior, None)
        ch = sc.night_channels(group.assign(z_hr=z["z_hr"], z_act_pop=z["z_act"]), staged, settings)
        out.append({"subject": subject, "night": night, "frag": ch["frag"], "onset": ch["onset"]})
    return pd.DataFrame(out).set_index(["subject", "night"])


def _score(targets, history, nights, prior, night_prior, staged, settings) -> pd.DataFrame:
    """Channel values of each target night version, for both baseline variants.

    ``history`` holds clean usable nights (the baselines); a target night's baseline is its
    subject's usable nights with a lower rank.
    """
    summaries = bl.night_summaries(history, prior)
    values = _night_values(history, prior, staged, settings)
    rank = nights.set_index(["subject", "night"])
    out = []
    for (subject, night, version), group in targets.groupby(
        ["subject", "night", "version"], sort=False
    ):
        group = group.sort_values("epoch")
        this = rank.loc[(subject, night), "night_rank"]
        mine = rank.loc[subject]
        earlier = [(subject, n) for n in mine.index[(mine["night_rank"] < this) & mine["usable"]]]
        earlier = [k for k in earlier if k in summaries.index]
        population_z = bl.score_epochs(group, prior, None)
        for variant in BASELINES:
            personal = variant == "personal"
            offsets = bl.personal_offsets(summaries.loc[earlier], prior) if personal else None
            z = population_z if not personal else bl.score_epochs(group, prior, offsets)
            ch = sc.night_channels(
                group.assign(z_hr=z["z_hr"], z_act_pop=population_z["z_act"]), staged, settings
            )
            row = {
                "subject": subject,
                "night": night,
                "version": version,
                "baseline": variant,
                "n_baseline": len(earlier),
                "hr_window": ch["hr_window"],
                "hr_night": ch["hr_night"],
                "frag_raw": ch["frag"],
                "onset_raw": ch["onset"],
                "sleep_onset": ch["sleep_onset"],
                "sleep_end": ch["sleep_end"],
                "window": ch["window"],
                "bouts": ch["bouts"],
            }
            for channel in sc.NIGHT_LEVEL:
                past = values.loc[earlier, channel] if earlier else pd.Series(dtype=float)
                row[channel] = sc.night_z(ch[channel], past, night_prior, channel, personal)
            if ch["window"] is not None:
                a, b = ch["window"]
                excess = group["hr"].to_numpy()[a:b] - z["expected_hr"].to_numpy()[a:b]
                finite = np.isfinite(excess).any()
                row["hr_excess_bpm"] = float(np.nanmean(excess)) if finite else np.nan
                stages = group["stage"].to_numpy()[a:b]
                row["window_stages"] = {
                    STAGE_NAMES[int(s)]: float(np.mean(stages == s))
                    for s in np.unique(stages[~np.isnan(stages)])
                }
            out.append(row)
    return pd.DataFrame(out)


def _in_nights(rows: pd.DataFrame, nights: pd.DataFrame) -> np.ndarray:
    keys = pd.MultiIndex.from_frame(nights[["subject", "night"]])
    return pd.MultiIndex.from_frame(rows[["subject", "night"]]).isin(keys)


def _fold_scores(cfg, fold_info, data, nights, injected, source):
    """Null (training subjects, inner folds) and test scores for one fold and stage source."""
    staged = source != "none"
    settings = cfg.channels
    train_s = sorted(set().union(*fold_info["groups"]))
    clean = data[_in_nights(data, nights[nights["usable"]])]
    train_rows = clean[clean["subject"].isin(train_s)].assign(gru=fold_info["train_stage"])
    train_in = _inputs(train_rows, source)
    scored = nights[nights["scored"]]

    null = []
    for group in fold_info["groups"]:
        rest = train_in[~train_in["subject"].isin(group)]
        prior = bl.fit_prior(rest)
        night_prior = sc.fit_night_prior(_night_values(rest, prior, staged, settings).reset_index())
        mine = train_in[train_in["subject"].isin(group)]
        targets = mine[_in_nights(mine, scored)]
        null.append(_score(targets, mine, nights, prior, night_prior, staged, settings))
    null = pd.concat(null, ignore_index=True)

    prior = bl.fit_prior(train_in)
    night_prior = sc.fit_night_prior(_night_values(train_in, prior, staged, settings).reset_index())
    test_s = sorted(set(data["subject"]) - set(train_s))
    test_rows = clean[clean["subject"].isin(test_s)].assign(gru=fold_info["test_stage"])
    history = _inputs(test_rows, source)
    targets = _inputs(injected, source)
    test = _score(targets, history, nights, prior, night_prior, staged, settings)

    results = []
    for variant in BASELINES:
        calibration = sc.calibrate(null[null["baseline"] == variant], cfg.target_fpr)
        part = test[test["baseline"] == variant].reset_index(drop=True)
        results.append(
            pd.concat([part, sc.evaluate(part, calibration)], axis=1).assign(
                threshold=calibration.threshold,
                null_rate=calibration.null_rate,
                null_nights=len(calibration.null),
            )
        )
    return (
        pd.concat(results, ignore_index=True).assign(fold=fold_info["fold"], source=source),
        null.assign(fold=fold_info["fold"], source=source),
    )


# --- 4. metrics --------------------------------------------------------------------------------


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC with ties counted half."""
    if not len(pos) or not len(neg):
        return np.nan
    neg = np.sort(neg)
    below = np.searchsorted(neg, pos, side="left")
    equal = np.searchsorted(neg, pos, side="right") - below
    return float((below + 0.5 * equal).sum() / (len(pos) * len(neg)))


def _ci(values: np.ndarray) -> list[float]:
    values = values[np.isfinite(values)]
    return np.quantile(values, [0.025, 0.975]).tolist() if len(values) else [np.nan, np.nan]


def _windows_hit(row, truth_windows) -> bool:
    if row["window"] is None:
        return False
    centre = (row["window"][0] + row["window"][1]) / 2
    return any(a <= centre < b for a, b in truth_windows)


def _overlap_share(row, truth_windows) -> float:
    bouts = row["bouts"] or []
    hit = [any(a < d and c < b for c, d in bouts) for a, b in truth_windows]
    return float(np.mean(hit)) if hit else np.nan


def compute_metrics(results: pd.DataFrame, truth: pd.DataFrame, cfg: AnomalyConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    subjects = np.array(sorted(results["subject"].unique()))
    draws = rng.integers(0, len(subjects), size=(cfg.n_boot, len(subjects)))
    truth = truth.set_index(["subject", "night", "version"])
    conditions = truth.reset_index().drop_duplicates("version").set_index("version")
    out = {"variants": {}, "comparisons": {}, "by_baseline_nights": {}}
    per_subject_recall = {}
    for (source, variant), rows in results.groupby(["source", "baseline"]):
        name = f"{source}/{variant}"
        clean = rows[rows["version"] == "clean"]
        by_subject = {s: g for s, g in rows.groupby("subject")}
        n_clean = np.array(
            [
                (by_subject[s]["version"] == "clean").sum() if s in by_subject else 0
                for s in subjects
            ]
        )
        f_clean = np.array(
            [
                by_subject[s].query("version == 'clean'")["flagged"].sum() if s in by_subject else 0
                for s in subjects
            ]
        )
        boot_fpr = f_clean[draws].sum(1) / np.maximum(n_clean[draws].sum(1), 1)
        fpr = float(clean["flagged"].mean())
        entry = {
            "false_alarm_rate": {"value": fpr, "ci95": _ci(boot_fpr), "nights": len(clean)},
            "conditions": {},
        }
        neg_by_subject = [
            by_subject[s].query("version == 'clean'")["score"].to_numpy()
            if s in by_subject
            else np.array([])
            for s in subjects
        ]
        for version, cond in conditions.iterrows():
            inj = rows[rows["version"] == version]
            if inj.empty:
                continue
            n_inj = np.array([(inj["subject"] == s).sum() for s in subjects])
            n_hit = np.array([inj.loc[inj["subject"] == s, "flagged"].sum() for s in subjects])
            per_subject_recall[(name, version)] = (n_inj, n_hit)
            recall = float(inj["flagged"].mean())
            boot_recall = n_hit[draws].sum(1) / np.maximum(n_inj[draws].sum(1), 1)
            pos_by_subject = [inj.loc[inj["subject"] == s, "score"].to_numpy() for s in subjects]
            auroc = _auc(inj["score"].to_numpy(), clean["score"].to_numpy())
            boot_auc = np.array(
                [
                    _auc(
                        np.concatenate([pos_by_subject[i] for i in d]),
                        np.concatenate([neg_by_subject[i] for i in d]),
                    )
                    for d in draws
                ]
            )
            precision = (
                recall * cfg.prevalence / (recall * cfg.prevalence + fpr * (1 - cfg.prevalence))
                if recall + fpr > 0
                else np.nan
            )
            detected = inj[inj["flagged"]]
            matching = sc.MATCHING[cond["kind"]]
            item = {
                "kind": cond["kind"],
                "size": float(cond["size"]),
                "duration_min": None
                if pd.isna(cond["duration_min"])
                else int(cond["duration_min"]),
                "nights": len(inj),
                "recall": recall,
                "recall_ci95": _ci(boot_recall),
                "auroc": auroc,
                "auroc_ci95": _ci(boot_auc),
                "precision_at_prevalence": precision,
                "type_correct": float(detected["top_channel"].isin(matching).mean())
                if len(detected)
                else np.nan,
            }
            windows = [
                truth.loc[(s, n, version), "windows"]
                for s, n in zip(inj["subject"], inj["night"], strict=True)
            ]
            if cond["kind"] == "hr" and cond["duration_min"]:
                hits = [
                    _windows_hit(r, w)
                    for (_, r), w in zip(inj.iterrows(), windows, strict=True)
                    if r["flagged"]
                ]
                item["localized"] = float(np.mean(hits)) if hits else np.nan
            if cond["kind"] == "frag":
                item["awakenings_seen"] = float(
                    np.nanmean(
                        [
                            _overlap_share(r, w)
                            for (_, r), w in zip(inj.iterrows(), windows, strict=True)
                        ]
                    )
                )
            entry["conditions"][version] = item
        out["variants"][name] = entry
        if (source, variant) == PRIMARY:
            for label, mask in (
                ("2", rows["n_baseline"] == 2),
                ("3", rows["n_baseline"] == 3),
                ("4+", rows["n_baseline"] >= 4),
            ):
                part = rows[mask & (rows["version"] != "clean")]
                out["by_baseline_nights"][label] = {
                    "nights": int(part[["subject", "night"]].drop_duplicates().shape[0]),
                    "recall": part.groupby("version")["flagged"].mean().to_dict(),
                    "false_alarm_rate": float(
                        rows[mask & (rows["version"] == "clean")]["flagged"].mean()
                    ),
                }
    primary = f"{PRIMARY[0]}/{PRIMARY[1]}"
    for other in ("gru/population", "expert/personal", "none/personal"):
        comparison = {}
        for version in conditions.index:
            a, b = (
                per_subject_recall.get((other, version)),
                per_subject_recall.get((primary, version)),
            )
            if a is None or b is None:
                continue
            diff = b[1].sum() / b[0].sum() - a[1].sum() / a[0].sum()
            boot = b[1][draws].sum(1) / np.maximum(b[0][draws].sum(1), 1) - a[1][draws].sum(
                1
            ) / np.maximum(a[0][draws].sum(1), 1)
            comparison[version] = {"recall_difference": float(diff), "ci95": _ci(boot)}
        out["comparisons"][f"{primary} minus {other}"] = comparison
    return out


# --- 5. summaries of real flagged nights -------------------------------------------------------

KNOWN_ISSUES = {("Bidslab06", 2): "interleaved heart-rate streams (docs/data.md)"}


def summaries(results: pd.DataFrame, inputs: pd.DataFrame, quality: pd.DataFrame) -> list[dict]:
    """``AnomalySummary`` for each clean test night the primary detector flags."""
    rows = results[
        (results["source"] == PRIMARY[0])
        & (results["baseline"] == PRIMARY[1])
        & (results["version"] == "clean")
        & results["flagged"]
    ]
    quality = quality.set_index(["subject", "night"])
    out = []
    for _, r in rows.iterrows():
        night = inputs[
            (inputs["subject"] == r["subject"])
            & (inputs["night"] == r["night"])
            & (inputs["version"] == "clean")
        ].sort_values("epoch")
        start_local = quality.loc[(r["subject"], r["night"]), "rec_start_local"]
        channels = [
            ChannelResult(
                channel=c,
                value=float(r[c]) if pd.notna(r[c]) else float("nan"),
                tail_probability=float(r[f"p_{c}"]),
                fired=bool(r[f"p_{c}"] <= r["threshold"]),
            )
            for c in sc.CHANNELS
        ]
        flags = (
            [KNOWN_ISSUES[(r["subject"], r["night"])]]
            if (r["subject"], r["night"]) in KNOWN_ISSUES
            else []
        )
        window = r["window"]
        if window is not None:
            a, b = window
            part = night.iloc[a:b]
            if part["qc_hr_coverage"].mean() < 0.9:
                flags.append("heart-rate coverage under 90% in the window")
            if part["qc_motion_coverage"].mean() < 0.9:
                flags.append("motion coverage under 90% in the window")
        summary = AnomalySummary(
            subject=r["subject"],
            night=int(r["night"]),
            date_local=start_local[:10],
            baseline_nights=int(r["n_baseline"]),
            top_channel=r["top_channel"],
            channels=channels,
            window_start_local=clock(start_local, window[0]) if window else None,
            window_end_local=clock(start_local, window[1]) if window else None,
            window_hours_since_onset=float(night["hours_since_onset"].iloc[window[0]])
            if window
            else None,
            window_stage_share=r["window_stages"]
            if isinstance(r.get("window_stages"), dict)
            else {},
            hr_excess_bpm=float(r["hr_excess_bpm"]) if pd.notna(r.get("hr_excess_bpm")) else None,
            wake_bouts_per_hour=float(r["frag_raw"]) if pd.notna(r["frag_raw"]) else None,
            onset_latency_min=float(r["onset_raw"]) if pd.notna(r["onset_raw"]) else None,
            data_quality=flags,
        )
        out.append(summary.model_dump())
    return out


# --- 6. run and save ---------------------------------------------------------------------------


def run(
    cfg: AnomalyConfig,
    epochs: pd.DataFrame,
    folds: dict[str, int],
    loader,
    feature_cfg: FeatureConfig,
    jobs: int = 1,
    log=print,
) -> dict:
    """The whole experiment; ``loader(subject, night)`` returns a raw
    :class:`~sleepwatch.data.loader.Night` (e.g. ``partial(load_night, raw_dir)``)."""
    base_cols = feature_columns(epochs)
    data = prepare(epochs)
    data["_target"] = data["expert"]
    data["_truth"] = data["expert"]
    nights = night_table(data, cfg)
    splits = [
        (f, tr, te)
        for f, (tr, te) in enumerate(outer_splits(folds))
        if cfg.folds is None or f in cfg.folds
    ]
    log(f"{int(nights['scored'].sum())} scored nights; training stagers for {len(splits)} folds")
    stagers = Parallel(n_jobs=min(jobs, len(splits)))(
        delayed(_stagers)(cfg, data, base_cols, *split) for split in splits
    )
    tasks = []
    for info, (_, _, test_s) in zip(stagers, splits, strict=True):
        keys = nights[nights["scored"] & nights["subject"].isin(test_s)][["subject", "night"]]
        for subject, night in keys.itertuples(index=False):
            stored = data[(data["subject"] == subject) & (data["night"] == night)]
            tasks.append(((subject, int(night)), info["model"], stored[["hr_mean", "activity"]]))
    log(f"injecting into {len(tasks)} test nights")
    frames, truth, checks = [], [], []
    done = Parallel(n_jobs=jobs, return_as="generator")(
        delayed(_inject_night)(cfg, feature_cfg, loader, key, model, stored)
        for key, model, stored in tasks
    )
    for i, (f, t, c) in enumerate(done, 1):
        frames += f
        truth += t
        checks.append(c)
        if i % 25 == 0:
            log(f"  {i}/{len(tasks)} nights")
    injected = pd.concat(frames, ignore_index=True)
    stages = []
    for info in stagers:
        for role, series in (("test", info["test_stage"]), ("train", info["train_stage"])):
            rows = data.loc[series.index, ["subject", "night", "epoch"]]
            stages.append(rows.assign(fold=info["fold"], role=role, gru=series.to_numpy()))
    stages = pd.concat(stages, ignore_index=True)
    clean_check = injected[injected["version"] == "clean"].merge(
        stages[stages["role"] == "test"], on=["subject", "night", "epoch"], suffixes=("", "_outer")
    )
    for c in checks:
        mine = clean_check[
            (clean_check["subject"] == c["subject"]) & (clean_check["night"] == c["night"])
        ]
        c["stages_equal"] = bool((mine["gru"] == mine["gru_outer"]).all())
    results, nulls = [], []
    for info, (_, _, test_s) in zip(stagers, splits, strict=True):
        mine = injected[injected["subject"].isin(test_s)]
        for source in STAGE_SOURCES:
            r, n = _fold_scores(cfg, info, data, nights, mine, source)
            results.append(r)
            nulls.append(n)
        log(f"fold {info['fold']} scored")
    results = pd.concat(results, ignore_index=True)
    truth = pd.DataFrame(truth)
    return {
        "results": results,
        "nulls": pd.concat(nulls, ignore_index=True),
        "truth": truth,
        "checks": pd.DataFrame(checks),
        "stages": stages,
        "injected": injected,
        "nights": nights,
        "folds": [{"fold": i["fold"], "groups": i["groups"]} for i in stagers],
        "metrics": compute_metrics(results, truth, cfg),
    }


def _jsonable(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if (
            out[column].dtype == object
            and out[column].map(lambda v: isinstance(v, (list, tuple, dict))).any()
        ):
            out[column] = out[column].map(lambda v: json.dumps(v) if v is not None else None)
    return out


def save_run(
    cfg: AnomalyConfig, output: dict, quality: pd.DataFrame, results_dir: Path, extra: dict
) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_dir / cfg.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    )
    git = git_revision()
    partial = cfg.folds is not None
    checks = output["checks"]
    info = {
        "created": timestamp(),
        "git": git,
        "partial": partial,
        "folds": output["folds"],
        "checks": {
            "nights": len(checks),
            "features_equal": int(checks["features_equal"].sum()),
            "stages_equal": int(checks["stages_equal"].sum()),
        },
        **extra,
    }
    (run_dir / "run.json").write_text(json.dumps(info, indent=2, default=str))
    _jsonable(output["results"]).to_parquet(run_dir / "night_scores.parquet", index=False)
    _jsonable(output["nulls"]).to_parquet(run_dir / "null_scores.parquet", index=False)
    _jsonable(output["truth"]).to_parquet(run_dir / "injections.parquet", index=False)
    output["stages"].to_parquet(run_dir / "stages.parquet", index=False)
    output["injected"].to_parquet(run_dir / "epochs.parquet", index=False)
    output["nights"].to_parquet(run_dir / "nights.parquet", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(output["metrics"], indent=2, default=float))
    flagged = summaries(output["results"], output["injected"], quality)
    (run_dir / "flagged_nights.json").write_text(json.dumps(flagged, indent=2, default=float))
    primary = output["metrics"]["variants"][f"{PRIMARY[0]}/{PRIMARY[1]}"]
    row = {
        "name": cfg.name,
        "run": run_id,
        "commit": (git["commit"] or "")[:8],
        "dirty": git["dirty"],
        "partial": partial,
        "false_alarm_rate": round(primary["false_alarm_rate"]["value"], 4),
        "flagged_real": len(flagged),
    }
    index = results_dir / "anomaly_index.csv"
    pd.DataFrame([row]).to_csv(index, mode="a", header=not index.exists(), index=False)
    return run_dir
