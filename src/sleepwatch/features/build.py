"""Build the epoch table for all verified nights, plus a per-night data-quality table."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from sleepwatch.constants import UNKNOWN
from sleepwatch.data.loader import load_night
from sleepwatch.features.epoch_features import FeatureConfig, night_features
from sleepwatch.provenance import git_revision, timestamp


def _one_night(key: tuple[str, int], raw_dir: Path, cfg: FeatureConfig):
    night = load_night(raw_dir, *key)
    features = night_features(night, cfg)
    labeled = night.expert != UNKNOWN
    quality = {
        **night.quality,
        "onset_epoch": features.attrs["onset_epoch"],
        "hr_coverage": float(features["qc_hr_coverage"].mean()),
        "motion_coverage": float(features["qc_motion_coverage"].mean()),
        "dreem_expert_agreement": (
            float((night.dreem == night.expert)[labeled].mean()) if labeled.any() else np.nan
        ),
    }
    return features, quality


def add_chronology(quality: pd.DataFrame) -> pd.DataFrame:
    """Flag subjects whose recording starts don't increase with the night number."""
    ordered = (
        quality.sort_values(["subject", "night"])
        .groupby("subject")["rec_start"]
        .apply(lambda s: bool(s.is_monotonic_increasing and s.is_unique))
    )
    return quality.assign(chronological=quality["subject"].map(ordered))


def build_epochs(
    raw_dir: Path, nights: list[tuple[str, int]], cfg: FeatureConfig, jobs: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features for every night (one row per epoch) and one quality row per night."""
    work = partial(_one_night, raw_dir=Path(raw_dir), cfg=cfg)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(work, nights))
    else:
        results = [work(key) for key in nights]
    epochs = pd.concat([features for features, _ in results], ignore_index=True)
    quality = add_chronology(pd.DataFrame([q for _, q in results]))
    return epochs, quality


def load_build(name: str, processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(epochs, quality, build_info)`` written by :func:`save_build`."""
    return (
        pd.read_parquet(processed_dir / f"epochs_{name}.parquet"),
        pd.read_csv(processed_dir / f"quality_{name}.csv"),
        json.loads((processed_dir / f"build_{name}.json").read_text()),
    )


def save_build(
    epochs: pd.DataFrame, quality: pd.DataFrame, cfg: FeatureConfig, out_dir: Path, extra: dict
) -> dict[str, Path]:
    """Write ``epochs_<name>.parquet``, ``quality_<name>.csv`` and ``build_<name>.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "epochs": out_dir / f"epochs_{cfg.name}.parquet",
        "quality": out_dir / f"quality_{cfg.name}.csv",
        "build": out_dir / f"build_{cfg.name}.json",
    }
    epochs.to_parquet(paths["epochs"], index=False)
    quality.to_csv(paths["quality"], index=False)
    info = {
        "created": timestamp(),
        "git": git_revision(),
        "config": cfg.model_dump(),
        "subjects": int(epochs["subject"].nunique()),
        "nights": len(quality),
        "epochs": len(epochs),
        **extra,
    }
    paths["build"].write_text(json.dumps(info, indent=2, default=str))
    return paths
