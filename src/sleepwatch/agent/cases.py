"""Anomaly cases for the follow-up agent, taken from a saved anomaly run (docs/decisions.md 49).

Test cases are every real night the primary detector flags plus flagged injected versions of
three kinds; dev cases are more injected versions. Every case comes from a different subject, so
dev and test share no subject. The summaries are rebuilt from the run's saved scores with the
same ``summaries`` function that wrote ``flagged_nights.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from sleepwatch.anomaly.experiment import PRIMARY, AnomalyConfig, summaries
from sleepwatch.anomaly.summary import AnomalySummary

Kind = Literal["real", "hr", "frag", "onset"]
# Injected kinds, scarcest first: (version prefix, channels that must be the top channel).
INJECTED: dict[str, tuple[str, tuple[str, ...]]] = {
    "onset": ("onset+60", ("onset",)),
    "frag": ("frag8", ("frag",)),
    "hr": ("hr+10_", ("hr_window", "hr_night")),
}


class AgentCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    split: Literal["dev", "test"]
    kind: Kind
    subject: str
    night: int
    version: str
    summary: AnomalySummary


def select(results: pd.DataFrame, seed: int, n_test: int, n_dev: int) -> pd.DataFrame:
    """Rows of ``results`` to use, with ``kind`` and ``split``: all flagged clean nights (test),
    then per injected kind ``n_test`` test and ``n_dev`` dev versions from unused subjects."""
    flagged = results[
        (results["source"] == PRIMARY[0])
        & (results["baseline"] == PRIMARY[1])
        & results["flagged"].astype(bool)
    ]
    real = flagged[flagged["version"] == "clean"].assign(kind="real", split="test")
    picked = [real]
    used = set(real["subject"])
    rng = np.random.default_rng(seed)
    for kind, (prefix, channels) in INJECTED.items():
        pool = flagged[
            flagged["version"].str.startswith(prefix) & flagged["top_channel"].isin(channels)
        ]
        pool = pool.sort_values(["subject", "night", "version"]).reset_index(drop=True)
        pool = pool.iloc[rng.permutation(len(pool))]
        chosen = []
        for _, row in pool.iterrows():
            if row["subject"] in used:
                continue
            used.add(row["subject"])
            chosen.append(row)
            if len(chosen) == n_test + n_dev:
                break
        if len(chosen) < n_test + n_dev:
            raise ValueError(f"only {len(chosen)} subjects available for {kind}")
        frame = pd.DataFrame(chosen).assign(kind=kind)
        frame["split"] = ["test"] * n_test + ["dev"] * n_dev
        picked.append(frame)
    return pd.concat(picked, ignore_index=True)


def _decode(rows: pd.DataFrame) -> pd.DataFrame:
    """Saved runs store windows and stage shares as JSON text; ``summaries`` wants objects."""
    out = rows.copy()
    out["window"] = out["window"].map(lambda v: tuple(json.loads(v)) if isinstance(v, str) else v)
    out["window_stages"] = out["window_stages"].map(
        lambda v: json.loads(v) if isinstance(v, str) else v
    )
    return out


def build(
    run_dir: Path, quality: pd.DataFrame, seed: int, n_test: int = 5, n_dev: int = 2
) -> list[AgentCase]:
    cfg = AnomalyConfig.model_validate(yaml.safe_load((run_dir / "config.yaml").read_text()))
    results = pd.read_parquet(run_dir / "night_scores.parquet")
    rows = _decode(select(results, seed, n_test, n_dev))
    epochs = pd.read_parquet(run_dir / "epochs.parquet")
    keys = rows[["subject", "night", "version"]]
    epochs = epochs.merge(keys, on=["subject", "night", "version"])
    found = summaries(rows, epochs, quality, cfg.screen, versions=None)
    if len(found) != len(rows):
        raise RuntimeError(f"{len(found)} summaries for {len(rows)} cases")
    cases = []
    for (_, row), summary in zip(rows.iterrows(), found, strict=True):
        if (summary["subject"], summary["night"]) != (row["subject"], row["night"]):
            raise RuntimeError("summaries came back in a different order")
        cases.append(
            AgentCase(
                case_id=f"{row['subject']}-{row['night']}-{row['version']}",
                split=row["split"],
                kind=row["kind"],
                subject=row["subject"],
                night=int(row["night"]),
                version=row["version"],
                summary=AnomalySummary(**summary),
            )
        )
    return cases
