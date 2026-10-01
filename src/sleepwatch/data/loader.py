"""Load one night of raw data, cropped to the span covered by its labels.

Only nights that :func:`sleepwatch.data.manifest.verified_nights` returns should be loaded. Raw
signal files routinely start after ``recStart``, end early, or run on for days; everything outside
``[recStart, recStart + 30 s x n_epochs)`` is dropped here and gaps are handled downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
from scipy.io import loadmat

from sleepwatch.constants import (
    EPOCH_S,
    HR_FILE,
    LABELS_FILE,
    MOTION_FILE,
    RECSTART_TZ,
    STAGE_NAMES,
    UNKNOWN,
)
from sleepwatch.data.align import crop

MOTION_HEADER = ["Timestamp", "x", "y", "z"]


@dataclass
class Night:
    subject: str
    night: int
    rec_start: float  # Unix seconds at the start of epoch 0
    rec_start_local: str  # as stored in labels.mat (US Eastern wall clock)
    expert: np.ndarray  # uint8 stage per epoch; ground truth
    dreem: np.ndarray  # uint8 stage per epoch, cut or padded (Unknown) to len(expert)
    hr: pd.DataFrame  # t, hr: cropped, sorted, duplicate timestamps averaged
    motion: pd.DataFrame  # t, x, y, z (g): cropped, sorted, duplicate timestamps dropped
    quality: dict = field(default_factory=dict)

    @property
    def n_epochs(self) -> int:
        return len(self.expert)

    @property
    def end(self) -> float:
        return self.rec_start + EPOCH_S * self.n_epochs


def rec_start_to_unix(local: str, near: float | None = None) -> tuple[float, bool]:
    """Convert a ``recStart`` wall-clock string (US Eastern) to Unix seconds.

    A wall-clock time inside a DST change is ambiguous (autumn) or doesn't exist (spring). Both
    readings are then candidates, and the one closest to ``near`` (the first signal timestamp)
    wins. Returns ``(unix_seconds, was_ambiguous)``.
    """
    naive = datetime.strptime(local.strip(), "%Y-%m-%d %H:%M:%S")
    tz = ZoneInfo(RECSTART_TZ)
    candidates = sorted({naive.replace(tzinfo=tz, fold=fold).timestamp() for fold in (0, 1)})
    if len(candidates) == 1:
        return candidates[0], False
    if near is None:
        raise ValueError(f"recStart {local!r} falls in a DST change; pass `near` to resolve it")
    return min(candidates, key=lambda c: abs(c - near)), True


def read_labels(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    """``(recStart string, expert_label, dreem_label)`` from labels.mat."""
    mat = loadmat(path)
    rec_start = str(np.ravel(mat["recStart"])[0])
    expert = np.ravel(mat["expert_label"]).astype(np.uint8)
    dreem = np.ravel(mat["dreem_label"]).astype(np.uint8)
    for name, labels in (("expert_label", expert), ("dreem_label", dreem)):
        if labels.size and labels.max() >= len(STAGE_NAMES):
            raise ValueError(f"{path}: {name} has codes outside 0-{len(STAGE_NAMES) - 1}")
    return rec_start, expert, dreem


def _read_csv(path: Path, column_names: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """Read a CSV with pyarrow, skipping rows with the wrong number of fields.

    Some files end in a partial record written as the recording stopped. Those rows are skipped
    and counted rather than failing the night or being guessed at.
    """
    skipped = 0

    def skip(row: pacsv.InvalidRow) -> str:
        nonlocal skipped
        skipped += 1
        return "skip"

    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=column_names),
        parse_options=pacsv.ParseOptions(invalid_row_handler=skip),
    )
    return table.to_pandas(), skipped


def read_hr(path: Path) -> tuple[pd.DataFrame, int, int]:
    """Heart rate sorted by time with duplicate timestamps averaged.

    Returns ``(hr, duplicate_count, malformed_row_count)``.
    """
    hr, malformed = _read_csv(path, column_names=["t", "hr"])
    if not all(pd.api.types.is_numeric_dtype(hr[c]) for c in hr):
        raise ValueError(f"{path}: expected two numeric columns (unix time, bpm) and no header")
    hr = hr.dropna()
    n_dup = int(hr["t"].duplicated().sum())
    hr = hr.groupby("t", as_index=False)["hr"].mean() if n_dup else hr.sort_values("t")
    return hr.reset_index(drop=True), n_dup, malformed


def read_motion(path: Path) -> tuple[pd.DataFrame, int, int]:
    """Accelerometry sorted by time with duplicate timestamps dropped.

    Returns ``(motion, duplicate_count, malformed_row_count)``.
    """
    motion, malformed = _read_csv(path)
    if list(motion.columns) != MOTION_HEADER:
        raise ValueError(f"{path}: expected header {MOTION_HEADER}, got {list(motion.columns)}")
    motion.columns = ["t", "x", "y", "z"]
    motion = motion.dropna().astype({"x": np.float32, "y": np.float32, "z": np.float32})
    if not motion["t"].is_monotonic_increasing:
        motion = motion.sort_values("t", kind="stable")
    dup = motion["t"].duplicated()
    return motion[~dup].reset_index(drop=True), int(dup.sum()), malformed


def _offsets(t: pd.Series, start: float, end: float) -> dict:
    if t.empty:
        return {"start_offset_min": np.nan, "end_offset_min": np.nan, "median_dt_s": np.nan}
    return {
        "start_offset_min": (t.iloc[0] - start) / 60,
        "end_offset_min": (t.iloc[-1] - end) / 60,
        "median_dt_s": float(np.median(np.diff(t))) if len(t) > 1 else np.nan,
    }


def load_night(raw_dir: Path, subject: str, night: int) -> Night:
    folder = Path(raw_dir) / subject / str(night)
    rec_local, expert, dreem_raw = read_labels(folder / LABELS_FILE)
    hr_raw, hr_dup, hr_bad = read_hr(folder / HR_FILE)
    motion_raw, motion_dup, motion_bad = read_motion(folder / MOTION_FILE)

    firsts = [s["t"].iloc[0] for s in (hr_raw, motion_raw) if not s.empty]
    rec_start, ambiguous = rec_start_to_unix(rec_local, near=min(firsts) if firsts else None)
    n = len(expert)
    end = rec_start + EPOCH_S * n

    # Dreem and expert labels both start at recStart; the dreem vector can be longer or shorter.
    dreem = np.full(n, UNKNOWN, dtype=np.uint8)
    dreem[: min(n, len(dreem_raw))] = dreem_raw[:n]

    hr, motion = crop(hr_raw, rec_start, end), crop(motion_raw, rec_start, end)
    quality = {
        "subject": subject,
        "night": night,
        "rec_start_local": rec_local,
        "rec_start": rec_start,
        "dst_ambiguous": ambiguous,
        "n_epochs": n,
        "n_dreem_raw": len(dreem_raw),
        "n_unknown": int((expert == UNKNOWN).sum()),
        "hr_rows_raw": len(hr_raw),
        "hr_rows": len(hr),
        "hr_duplicate_ts": hr_dup,
        "hr_malformed_rows": hr_bad,
        **{f"hr_{k}": v for k, v in _offsets(hr_raw["t"], rec_start, end).items()},
        "motion_rows_raw": len(motion_raw),
        "motion_rows": len(motion),
        "motion_duplicate_ts": motion_dup,
        "motion_malformed_rows": motion_bad,
        **{f"motion_{k}": v for k, v in _offsets(motion_raw["t"], rec_start, end).items()},
    }
    return Night(subject, night, rec_start, rec_local, expert, dreem, hr, motion, quality)
