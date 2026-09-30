"""Checksum manifest of the raw dataset.

Raw files are trusted only after checking them against PhysioNet's ``SHA256SUMS.txt``: the first
local download of this dataset was silently incomplete (an interrupted wget mirror whose last file
was NUL-padded). Loaders therefore start from :func:`verified_nights` on a saved manifest, never
from a directory listing. Only paths listed in the checksum file are examined, so stray files such
as macOS ``._*`` resource forks or wget ``index.html`` pages are ignored.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sleepwatch.constants import CHECKSUM_FILE, NIGHT_FILES

MANIFEST_FILE = "manifest.parquet"  # saved under the interim data dir

OK = "ok"  # SHA-256 matches
UNVERIFIED = "unverified"  # present and passes cheap checks, not hashed (fast mode)
CORRUPT = "corrupt"  # SHA-256 mismatch, or (fast mode) empty / NUL-padded text file
MISSING = "missing"
STALE = "stale"  # changed on disk since the manifest was saved

_COLUMNS = ["path", "subject", "night", "file", "status", "size", "mtime_ns"]
_NIGHT_PATH = re.compile(r"^(?P<subject>Bidslab\d+)/(?P<night>\d+)/(?P<file>[^/]+)$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TEXT_SUFFIXES = {".csv", ".txt", ".md"}  # MAT v5 files may legitimately end in zero padding


@dataclass(frozen=True)
class ExpectedFile:
    path: str  # relative to the raw directory
    sha256: str
    subject: str | None  # None for top-level files such as README.md
    night: int | None


def read_checksums(raw_dir: Path) -> list[ExpectedFile]:
    """Parse ``SHA256SUMS.txt`` (``<sha256> <relative path>`` per line) in ``raw_dir``."""
    checksum_path = Path(raw_dir) / CHECKSUM_FILE
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"{checksum_path} not found. Point the raw directory at the dataset folder that "
            f"contains {CHECKSUM_FILE} (and check that the external drive is mounted)."
        )
    entries = []
    for lineno, line in enumerate(checksum_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _SHA256.match(parts[0].lower()):
            raise ValueError(f"{checksum_path}:{lineno}: malformed checksum line {line!r}")
        rel = parts[1].strip().removeprefix("*")  # '*' marks binary mode in sha256sum output
        m = _NIGHT_PATH.match(rel)
        entries.append(
            ExpectedFile(
                path=rel,
                sha256=parts[0].lower(),
                subject=m["subject"] if m else None,
                night=int(m["night"]) if m else None,
            )
        )
    return entries


def _sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _looks_truncated(path: Path, size: int) -> bool:
    """Cheap check for interrupted downloads: an empty file or a NUL-padded text file."""
    if size == 0:
        return True
    if path.suffix not in _TEXT_SUFFIXES:
        return False
    with path.open("rb") as fh:
        fh.seek(max(0, size - 4096))
        return fh.read().endswith(b"\x00")


def verify(
    raw_dir: Path,
    *,
    checksums: bool = True,
    wrap: Callable[[list[ExpectedFile]], Iterable[ExpectedFile]] | None = None,
) -> pd.DataFrame:
    """Check every file listed in ``SHA256SUMS.txt``; one row per listed file.

    ``checksums=False`` skips hashing: files that exist and pass cheap checks come back as
    ``unverified``, so such a manifest never yields verified nights. ``wrap`` can wrap the file
    iterable, e.g. with a progress bar.
    """
    raw_dir = Path(raw_dir)
    entries = read_checksums(raw_dir)
    rows = []
    for entry in wrap(entries) if wrap else entries:
        path = raw_dir / entry.path
        size = mtime_ns = None
        if not path.is_file():
            status = MISSING
        else:
            stat = path.stat()
            size, mtime_ns = stat.st_size, stat.st_mtime_ns
            if checksums:
                status = OK if _sha256(path) == entry.sha256 else CORRUPT
            else:
                status = CORRUPT if _looks_truncated(path, size) else UNVERIFIED
        rows.append(
            {
                "path": entry.path,
                "subject": entry.subject,
                "night": entry.night,
                "file": Path(entry.path).name,
                "status": status,
                "size": size,
                "mtime_ns": mtime_ns,
            }
        )
    df = pd.DataFrame(rows, columns=_COLUMNS)
    # Rebuild the integer columns from the Python ints: letting pandas infer them goes through
    # float64 when values are missing, which rounds nanosecond mtimes and breaks staleness checks.
    for col in ("night", "size", "mtime_ns"):
        df[col] = pd.array([row[col] for row in rows], dtype="Int64")
    df.attrs["raw_dir"] = str(raw_dir.resolve())
    return df


def verified_nights(manifest: pd.DataFrame) -> list[tuple[str, int]]:
    """``(subject, night)`` pairs whose hr.csv, motion.csv and labels.mat all passed SHA-256."""
    files = manifest[manifest["subject"].notna() & manifest["file"].isin(NIGHT_FILES)]
    per_night = (
        files.assign(is_ok=files["status"].eq(OK))
        .groupby(["subject", "night"])
        .agg(n_files=("file", "nunique"), all_ok=("is_ok", "all"))
    )
    good = per_night[(per_night["n_files"] == len(NIGHT_FILES)) & per_night["all_ok"]]
    return sorted((str(subject), int(night)) for subject, night in good.index)


def summarize(manifest: pd.DataFrame) -> dict:
    """Counts per status plus subject- and night-level completeness."""
    files = manifest[manifest["subject"].notna()]
    nights = files[["subject", "night"]].drop_duplicates()
    good = pd.DataFrame(verified_nights(manifest), columns=["subject", "night"])
    n_nights = nights.groupby("subject").size()
    n_good = good.groupby("subject").size().reindex(n_nights.index, fill_value=0)
    return {
        "files": len(manifest),
        "status_counts": manifest["status"].value_counts().to_dict(),
        "subjects": len(n_nights),
        "subjects_complete": int((n_good == n_nights).sum()),
        "incomplete_subjects": sorted(n_nights.index[n_good < n_nights]),
        "nights": len(nights),
        "nights_verified": len(good),
        "corrupt": manifest.loc[manifest["status"] == CORRUPT, "path"].tolist(),
    }


def save_manifest(manifest: pd.DataFrame, path: Path) -> None:
    raw_dir = manifest.attrs.get("raw_dir")
    if raw_dir is None:
        raise ValueError("manifest has no raw_dir attribute; create it with verify()")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.assign(raw_dir=raw_dir).to_parquet(path, index=False)


def load_manifest(path: Path, raw_dir: Path) -> pd.DataFrame:
    """Load a saved manifest, downgrading rows that no longer describe the files on disk.

    A row becomes ``missing`` if its file is gone, and ``stale`` if the manifest was made for a
    different raw directory, the file appeared after verification, or its size or mtime changed.
    """
    manifest = pd.read_parquet(path)
    raw_dir = Path(raw_dir)
    same_root = bool((manifest.pop("raw_dir") == str(raw_dir.resolve())).all())
    statuses = []
    for row in manifest.itertuples(index=False):
        file = raw_dir / row.path
        if not file.is_file():
            statuses.append(MISSING)
        elif row.status == MISSING or not same_root:
            statuses.append(STALE)
        else:
            stat = file.stat()
            unchanged = stat.st_size == row.size and stat.st_mtime_ns == row.mtime_ns
            statuses.append(row.status if unchanged else STALE)
    manifest["status"] = statuses
    manifest.attrs["raw_dir"] = str(raw_dir.resolve())
    return manifest
