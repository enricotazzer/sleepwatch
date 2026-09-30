"""Filesystem layout, read from ``SLEEPWATCH_*`` environment variables or ``.env``.

Defaults follow the repo layout: ``data/raw`` (the dataset folder containing ``SHA256SUMS.txt``),
``data/interim``, ``data/processed`` and ``results``. Setting only ``SLEEPWATCH_DATA_DIR`` moves
all three data subfolders; each can also be set on its own. Relative paths are resolved against
the project root, so notebooks and scripts see the same folders from any working directory.

Experiment settings (seeds, folds, hyperparameters) live in ``configs/``, not here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root() -> Path:
    """Nearest ancestor of this file with a ``pyproject.toml``, else the working directory."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
_DATA_SUBDIRS = ("raw", "interim", "processed")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SLEEPWATCH_", env_file=PROJECT_ROOT / ".env", extra="ignore"
    )

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    results_dir: Path = Path("results")

    @model_validator(mode="before")
    @classmethod
    def _derive_data_subdirs(cls, values: Any) -> Any:
        """Put unset data subfolders under ``data_dir`` (which may itself be overridden)."""
        if isinstance(values, dict):
            data_dir = Path(values.get("data_dir") or "data")
            for name in _DATA_SUBDIRS:
                values.setdefault(f"{name}_dir", data_dir / name)
        return values

    @field_validator("data_dir", "raw_dir", "interim_dir", "processed_dir", "results_dir")
    @classmethod
    def _anchor(cls, path: Path) -> Path:
        path = path.expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
