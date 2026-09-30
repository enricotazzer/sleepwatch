import os
from pathlib import Path

import pytest

from sleepwatch.config import PROJECT_ROOT, Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SLEEPWATCH_"):
            monkeypatch.delenv(key)


def test_defaults_follow_repo_layout():
    settings = Settings(_env_file=None)
    assert settings.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert settings.processed_dir == PROJECT_ROOT / "data" / "processed"
    assert settings.results_dir == PROJECT_ROOT / "results"


def test_data_dir_moves_all_data_subdirs(monkeypatch, tmp_path):
    monkeypatch.setenv("SLEEPWATCH_DATA_DIR", str(tmp_path))
    settings = Settings(_env_file=None)
    assert (settings.raw_dir, settings.interim_dir, settings.processed_dir) == (
        tmp_path / "raw",
        tmp_path / "interim",
        tmp_path / "processed",
    )


def test_one_subdir_can_be_overridden_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("SLEEPWATCH_RAW_DIR", str(tmp_path / "dataset"))
    settings = Settings(_env_file=None)
    assert settings.raw_dir == tmp_path / "dataset"
    assert settings.interim_dir == PROJECT_ROOT / "data" / "interim"


def test_relative_and_home_paths_are_resolved(monkeypatch):
    monkeypatch.setenv("SLEEPWATCH_RESULTS_DIR", "out/results")
    monkeypatch.setenv("SLEEPWATCH_PROCESSED_DIR", "~/sleep")
    settings = Settings(_env_file=None)
    assert settings.results_dir == PROJECT_ROOT / "out" / "results"
    assert settings.processed_dir == Path.home() / "sleep"
