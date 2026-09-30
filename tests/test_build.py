import hashlib

import numpy as np
import pandas as pd
import pytest
from synth import REC_START, make_night, write_night_files
from typer.testing import CliRunner

from sleepwatch.cli import app
from sleepwatch.config import get_settings
from sleepwatch.features.build import add_chronology


@pytest.fixture
def two_night_raw(tmp_path):
    raw = tmp_path / "raw"
    for night, local, start in [
        (1, "2022-01-10 23:00:00", REC_START),
        (2, "2022-01-11 23:00:00", REC_START + 86400),
    ]:
        n = make_night(n_epochs=40 + night, rec_start=start)
        write_night_files(raw / "Bidslab99" / str(night), n.hr, n.motion, n.expert, n.dreem, local)
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()} {p.relative_to(raw).as_posix()}"
        for p in sorted(raw.rglob("*"))
        if p.is_file()
    ]
    (raw / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    return raw


def test_build_epochs_end_to_end(two_night_raw, tmp_path, monkeypatch):
    monkeypatch.setenv("SLEEPWATCH_RAW_DIR", str(two_night_raw))
    monkeypatch.setenv("SLEEPWATCH_INTERIM_DIR", str(tmp_path / "interim"))
    monkeypatch.setenv("SLEEPWATCH_PROCESSED_DIR", str(tmp_path / "processed"))
    get_settings.cache_clear()
    try:
        runner = CliRunner()
        assert runner.invoke(app, ["data", "verify"]).exit_code == 0
        result = runner.invoke(app, ["data", "build-epochs", "--jobs", "1"])
        assert result.exit_code == 0, result.output
    finally:
        get_settings.cache_clear()

    epochs = pd.read_parquet(tmp_path / "processed" / "epochs_v1.parquet")
    quality = pd.read_csv(tmp_path / "processed" / "quality_v1.csv")
    assert len(epochs) == 41 + 42
    assert quality["chronological"].all()
    assert (tmp_path / "processed" / "build_v1.json").is_file()


def test_build_refuses_without_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("SLEEPWATCH_INTERIM_DIR", str(tmp_path / "interim"))
    get_settings.cache_clear()
    try:
        result = CliRunner().invoke(app, ["data", "build-epochs"])
    finally:
        get_settings.cache_clear()
    assert result.exit_code == 2
    assert "verify" in result.output


def test_chronology_flags_out_of_order_nights():
    quality = pd.DataFrame(
        {
            "subject": ["A", "A", "B", "B"],
            "night": [1, 2, 1, 2],
            "rec_start": [100.0, 200.0, 300.0, np.float64(250.0)],
        }
    )
    flags = add_chronology(quality).set_index("subject")["chronological"]
    assert flags["A"].all() and not flags["B"].any()
