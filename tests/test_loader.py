from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from synth import REC_LOCAL, REC_START, make_night, write_night_files

from sleepwatch.constants import N2, REM, UNKNOWN
from sleepwatch.data.loader import load_night, rec_start_to_unix


def utc(*args) -> float:
    return datetime(*args, tzinfo=UTC).timestamp()


def test_rec_start_is_us_eastern():
    assert rec_start_to_unix(REC_LOCAL) == (REC_START, False)
    assert rec_start_to_unix("2022-04-01 23:00:00")[0] == utc(2022, 4, 2, 3)  # daylight time


@pytest.mark.parametrize(
    ("local", "edt", "est"),
    [
        ("2021-11-07 01:30:00", utc(2021, 11, 7, 5, 30), utc(2021, 11, 7, 6, 30)),  # repeated
        ("2022-03-13 02:30:00", utc(2022, 3, 13, 6, 30), utc(2022, 3, 13, 7, 30)),  # skipped
    ],
)
def test_dst_changes_resolve_to_the_reading_nearest_the_signal(local, edt, est):
    assert rec_start_to_unix(local, near=edt - 60) == (edt, True)
    assert rec_start_to_unix(local, near=est + 60) == (est, True)
    with pytest.raises(ValueError, match="DST"):
        rec_start_to_unix(local)


def write(tmp_path, n_epochs=4, dreem_len=None, extra_hr=None, **kw):
    night = make_night(n_epochs=n_epochs, **kw)
    hr = pd.concat([night.hr, extra_hr]) if extra_hr is not None else night.hr
    dreem = np.full(dreem_len or n_epochs, REM, np.uint8)
    write_night_files(tmp_path / "Bidslab99" / "1", hr, night.motion, night.expert, dreem)
    return night


def test_signals_are_cropped_to_the_label_window(tmp_path):
    outside = pd.DataFrame({"t": [REC_START - 100.0, REC_START + 4 * 30 + 5000.0], "hr": 80.0})
    write(tmp_path, extra_hr=outside)
    night = load_night(tmp_path, "Bidslab99", 1)
    assert night.rec_start == REC_START
    assert night.hr["t"].between(REC_START, night.end, inclusive="left").all()
    assert night.quality["hr_start_offset_min"] == pytest.approx(-100 / 60)
    assert night.quality["hr_end_offset_min"] > 80
    # In-window statistics ignore the samples outside the window (5 s steps, constant 60 bpm).
    assert night.quality["hr_window_median_dt_s"] == pytest.approx(5.0)
    assert night.quality["hr_window_median_step_bpm"] == 0.0


def test_duplicate_hr_timestamps_are_averaged(tmp_path):
    dup = pd.DataFrame({"t": [REC_START + 10.0], "hr": [70.0]})  # a 60 bpm sample exists at +10 s
    write(tmp_path, extra_hr=dup)
    night = load_night(tmp_path, "Bidslab99", 1)
    assert night.quality["hr_duplicate_ts"] == 1
    assert night.hr.set_index("t").loc[REC_START + 10.0, "hr"] == 65.0
    assert night.hr["t"].is_monotonic_increasing


@pytest.mark.parametrize("dreem_len", [2, 7])
def test_dreem_labels_match_expert_length(tmp_path, dreem_len):
    write(tmp_path, dreem_len=dreem_len)
    night = load_night(tmp_path, "Bidslab99", 1)
    assert len(night.dreem) == len(night.expert) == 4
    assert night.expert.tolist() == [N2] * 4
    expected = [REM] * min(dreem_len, 4) + [UNKNOWN] * max(0, 4 - dreem_len)
    assert night.dreem.tolist() == expected
    assert night.quality["n_dreem_raw"] == dreem_len


def test_unexpected_formats_fail_loudly(tmp_path):
    night = write(tmp_path)
    folder = tmp_path / "Bidslab99" / "1"
    night.motion.to_csv(folder / "motion.csv", index=False)  # header t,x,y,z instead of Timestamp
    with pytest.raises(ValueError, match="header"):
        load_night(tmp_path, "Bidslab99", 1)

    write(tmp_path)
    night.hr.to_csv(folder / "hr.csv", index=False)  # hr.csv with a header row
    with pytest.raises(ValueError, match="no header"):
        load_night(tmp_path, "Bidslab99", 1)


def test_invalid_stage_codes_fail(tmp_path):
    night = make_night(n_epochs=2)
    write_night_files(tmp_path / "Bidslab99" / "1", night.hr, night.motion, [2, 9], [2, 2])
    with pytest.raises(ValueError, match="codes"):
        load_night(tmp_path, "Bidslab99", 1)


@pytest.mark.data
def test_real_nights_load_inside_their_label_window():
    from sleepwatch.config import get_settings
    from sleepwatch.data import manifest as mf

    settings = get_settings()
    path = settings.interim_dir / mf.MANIFEST_FILE
    if not path.is_file():
        pytest.skip("no manifest; run `sleepwatch data verify`")
    nights = mf.verified_nights(mf.load_manifest(path, settings.raw_dir))
    for subject, number in nights[:: max(1, len(nights) // 5)]:
        night = load_night(settings.raw_dir, subject, number)
        for signal in (night.hr, night.motion):
            assert signal["t"].between(night.rec_start, night.end, inclusive="left").all()
            assert signal["t"].is_monotonic_increasing and signal["t"].is_unique
        assert len(night.dreem) == night.n_epochs > 0


def test_truncated_last_rows_are_skipped_and_counted(tmp_path):
    """As in Bidslab42/3-4: the recording stopped mid-record, leaving a partial last line."""
    night = write(tmp_path)
    folder = tmp_path / "Bidslab99" / "1"
    with open(folder / "hr.csv", "a") as fh:
        fh.write(f"{REC_START + 100.5}\n")
    with open(folder / "motion.csv", "a") as fh:
        fh.write(f"{REC_START + 100.5},-0.8913\n")
    loaded = load_night(tmp_path, "Bidslab99", 1)
    assert loaded.quality["hr_malformed_rows"] == 1
    assert loaded.quality["motion_malformed_rows"] == 1
    assert len(loaded.hr) == len(night.hr) and len(loaded.motion) == len(night.motion)
