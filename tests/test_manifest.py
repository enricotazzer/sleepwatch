import shutil

import pytest

from sleepwatch.config import get_settings
from sleepwatch.data import manifest as mf


def status_of(manifest, path):
    return manifest.set_index("path").loc[path, "status"]


def test_read_checksums_parses_subject_and_night(fake_raw):
    entries = {e.path: e for e in mf.read_checksums(fake_raw)}
    assert (entries["Bidslab00/2/motion.csv"].subject, entries["Bidslab00/2/motion.csv"].night) == (
        "Bidslab00",
        2,
    )
    assert entries["README.md"].subject is None
    assert not any(path.endswith("._hr.csv") for path in entries)


def test_read_checksums_rejects_malformed_line(tmp_path):
    (tmp_path / "SHA256SUMS.txt").write_text("not-a-hash Bidslab00/1/hr.csv\n")
    with pytest.raises(ValueError, match="malformed"):
        mf.read_checksums(tmp_path)


def test_missing_checksum_file_names_it(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"SHA256SUMS\.txt"):
        mf.read_checksums(tmp_path)


def test_verify_detects_ok_corrupt_and_missing(fake_raw):
    manifest = mf.verify(fake_raw)
    assert status_of(manifest, "Bidslab00/1/hr.csv") == mf.OK
    assert status_of(manifest, "README.md") == mf.OK
    assert status_of(manifest, "Bidslab00/2/motion.csv") == mf.CORRUPT
    assert status_of(manifest, "Bidslab01/1/labels.mat") == mf.MISSING


def test_fast_mode_flags_nul_padded_text_but_not_mat_padding(fake_raw):
    manifest = mf.verify(fake_raw, checksums=False)
    assert status_of(manifest, "Bidslab00/2/motion.csv") == mf.CORRUPT
    assert status_of(manifest, "Bidslab00/1/labels.mat") == mf.UNVERIFIED  # ends in zero padding
    assert status_of(manifest, "Bidslab01/1/labels.mat") == mf.MISSING


def test_only_fully_hashed_nights_count_as_verified(fake_raw):
    assert mf.verified_nights(mf.verify(fake_raw)) == [("Bidslab00", 1)]
    assert mf.verified_nights(mf.verify(fake_raw, checksums=False)) == []


def test_summary(fake_raw):
    summary = mf.summarize(mf.verify(fake_raw))
    assert (summary["subjects"], summary["subjects_complete"]) == (2, 0)
    assert (summary["nights"], summary["nights_verified"]) == (3, 1)
    assert summary["incomplete_subjects"] == ["Bidslab00", "Bidslab01"]
    assert summary["corrupt"] == ["Bidslab00/2/motion.csv"]


def test_saved_manifest_round_trips(fake_raw, tmp_path):
    path = tmp_path / mf.MANIFEST_FILE
    manifest = mf.verify(fake_raw)
    mf.save_manifest(manifest, path)
    reloaded = mf.load_manifest(path, fake_raw)
    assert reloaded["status"].tolist() == manifest["status"].tolist()
    assert reloaded["mtime_ns"].tolist() == manifest["mtime_ns"].tolist()  # exact, no float trip


def test_file_changed_after_hashing_is_stale(fake_raw, tmp_path):
    path = tmp_path / mf.MANIFEST_FILE
    mf.save_manifest(mf.verify(fake_raw), path)
    (fake_raw / "Bidslab00/1/hr.csv").write_text("edited\n")
    reloaded = mf.load_manifest(path, fake_raw)
    assert status_of(reloaded, "Bidslab00/1/hr.csv") == mf.STALE
    assert mf.verified_nights(reloaded) == []


def test_file_appearing_after_verification_is_stale(fake_raw, tmp_path):
    path = tmp_path / mf.MANIFEST_FILE
    mf.save_manifest(mf.verify(fake_raw), path)
    (fake_raw / "Bidslab01/1/labels.mat").write_bytes(b"downloaded later")
    assert status_of(mf.load_manifest(path, fake_raw), "Bidslab01/1/labels.mat") == mf.STALE


def test_manifest_of_another_raw_dir_is_stale(fake_raw, tmp_path):
    path = tmp_path / mf.MANIFEST_FILE
    mf.save_manifest(mf.verify(fake_raw), path)
    copy = shutil.copytree(fake_raw, tmp_path / "copy")
    assert mf.verified_nights(mf.load_manifest(path, copy)) == []


@pytest.mark.data
def test_published_checksum_list_covers_the_whole_dataset():
    entries = mf.read_checksums(get_settings().raw_dir)
    nights = {(e.subject, e.night) for e in entries if e.subject}
    assert len({subject for subject, _ in nights}) == 47
    assert len(nights) == 253
    assert sum(e.subject is not None for e in entries) == 3 * 253
