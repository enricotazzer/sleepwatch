"""Shared fixtures: a tiny fake copy of the raw dataset, and auto-skipping of real-data tests."""

import hashlib
from pathlib import Path

import pytest

from sleepwatch.config import Settings, get_settings
from sleepwatch.constants import CHECKSUM_FILE

# Relative path -> the content SHA256SUMS.txt describes.
_FAKE_FILES = {
    "README.md": b"# fake dataset\n",
    "Bidslab00/1/hr.csv": b"1638504593.03,69.0\n1638504598.03,68.0\n",
    "Bidslab00/1/motion.csv": b"Timestamp,x,y,z\n1638504587.43,-0.09,-0.18,-0.96\n",
    "Bidslab00/1/labels.mat": b"MATLAB 5.0 MAT-file" + b"\x00" * 8,  # MAT v5 pads with zeros
    "Bidslab00/2/hr.csv": b"1638590000.00,61.0\n",
    "Bidslab00/2/motion.csv": b"Timestamp,x,y,z\n1638590000.00,0.01,0.02,-0.99\n",
    "Bidslab00/2/labels.mat": b"MATLAB 5.0 MAT-file night 2",
    "Bidslab01/1/hr.csv": b"1638600000.00,55.0\n",
    "Bidslab01/1/motion.csv": b"Timestamp,x,y,z\n1638600000.00,0.00,0.00,-1.00\n",
    "Bidslab01/1/labels.mat": b"MATLAB 5.0 MAT-file subject 1",
}
_NUL_PADDED = "Bidslab00/2/motion.csv"  # written like an interrupted download
_MISSING = "Bidslab01/1/labels.mat"  # listed but never written


@pytest.fixture
def fake_raw(tmp_path: Path) -> Path:
    """Fake dataset: night Bidslab00/1 is intact, Bidslab00/2 has a NUL-padded motion.csv and
    Bidslab01/1 lacks labels.mat."""
    raw = tmp_path / "raw"
    lines = []
    for rel, content in _FAKE_FILES.items():
        lines.append(f"{hashlib.sha256(content).hexdigest()} {rel}")
        if rel == _MISSING:
            continue
        path = raw / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel == _NUL_PADDED:
            half = len(content) // 2
            content = content[:half] + b"\x00" * (len(content) - half)
        path.write_bytes(content)
    (raw / "Bidslab00/1/._hr.csv").write_bytes(b"\x00\x05\x16\x07")  # macOS resource fork
    (raw / CHECKSUM_FILE).write_text("\n".join(lines) + "\n")
    return raw


@pytest.fixture
def interim_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the interim data dir at a temp folder for code that calls get_settings()."""
    path = tmp_path / "interim"
    monkeypatch.setenv("SLEEPWATCH_INTERIM_DIR", str(path))
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked ``data`` when the raw dataset isn't available (e.g. in CI)."""
    if (Settings().raw_dir / CHECKSUM_FILE).is_file():
        return
    skip = pytest.mark.skip(reason="raw dataset not found; see README > Data")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
