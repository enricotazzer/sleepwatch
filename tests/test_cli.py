import re

from typer.testing import CliRunner

from sleepwatch.cli import app
from sleepwatch.data import manifest as mf

runner = CliRunner()
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def output(result) -> str:
    """CLI output without ANSI styling, which Rich adds when e.g. FORCE_COLOR is set (Jupyter)."""
    return ANSI.sub("", result.output)


def test_verify_reports_problems_and_saves_manifest(fake_raw, interim_dir):
    result = runner.invoke(app, ["data", "verify", "--raw-dir", str(fake_raw)])
    assert result.exit_code == 1  # one corrupt and one missing file
    assert "Nights: 1/3 verified" in output(result)
    assert "corrupt: Bidslab00/2/motion.csv" in output(result)
    assert (interim_dir / mf.MANIFEST_FILE).is_file()


def test_fast_verify_saves_nothing(fake_raw, interim_dir):
    result = runner.invoke(app, ["data", "verify", "--fast", "--raw-dir", str(fake_raw)])
    assert result.exit_code == 1
    assert "manifest not saved" in output(result)
    assert not interim_dir.exists()


def test_verify_without_checksum_file_exits_2(tmp_path, interim_dir):
    result = runner.invoke(app, ["data", "verify", "--raw-dir", str(tmp_path / "absent")])
    assert result.exit_code == 2
    assert "SHA256SUMS.txt" in output(result)
