"""Provenance recorded next to every generated artefact."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from sleepwatch.config import PROJECT_ROOT


def git_revision() -> dict:
    """Current commit and whether the working tree has uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
