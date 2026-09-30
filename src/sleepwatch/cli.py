"""Command-line entry point: ``sleepwatch <group> <command>``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import track

from sleepwatch.config import get_settings
from sleepwatch.data import manifest as mf

app = typer.Typer(no_args_is_help=True, help="Personalized sleep staging and anomaly detection.")
data_app = typer.Typer(no_args_is_help=True, help="Dataset utilities.")
app.add_typer(data_app, name="data")
console = Console(soft_wrap=True)  # don't break paths across lines


@data_app.command("verify")
def verify(
    fast: Annotated[
        bool,
        typer.Option(
            "--fast", help="Skip hashing; only check presence and truncation. Nothing is saved."
        ),
    ] = False,
    raw_dir: Annotated[
        Path | None,
        typer.Option(help="Dataset folder containing SHA256SUMS.txt [default: settings.raw_dir]."),
    ] = None,
) -> None:
    """Check the raw dataset against SHA256SUMS.txt and save the manifest that loaders read.

    Exits with code 1 if any listed file is missing or corrupt.
    """
    settings = get_settings()
    action = "Checking" if fast else "Hashing"
    try:
        manifest = mf.verify(
            raw_dir or settings.raw_dir,
            checksums=not fast,
            wrap=lambda files: track(
                files, description=f"{action} {len(files)} files", console=console, transient=True
            ),
        )
    except FileNotFoundError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(code=2) from err

    summary = mf.summarize(manifest)
    counts = "  ".join(f"{k}={v}" for k, v in sorted(summary["status_counts"].items()))
    console.print(f"Raw dir:  {manifest.attrs['raw_dir']}")
    console.print(f"Files:    {summary['files']} listed  {counts}")
    console.print(
        f"Subjects: {summary['subjects_complete']}/{summary['subjects']} complete   "
        f"Nights: {summary['nights_verified']}/{summary['nights']} verified"
    )
    if incomplete := summary["incomplete_subjects"]:
        shown = ", ".join(incomplete[:10]) + (", ..." if len(incomplete) > 10 else "")
        console.print(f"Incomplete subjects ({len(incomplete)}): {shown}")
    for path in summary["corrupt"]:
        console.print(f"[red]corrupt:[/red] {path}")

    if fast:
        console.print("Fast mode: nothing hashed, manifest not saved.")
    else:
        out = settings.interim_dir / mf.MANIFEST_FILE
        mf.save_manifest(manifest, out)
        console.print(f"Manifest saved to {out}")

    if summary["status_counts"].get(mf.MISSING) or summary["corrupt"]:
        raise typer.Exit(code=1)
