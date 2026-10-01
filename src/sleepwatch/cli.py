"""Command-line entry point: ``sleepwatch <group> <command>``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import track

from sleepwatch.config import PROJECT_ROOT, get_settings
from sleepwatch.data import manifest as mf

app = typer.Typer(no_args_is_help=True, help="Personalized sleep staging and anomaly detection.")
data_app = typer.Typer(no_args_is_help=True, help="Dataset utilities.")
splits_app = typer.Typer(no_args_is_help=True, help="Cross-validation folds.")
app.add_typer(data_app, name="data")
app.add_typer(splits_app, name="splits")
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


@data_app.command("build-epochs")
def build_epochs(
    config: Annotated[Path, typer.Option(help="Feature config YAML.")] = Path(
        "configs/features/v1.yaml"
    ),
    jobs: Annotated[int, typer.Option(help="Nights processed in parallel.")] = 4,
    limit: Annotated[int | None, typer.Option(help="Only the first N nights (testing).")] = None,
) -> None:
    """Build the epoch feature table from the checksum-verified nights in the saved manifest."""
    from sleepwatch.features.build import build_epochs as build
    from sleepwatch.features.build import save_build
    from sleepwatch.features.epoch_features import FeatureConfig

    settings = get_settings()
    manifest_path = settings.interim_dir / mf.MANIFEST_FILE
    if not manifest_path.is_file():
        console.print("[red]No manifest yet: run `sleepwatch data verify` first.[/red]")
        raise typer.Exit(code=2)
    manifest = mf.load_manifest(manifest_path, settings.raw_dir)
    summary = mf.summarize(manifest)
    nights = mf.verified_nights(manifest)[:limit]
    if not nights:
        console.print("[red]No verified nights; re-run `sleepwatch data verify`.[/red]")
        raise typer.Exit(code=2)
    if len(nights) < summary["nights"]:
        console.print(
            f"[yellow]Partial dataset: {len(nights)}/{summary['nights']} nights verified. "
            "Use for development only.[/yellow]"
        )

    if not config.is_absolute() and not config.exists():
        config = PROJECT_ROOT / config
    cfg = FeatureConfig.from_yaml(config)
    console.print(f"Building '{cfg.name}' features for {len(nights)} nights with {jobs} jobs...")
    epochs, quality = build(settings.raw_dir, nights, cfg, jobs=jobs)
    paths = save_build(
        epochs,
        quality,
        cfg,
        settings.processed_dir,
        extra={"raw_dir": str(settings.raw_dir.resolve()), "nights_listed": summary["nights"]},
    )
    console.print(
        f"{len(epochs)} epochs from {len(quality)} nights / {epochs['subject'].nunique()} subjects"
    )
    for kind, path in paths.items():
        console.print(f"  {kind}: {path}")


@splits_app.command("make")
def make_splits(
    n_folds: Annotated[int, typer.Option(help="Number of folds.")] = 5,
    seed: Annotated[int, typer.Option(help="Shuffle seed.")] = 42,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing fold file.")
    ] = False,
) -> None:
    """Assign subjects to folds once; every experiment reuses this file."""
    from sleepwatch.models.splits import SPLITS_FILE, make_folds, save_folds

    if SPLITS_FILE.exists() and not force:
        console.print(f"[red]{SPLITS_FILE} exists; folds are fixed. Use --force to replace.[/red]")
        raise typer.Exit(code=2)
    settings = get_settings()
    manifest = mf.load_manifest(settings.interim_dir / mf.MANIFEST_FILE, settings.raw_dir)
    nights = mf.verified_nights(manifest)
    summary = mf.summarize(manifest)
    if len(nights) < summary["nights"]:
        console.print("[red]Not every night is verified; folds must cover the full dataset.[/red]")
        raise typer.Exit(code=2)
    folds = make_folds(nights, n_folds, seed)
    save_folds(folds, SPLITS_FILE, n_folds, seed)
    per_fold = {f: sum(1 for s, n in nights if folds[s] == f) for f in range(n_folds)}
    console.print(f"{len(folds)} subjects in {n_folds} folds; nights per fold: {per_fold}")
    console.print(f"Saved {SPLITS_FILE}")


@app.command("train")
def train(
    config: Annotated[Path, typer.Argument(help="Staging experiment YAML.")],
    fold: Annotated[
        list[int] | None, typer.Option(help="Run only these folds (quick checks; marked partial).")
    ] = None,
) -> None:
    """Run a cross-validated staging experiment and save predictions and metrics."""
    from sleepwatch.features.build import load_build
    from sleepwatch.models.experiment import StagingConfig, compute_metrics, run_folds, save_run
    from sleepwatch.models.splits import SPLITS_FILE, load_folds

    if not config.is_absolute() and not config.exists():
        config = PROJECT_ROOT / config
    cfg = StagingConfig.from_yaml(config)
    if fold:
        cfg = cfg.model_copy(update={"folds": fold})
    settings = get_settings()
    epochs, _, build = load_build(cfg.features, settings.processed_dir)
    folds = load_folds()
    console.print(f"Experiment '{cfg.name}' ({cfg.model}); folds: {cfg.folds or 'all'}")
    pred, fold_log = run_folds(cfg, epochs, folds, log=console.print)
    metrics = compute_metrics(pred, cfg.n_boot, cfg.seed)
    run_dir = save_run(
        cfg,
        pred,
        fold_log,
        metrics,
        settings.results_dir,
        extra={"features_build": build, "splits_file": str(SPLITS_FILE)},
    )
    pooled = metrics["population"]["all"]["5"]["pooled"]
    console.print(
        f"Population, 5-class: kappa {pooled['kappa']:.3f}, macro-F1 {pooled['macro_f1']:.3f}"
    )
    console.print(f"Saved {run_dir}")
