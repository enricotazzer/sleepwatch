# sleepwatch

Research-grade portfolio project and deployable app: personalized sleep staging and anomaly
detection from Apple Watch heart rate (HR) and accelerometry, plus an LLM follow-up agent that asks
about the day after an anomalous night and stores reported factors as structured data.
Research prototype, not a medical device.

## Workflow
Work is split into phases 0–6 (see README). Each phase starts with a plan the user approves, and
ends with a results summary; then wait for the go-ahead. Ask when a decision is the user's.
Phases 0 (setup) and 1 (data pipeline + EDA) are done.

## Commands
```bash
uv sync                                  # .venv from uv.lock (Python 3.12)
uv run pytest                            # all tests; `-m data` = real-data tests only
uv run ruff check . && uv run ruff format .
uv run pre-commit run --all-files
uv run sleepwatch data verify            # hash raw files vs SHA256SUMS.txt, save the manifest
uv run sleepwatch data verify --fast     # presence/truncation check only, saves nothing
uv run sleepwatch data build-epochs      # epoch table from verified nights (configs/features/v1.yaml)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/phase1_2_check.ipynb  # review checks
```

## Layout
- `src/sleepwatch/`: `config.py` (paths from `SLEEPWATCH_*` env / `.env`), `constants.py`
  (dataset facts, stage codes), `cli.py`, `provenance.py` (git revision for outputs), `data/`
  (`manifest`, `loader` with `load_night`, `align`, `label_timing`), `features/`
  (`epoch_features`, `build` with `load_build`), `models/`, `anomaly/`, `agent/`, `app/`.
- Epoch table: `data/processed/epochs_<name>.parquet`; model inputs are
  `feature_columns(df)`, which excludes metadata, labels and `qc_*` coverage columns.
- `configs/`: experiment YAMLs extending `base.yaml`; `configs/splits/` holds the fold file.
- `data/` (gitignored): `raw` → symlink to the dataset on the external T7 drive; `interim` →
  symlink to `/Volumes/T7/sleepwatch-data/interim`; `processed/` local. `results/` (gitignored).
- `docs/data.md`: dataset facts and known quirks. `docs/decisions.md`: decision log.

## Data facts (details in docs/data.md)
- BIDSleep v1.0.1: 47 subjects, 253 nights, 3–7 nights each. Per night: `hr.csv` (no header:
  unix_t, bpm; 2–5 s intervals), `motion.csv` (`Timestamp,x,y,z`, ~50 Hz), `labels.mat`
  (`recStart` as a US Eastern local-time string, `dreem_label`, `expert_label`).
- Stages: 0 Wake, 1 N1, 2 N2, 3 N3, 4 REM, 5 Unknown. Epoch k (0-based) covers
  `[recStart + 30k, recStart + 30(k+1))`.
- Signal files overhang the label window (up to days) or start/end inside it: always crop and mask.
- All 253 nights are checksum-verified (2026-10-01); `data/raw` points to the unzipped ZIP on the T7.
- Open issues (docs/data.md): expert labels appear to run ~3 epochs (90 s) late relative to the
  signals, a constant offset (Dreem doesn't), and `Bidslab01/4` has suspect expert labels.
  Decided: headline results use the documented alignment, with a shifted-label sensitivity
  analysis (offset estimated on training subjects only); `Bidslab01/4` is kept.
- The T7 is exFAT (no symlinks, `._*` litter): the repo and venv stay on the internal disk, which
  has little free space, so raw data and large caches stay on the T7.

## Rules
- Never commit data (gitignore + pre-commit hook). Never copy raw data to the internal disk.
- Load only nights from `verified_nights()` on the saved manifest; never glob the raw folder.
- Ground truth is `expert_label`. `dreem_label` may be a noisy training signal, never a feature or
  an evaluation target. Unknown epochs are masked from losses and metrics.
- Split by subject (5-fold GroupKFold, one saved fold file). Tune hyperparameters, thresholds and
  early stopping only on inner splits of training subjects.
- "Previous nights" = strictly earlier nights of the same subject. Headline personalization is
  label-free; fine-tuning on prior nights' labels is reported only as an upper bound.
- No label-derived features (e.g. sleep onset from labels): use time since recording start or
  actigraphy-estimated onset.
- No HRV claims (HR is too coarse). No causal, psychological or medical claims; the agent never
  tells a user what caused an anomaly.
- Report per-subject mean ± SD and pooled metrics with subject-bootstrap CIs, and the number of
  subjects/nights behind every number. Partial-download results are sanity checks, never reported.
- Experiments are config-driven with fixed seeds; each run writes its resolved config, git SHA and
  metrics to `results/<experiment>/<timestamp>/`. Flag anything that could inflate results.
- Tests: synthetic fixtures for logic (run in CI without data); `@pytest.mark.data` for real-data
  checks. Alignment and metric code always gets tests.
