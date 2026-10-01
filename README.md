# sleepwatch

Personalized sleep staging and anomaly detection from Apple Watch heart rate and accelerometry,
with an LLM follow-up agent that asks about your day after an unusual night.

> **Research prototype. Not a medical device; not for diagnosis or treatment.**

## What it does

1. Learns each person's own heart-rate and movement patterns per sleep stage from their previous
   nights.
2. Stages sleep from Apple Watch heart rate and accelerometry, personalized with the same person's
   earlier nights.
3. Flags nights and periods that deviate from that personal baseline.
4. For each flagged night, holds a short conversation with an LLM that asks open questions about
   the day (caffeine, alcohol, late exercise or meals, stress, illness, schedule changes) and stores
   the answers as structured data, without telling the person what caused anything.
5. Runs as a web app (FastAPI, SQLite, Docker), with local LLMs by default so data stays on the
   machine.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Project setup, dataset verification | done |
| 1 | Data pipeline, epoch features, exploratory analysis | done |
| 2 | Sleep staging with multi-night personalization | in progress |
| 3 | Personalized anomaly detection | planned |
| 4 | LLM follow-up agent and simulated-user evaluation | planned |
| 5 | Deployable app | planned |
| 6 | Results and documentation | planned |

## Data

The project uses *A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep
Stage Labels* (PhysioNet, v1.0.1, ODC-BY 1.0): 47 healthy adults, 253 nights, Apple Watch heart
rate and accelerometry, and sleep stages from a Dreem-2 EEG headband scored to AASM rules and
corrected by a sleep expert. The data is not included in this repository.

1. Download the ZIP (6.35 GB) from <https://physionet.org/content/bidsleep-dataset/1.0.1/> and
   unzip it on a drive with at least 28 GB free.
2. Link the folder that contains `SHA256SUMS.txt`:
   `ln -sfn /path/to/that/folder data/raw` (or set `SLEEPWATCH_RAW_DIR` in `.env`).
3. Verify it: `uv run sleepwatch data verify`. This hashes every file against the published
   checksums and saves the manifest the loaders read. A complete download reports 47/47 subjects
   and 253/253 nights.

See [docs/data.md](docs/data.md) for file formats and known quirks.

## Setup

```bash
brew install uv            # or https://docs.astral.sh/uv/getting-started/installation/
uv sync
uv run pre-commit install
uv run pytest
```

## Repository layout

```
src/sleepwatch/   package: config, constants, CLI, data, features, models, anomaly, agent, app
configs/          experiment configurations (YAML)
notebooks/        exploratory analysis
tests/            unit tests (synthetic data; real-data tests skip when the data is absent)
docs/             dataset notes and decision log
```

## Methodology principles

- Train/test splits are always by subject; tuning happens only on training subjects.
- Expert-corrected labels are the ground truth; the automatic Dreem labels are at most a noisy
  training signal.
- Personalization uses only a person's earlier nights, and the headline variant needs no sleep
  labels for them, as in real use.
- No heart-rate-variability analysis (heart rate is sampled every 2–5 s) and no causal or medical
  claims.

## Citation

If you use this work, please cite the dataset and the related publications:

- Song T, Zhang Y, Zhou Z, Dutta J. *A Multi-Night Instantaneous Heart Rate and Accelerometry
  Dataset with EEG Sleep Stage Labels* (version 1.0.1). PhysioNet, 2026.
  <https://doi.org/10.13026/rees-1092>
- Song TA, Zhang Y, Zhou Z, Hou L, Malekzadeh M, Behzad A, Dutta J. AI-driven sleep staging using
  instantaneous heart rate and accelerometry: insights from an Apple Watch study. *IEEE
  Transactions on Biomedical Engineering*. <https://doi.org/10.1109/TBME.2025.3612158>
- Pollard T, Moody BE, Lehman L, Gow B, Fernandes C, Xie C, Johnson A, Mark RG, Heldt T. PhysioNet
  as a global platform for biomedical research. *Nature Health*, 2026.
  <https://doi.org/10.1038/s44360-026-00096-z>

## License

Code: MIT (see [LICENSE](LICENSE)). The dataset is licensed separately under ODC-BY 1.0 and is not
redistributed here.
