# sleepwatch

Personalized sleep staging and anomaly detection from Apple Watch heart rate and accelerometry,
with an LLM follow-up agent that asks about your day after an unusual night.

> **Research prototype. Not a medical device; not for diagnosis or treatment.**

## What it does

1. Learns each person's own heart-rate and movement patterns per sleep stage from their previous
   nights.
2. Stages sleep from Apple Watch heart rate and accelerometry, and tests whether the same
   person's earlier nights improve it (so far they don't; see below).
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
| 2 | Sleep staging with multi-night personalization | done |
| 2b | Three more personalization designs | done |
| 3 | Personalized anomaly detection | planned |
| 4 | LLM follow-up agent and simulated-user evaluation | planned |
| 5 | Deployable app | planned |
| 6 | Results and documentation | planned |

### Phase 2 results (5-fold cross-validation by subject, 47 subjects, 253 nights)

| Model | Kappa, 5-class [95% CI] | Macro-F1, 5-class | Kappa, 4-class | Accuracy, 4-class |
|---|---|---|---|---|
| Bidirectional GRU over whole nights | 0.510 [0.482, 0.537] | 0.566 | 0.533 | 0.684 |
| Gradient-boosted trees, per epoch | 0.381 [0.357, 0.404] | 0.504 | 0.390 | 0.592 |

- **Personalization with 1–3 earlier nights didn't improve staging.** This was compared against
  a control trained on the same nights. It held for label-free personal features, and even for an
  upper bound that uses the earlier nights' expert labels.
- **Phase 2b tried three more designs, fixed in advance; none improved staging without labels.**
  - A baseline from *all* earlier nights: −0.004 kappa for the trees and −0.007 for the GRU.
  - A learned summary of the earlier nights: −0.006 against the same model given another
    person's nights.
  - Fine-tuning the GRU on the person's labelled nights beat fine-tuning on another person's
    nights by +0.02 to +0.035, but beat the population model by only +0.004 to +0.014, with CIs
    including 0. See [`notebooks/02b_personalization.ipynb`](notebooks/02b_personalization.ipynb).
- **N1 is the hard stage** (GRU F1 0.16). The probabilities are reasonably calibrated (ECE 0.04).
- **Robustness checks.** Re-pairing the labels by the ~90 s offset found in Phase 1, or training on
  the automatic Dreem labels instead of the expert labels, changes kappa by at most 0.012.
- **Two nights have labels offset from the watch by 31 and 59 min.** They are kept, and they
  affect kappa by 0.005 at most.

Details, figures and caveats are in [`notebooks/02_staging.ipynb`](notebooks/02_staging.ipynb).

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
