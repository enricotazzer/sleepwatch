# Dataset notes: BIDSleep v1.0.1

## Source

- *A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels*,
  PhysioNet, version 1.0.1 (released 2026-09-25), ODC-BY 1.0.
  <https://physionet.org/content/bidsleep-dataset/1.0.1/>. DOI (v1.0.1): 10.13026/rees-1092;
  DOI (latest version): 10.13026/hace-p847.
- **Devices.** An Apple Watch running a custom watchOS app (BIDSleep) recorded instantaneous heart
  rate and 3-axis accelerometry. A Dreem-2 EEG headband provided the sleep-staging reference:
  Wake, N1, N2, N3, REM, scored in 30 s epochs to AASM rules.
- **Labels.** `dreem_label` is the Dreem automatic staging. `expert_label` is that staging after
  epoch-by-epoch review and correction by a single trained sleep expert following AASM guidelines,
  so there was no inter-scorer adjudication. Both devices recorded through the same iPhone to share
  a time reference.
- **Size.** 47 healthy adults, 3–7 nights each, 253 nights in total. Four subjects have only 3
  nights (Bidslab15, 38, 49, 63); the other 43 have at least 4. The ZIP is 6.35 GB and unpacks to
  27.9 GB. `SHA256SUMS.txt` lists 761 files: 759 night files plus `README.md` and `LICENSE.txt`.

## Files per night (`BidslabXX/<night>/`)

| File | Content |
|---|---|
| `hr.csv` | No header. Columns: Unix time (s), heart rate (bpm). |
| `motion.csv` | Header `Timestamp,x,y,z`. Unix time (s) and acceleration (g), about 50 Hz. |
| `labels.mat` | `recStart`: local US Eastern time as `YYYY-MM-DD HH:MM:SS`. `dreem_label`, `expert_label`: uint8 vectors, one value per 30 s epoch starting at `recStart`. |

Stage codes: 0 Wake, 1 N1, 2 N2, 3 N3, 4 REM, 5 Unknown. With a 0-based epoch index `k`, epoch `k`
covers `[recStart + 30k, recStart + 30(k+1))`. The dataset README uses 1-based
`k = floor((t - recStart) / 30) + 1`.

## Known quirks

From the 52 checksum-verified nights available on 2026-09-30 (9 subjects), via the Phase 1
loader, the quality table and notebook `01_eda`. They must be re-checked once all 253 nights are
verified.

- **Signal files don't match the label window.** Heart rate and motion start between 7 min before
  and 48 min after `recStart`. They end anywhere from 154 min before the last label to 72 h after
  it, and raw files contain gaps of up to about 10 h. Crop to the label window and keep a coverage
  mask per epoch; don't interpolate across long gaps.
- **Sampling varies by night.** Heart rate comes every 5 s on 51 of the 52 nights and every 2 s on
  `Bidslab00/2`. Motion runs at about 50 Hz, except three nights at about 33 Hz and one at about
  64 Hz, with some jitter. Features are computed on uniform grids so they don't depend on the rate.
  The data is too coarse for HRV.
- **Partial coverage inside the window.** 1.6% of epochs have no heart rate and 3.4% no motion.
  The watch stopped 154 min early on `Bidslab01/3`, and `Bidslab08/7` has motion for only 19% of
  the night.
- **Label lengths can differ.** `Bidslab01/4` has 935 `dreem_label` epochs but only 771
  `expert_label` epochs.
- **Unknown epochs.** 175 of 45,125 expert epochs (0.4%, in 11 nights) are Unknown.
- **Duplicate timestamps.** Two nights repeat 1–2 heart-rate timestamps.
- **`recStart` time zone.** Interpreting `recStart` as US Eastern time is confirmed: signals start
  a median of 0.6 min before it. Recordings start between 21:00 and 02:00, so a start during the
  autumn DST change is ambiguous and has to be resolved explicitly, for example against the signal
  start time.
- **Dreem vs expert agreement.** Pooled Cohen's kappa is 0.74 (5-class) and 0.81 (4-class), with a
  per-night median of 0.75. Most disagreement is N1, which Dreem calls N2 60% of the time. Because
  the expert labels are corrections of the Dreem labels, this is not an independent inter-rater
  agreement.
- **The expert labels appear to run 60–90 s late (open issue).** With the documented alignment,
  activity and heart rate separate expert-Wake from sleep best when label epoch `k` is paired with
  signal epoch `k-2` or `k-3`; this holds on 49 of 51 nights. Dreem labels line up within one
  epoch, and expert labels match Dreem best when shifted by 1–2 epochs. The documentation doesn't
  mention an offset. `sleepwatch.data.label_timing` reproduces the check, and notebook
  `01_eda` (section 9) shows it. No correction is applied yet; that decision belongs to Phase 2.
- **`Bidslab01/4` has suspect expert labels.** They cover 771 epochs against Dreem's 935, agree with
  Dreem on 22% of epochs (kappa −0.03), and no shift within ±200 epochs repairs this. The proposal
  is to exclude this night from evaluation.

## Epoch table (feature set `v1`)

`uv run sleepwatch data build-epochs` writes `data/processed/epochs_v1.parquet` (one row per
labelled epoch), `quality_v1.csv` (one row per night) and `build_v1.json` (config, git commit,
counts). The feature definitions are in `configs/features/v1.yaml` and
`sleepwatch/features/epoch_features.py`.

- **Metadata and labels:** `subject`, `night`, `epoch`, `t_start`, `expert`, `dreem`, and `labeled`
  (expert label is not Unknown).
- **Coverage columns (`qc_*`):** the share of each epoch with heart rate or motion, and the raw HR
  sample count. They are for masking and diagnostics, never model inputs, because the sample count
  is slightly lower in Wake and would act as a device shortcut.
- **Heart rate** (1 Hz grid, gaps over 30 s not bridged, NaN when coverage is under 50%): mean,
  SD, min and max, value relative to the night's median, robust z-score, detrended value (centred
  ~20 min rolling median), and first difference.
- **Motion** (50 Hz grid, gaps over 1 s not bridged): ENMO (Euclidean norm minus one) mean and max,
  magnitude SD, activity (mean absolute 0.25–2.5 Hz band-passed magnitude), arm angle, and mean and
  max arm-angle change between 5 s blocks.
- **Time:** hours since recording start, hours since estimated sleep onset, and hours since noon
  (local time). Onset is the first run of 10 min in which the arm angle changes by less than 5°
  per block. It is label-free and up to one epoch late by construction.
- **Context:** centred rolling mean and SD over ±2, ±5 and ±10 epochs for heart-rate mean,
  detrended heart rate, ENMO, activity and arm-angle change.
- **Scope:** night-relative heart rate, detrending, centred windows and the zero-phase filter all
  use the whole night. That suits next-morning analysis, not real-time staging.

## Caveats for any claims

- The reference is a Dreem-2 headband, not full polysomnography.
- All participants were healthy adults, so results may not transfer to clinical populations.
- Heart rate every 2–5 s supports heart-rate level and trend features, not HRV.

## Checking a local copy

`uv run sleepwatch data verify` hashes every listed file against `SHA256SUMS.txt`, reports which
subjects and nights are complete, and saves `data/interim/manifest.parquet`. Loaders only read
nights marked verified in that manifest. Rows turn `stale` if a file changes afterwards or
`data/raw` is repointed; re-run the command then.
