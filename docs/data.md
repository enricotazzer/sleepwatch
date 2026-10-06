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

From all 253 nights, checksum-verified on 2026-10-01, via the loader, the quality table
(`data/processed/quality_v1.csv`) and notebook `01_eda`. Recordings run from 2021-07-26 to
2022-09-18.

- **Signal files don't match the label window.** Heart rate and motion start between 39 min before
  and 122 min after `recStart`. They end anywhere from 4.5 h before the last label to 95 h after
  it. The loader crops to the label window, and coverage masks mark the gaps; gaps are never
  interpolated. 2.7% of labelled epochs have no heart rate and 5.0% have no motion.
- **Some nights have little usable signal.** 44 nights are under 90% coverage on one signal:
  - 7 have motion for under a third of the night (`Bidslab42/3` 1%, `Bidslab49/1` 12%,
    `Bidslab08/7` 19%, `Bidslab16/2` 24%, `Bidslab17/3` 30%, `Bidslab42/1` 30%,
    `Bidslab60/1` 32%);
  - on 12 nights the watch stops about an hour before the labels end, while the labels show the
    person still asleep (mostly N2/N1);
  - `Bidslab01/3` stops 154 min early and `Bidslab34/1` 272 min early.
- **Truncated last rows.** `Bidslab42/3` (`hr.csv`, `motion.csv`) and `Bidslab42/4`
  (`motion.csv`) end in a partial record. The loader skips rows with the wrong number of fields
  and counts them (`*_malformed_rows`).
- **Sampling varies by night.** Inside the label window, heart rate comes every 5 s on 252 nights
  and every 2 s on `Bidslab06/2` (`hr_window_median_dt_s` in the quality table). An earlier
  version of these notes named `Bidslab00/2`; that night only has extra readings outside the
  window. Motion runs at about 50 Hz, with four nights at about 33 Hz and four at about 64 Hz,
  plus some jitter. Features are computed on uniform grids so they don't depend on the rate.
  The data is too coarse for HRV.
- **`Bidslab06/2` looks like two interleaved heart-rate streams.** Consecutive readings alternate
  between two levels about 10 bpm apart (for example 62 and 71 bpm), so the typical step between
  readings is 12 bpm against 0–1 bpm on every other night (`hr_window_median_step_bpm`). Per-epoch
  means average the two streams, and the within-epoch HR spread is inflated. No correction is
  applied; the night is flagged.
- **`Bidslab43/3` has a stretch of implausible heart rate (found in Phase 3).** From about 31 to
  104 min after the start, HR sits flat at about 130, then 165–172, then 115, then 110 bpm. It
  jumps by 40–90 bpm between readings a few seconds apart. While HR is above 110 bpm, the expert
  labels are mostly N3 (95 of 132 epochs, plus 16 N2 and 21 Wake). This is the only night with
  more than 2 readings at 150 bpm or more (it has 162). Heart rate doesn't behave like this in
  sleep, so the stretch is most likely a measurement artifact. Isolated jumps of 40 bpm or more
  within 30 s occur on 32 of 253 nights, but outside `Bidslab06/2` (83) and this night (9) there
  are at most 5 per night. No correction is applied, and the quality table doesn't flag the
  night. In Phase 3 it sets the training nights' extreme for the 30-min HR channel in three folds,
  and it is one of the five real nights flagged (notebook `03_anomaly`, section 7).
- **Label lengths can differ.** `Bidslab01/4` has 935 Dreem epochs against 771 expert epochs, and
  `Bidslab30/6` has 849 against 851. Dreem labels are cut or padded to the expert length.
- **Unknown epochs.** 2,363 expert epochs (1.1%, in 62 nights) are Unknown; they are masked.
- **Duplicate timestamps.** 11 nights repeat a few heart-rate timestamps; their values are
  averaged.
- **`recStart` time zone.** Interpreting it as US Eastern time is confirmed by the signals starting
  within about a minute of it on typical nights. No recording starts in a DST-ambiguous hour; the
  loader resolves such cases against the signal start anyway. All subjects' nights are in date
  order.
- **Two nights' labels start later than `recStart` says (found in Phase 2).** On `Bidslab42/1`
  and `Bidslab68/2` the watch starts 30.7 and 59.2 min after `recStart`, and the labels appear to
  start then too. Pairing each night's cross-validated GRU predictions with the expert labels at
  every lag up to ±75 min, agreement peaks at +62 and +118 epochs: kappa rises from −0.21 to 0.57
  and from −0.05 to 0.32. Those lags are the watch start delays to within one epoch. On every
  other night the best lag gains at most 0.06 kappa over lag 0, so the curve is flat around the
  documented alignment (203 of those 251 nights peak within ±3 epochs). Late watch starts alone
  don't imply this: `Bidslab14/4` (122 min late) and `Bidslab02/4` (48 min) are aligned at lag 0.
  So it can't be detected without labels, and neither night is near a DST change. No correction is
  applied; notebook `02_staging` (section 5) shows the scan and the scores without these two
  nights.
- **Dreem vs expert agreement.** Pooled Cohen's kappa is 0.75 (5-class) and 0.82 (4-class), with a
  per-night median of 0.75. Most disagreement is N1. Because the expert labels are corrections of
  the Dreem labels, this is not an independent inter-rater agreement.
- **The expert labels appear to run about 90 s late (open issue).** With the documented alignment,
  activity and heart rate separate expert-Wake from sleep best when label epoch `k` is paired with
  signal epoch `k-3`. That holds on the averaged curve and as the per-night median, with most
  subjects between −2 and −5. The offset is the same in the first and second halves of the night,
  so it is constant rather than clock drift. Dreem labels peak at −1 for activity and 0 for heart
  rate, consistent with movement slightly preceding EEG-scored Wake. The expert labels match Dreem
  best when shifted by 1–2 epochs. The documentation says both share `recStart` (recorded on one
  iPhone) and doesn't mention any offset. `sleepwatch.data.label_timing` reproduces the check, and
  notebook `01_eda` (section 9) shows it. Phase 2 keeps the documented alignment for the headline
  results and retrains with the shifted labels as a sensitivity analysis: the lag estimated on
  each fold's training subjects is −3 in every fold, and scores change by at most 0.012 kappa
  (notebook `02_staging`, section 7).
- **On `Bidslab01/4` the Dreem labels, not the expert labels, are the odd ones out.** The two
  agree on 22% of epochs (kappa −0.03), and no shift repairs this. Phase 1 suspected the expert
  labels. Phase 2's cross-validated watch models agree with the expert labels at a typical level
  (GRU kappa 0.50, boosting 0.37) and with the Dreem labels much less (0.11 and 0.25). The Dreem
  file is also longer (935 vs 771 epochs), so it probably comes from a different recording window.
  The night is kept, as decided in Phase 1. `Bidslab47/2` (kappa 0.47) is the only other night
  under 0.5 and behaves like a genuinely hard night.

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
