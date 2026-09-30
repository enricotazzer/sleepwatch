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
  review and correction by a sleep expert; who reviewed it and how is not documented.
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

From an audit of the 52 checksum-verified nights available on 2026-09-30 (9 subjects). Phase 1
repeats these checks on all 253 nights.

- **Signal files don't match the label window.** Heart rate and motion start between 7 min before
  and 48 min after `recStart`. They end anywhere from 154 min before the last label to 72 h after
  it, and raw files contain gaps of up to about 10 h. Crop to the label window and keep a coverage
  mask per epoch; don't interpolate across long gaps.
- **Heart-rate sampling varies by night.** The median interval is 2–5 s (not a uniform 0.2 Hz), so
  Bidslab00 night 2 has about twice the samples of its other nights. Features must not depend on
  the sampling rate. The data is still too coarse for HRV.
- **Label lengths can differ.** `Bidslab01/4` has 935 `dreem_label` epochs but only 771
  `expert_label` epochs.
- **Unknown epochs.** 175 of 45,125 expert epochs (0.4%, in 11 nights) are Unknown.
- **Duplicate timestamps.** Two nights repeat 1–2 heart-rate timestamps.
- **`recStart` time zone.** Interpreting `recStart` as US Eastern time is confirmed: signals start
  a median of 0.6 min before it. Recordings start between 21:00 and 02:00, so a start during the
  autumn DST change is ambiguous and has to be resolved explicitly, for example against the signal
  start time.
- **Dreem vs expert agreement** per night ranges from 0.22 to near 1 (median 0.82). Because the
  expert labels are corrections of the Dreem labels, this is not an independent inter-rater
  agreement.

## Caveats for any claims

- The reference is a Dreem-2 headband, not full polysomnography.
- All participants were healthy adults, so results may not transfer to clinical populations.
- Heart rate every 2–5 s supports heart-rate level and trend features, not HRV.

## Checking a local copy

`uv run sleepwatch data verify` hashes every listed file against `SHA256SUMS.txt`, reports which
subjects and nights are complete, and saves `data/interim/manifest.parquet`. Loaders only read
nights marked verified in that manifest. Rows turn `stale` if a file changes afterwards or
`data/raw` is repointed; re-run the command then.
