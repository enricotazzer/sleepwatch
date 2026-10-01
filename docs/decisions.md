# Decision log

Newest entries last. Each entry gives the decision and the reason for it.

## 2026-09-30 — Phase 0

1. **Repo on the internal disk, data on the external T7 drive.** The T7 is exFAT: it has no
   symlinks (so virtual environments break) and macOS writes `._*` files onto it. The internal
   disk has about 22 GB free, but the unpacked dataset is 27.9 GB.
2. **`data/` holds symlinks.** It's a gitignored directory containing `raw` (a symlink to the
   dataset folder), `interim` (a symlink to `/Volumes/T7/sleepwatch-data/interim`, since caches
   run to several GB) and a local `processed/`. The gitignore entry is `/data` with no trailing
   slash, because a `data/` pattern doesn't match symlinks. Each path can be overridden with a
   `SLEEPWATCH_*` variable.
3. **Loaders read only checksum-verified nights.** The manifest is built from PhysioNet's
   `SHA256SUMS.txt`. A row turns stale when the file's size or mtime changes or the raw folder is
   repointed. Reason: the first local download was silently incomplete, with 9 of 47 subjects
   intact and the last file NUL-padded.
4. **Tooling.** uv with a lockfile and Python 3.12; ruff; pytest; pre-commit. The pre-commit ruff
   hook uses the version in the lockfile, and a separate hook refuses data files.
5. **Where settings live.** `config.Settings` holds only machine-specific paths. Seeds and
   experiment defaults live in `configs/base.yaml`, so each value has a single source.
6. **License.** The code is MIT.
7. **Evaluation protocol.**
   - Subject-level 5-fold `GroupKFold`, seeded, with the fold file saved once and reused.
   - Tuning is nested inside the training subjects.
   - Metrics are reported per subject as mean ± SD, plus pooled, with subject-bootstrap 95% CIs.
   - The N-nights curve (N = 0–3) is scored on fixed evaluation nights: night 4 onward for the 43
     subjects with at least 4 nights (112 nights), adapting on nights 1..N. Every N is therefore
     scored on the same nights.
8. **Personalization.** The headline variant needs no labels, because a real user has no EEG
   labels. It uses prior-night normalization or subject embeddings learned from unlabelled nights.
   Fine-tuning on the prior nights' expert labels is reported only as an upper bound.
9. **Leakage rules** are listed in `CLAUDE.md`. They are enforced by tests where possible.
10. **Experiment tracking.**
    - Each experiment is a YAML config validated by pydantic and run with one CLI command.
    - Outputs go to `results/<experiment>/<timestamp>/`: the resolved config, git SHA, seed,
      `metrics.json` and a per-subject CSV. An index CSV lists all runs.
    - No MLflow for now.
11. **Notebooks keep their outputs** so they render on GitHub. A 1 MB large-file hook limits bloat.
12. **CI.** GitHub Actions runs ruff and the tests that need no data.
13. **Still open.** The frontend (Streamlit vs React) is decided at Phase 5 and the LLM models at
    Phase 4. In Phase 4, Ollama models are stored on the T7 via `OLLAMA_MODELS`.

## 2026-09-30 — Phase 1

14. **No cache of raw signals.** A night's CSVs parse in 0.1–0.2 s with the pyarrow engine, so
    only the epoch table is cached. `data/interim` holds just the manifest for now.
15. **Features on uniform grids.** Heart rate is put on a 1 Hz grid and motion on a 50 Hz grid,
    with gaps never bridged, because the watch's sampling rates differ between nights. Tests check
    that features agree at 2 s vs 5 s heart rate and at 50 vs 33 Hz motion.
16. **Coverage is not a feature.** `qc_*` columns (coverage, raw sample counts) are for masking and
    diagnostics only; the HR sample count is slightly lower in Wake and would be a device shortcut.
17. **Features never read labels.** A test checks that shuffling labels leaves every feature
    unchanged. Sleep onset is estimated from arm-angle stillness.
18. **Whole-night features are accepted.** Night-relative heart rate, detrending, centred windows
    and the zero-phase filter suit next-morning analysis. A real-time variant would need causal
    versions.
19. **Label timing is measured, not assumed.** `data/label_timing.py` estimates label-to-signal
    offsets. It found the expert labels about 2–3 epochs late (see docs/data.md). No shift is
    applied until a decision in Phase 2.

## 2026-10-01 — Phase 1 on the full dataset

20. **The raw folder is the unzipped PhysioNet ZIP** at
    `/Volumes/T7/a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.1`.
    All 761 files pass their checksums. The old partial wget mirror is no longer used.
21. **Malformed CSV rows are skipped and counted, not fatal.** Three files in `Bidslab42` end in a
    partial record. Rows with the wrong number of fields are skipped and recorded as
    `*_malformed_rows` in the quality table; anything else unexpected still fails loudly.
22. **Label timing: documented alignment is primary (user decision).** Headline Phase 2 results use
    the dataset's stated alignment, so they stay comparable with other work on this dataset. A
    sensitivity analysis re-runs the key experiments with the expert labels shifted by the offset
    estimated on training subjects only, and reports the difference.
23. **`Bidslab01/4` stays in (user decision).** It is used like any other night, despite its expert
    labels disagreeing with Dreem at every lag. Results report its per-night scores so its effect is
    visible.
24. **Quality table gains in-window heart-rate statistics.** `hr_window_median_dt_s` and
    `hr_window_median_step_bpm` describe the readings inside the label window. The review notebook
    found that whole-file statistics had misattributed the 2 s night (it is `Bidslab06/2`, not
    `Bidslab00/2`), and that `Bidslab06/2` alternates between two heart-rate levels. That night is
    flagged, not corrected.

## 2026-10-01 — Phase 2 (sleep staging)

25. **Gradient boosting is scikit-learn's `HistGradientBoostingClassifier`, not LightGBM.**
    LightGBM's macOS wheel needs a system OpenMP library (Homebrew `libomp`). The scikit-learn
    model is the same algorithm family, handles missing values, accepts a separate validation
    set for early stopping, and adds no system dependency.
26. **Early stopping and tuning use held-out training subjects, never random epochs.** Random
    validation epochs would share nights with the training data. Boosting: a 6-point grid
    (leaves x class weighting) chosen by validation macro-F1, then refitted on all training
    subjects with the selected number of rounds.
27. **GRU size chosen for speed on fold 0's validation subjects.** One 48-unit bidirectional layer
    matched two 64-unit layers (validation macro-F1 0.536 vs 0.539) at about half the training
    time; CPU is faster than MPS for this model and is deterministic.
28. **The label-using upper bound is `prior_stage` features, not fine-tuning.** Continuing to
    boost on a person's own nights fails whenever a stage is missing from them (common for N1),
    and features keep both models comparable: per-stage heart rate and activity, and the stage
    mix, from the expert labels of the first N nights.
29. **Personalization is judged against a matched control.** A model that uses N prior nights can
    only train on nights after the first N, so comparing it with the full population model mixes
    personalization with a smaller training set. The `matched` variant trains on exactly the same
    nights without personal features; the paired difference against it is the headline
    personalization result.
30. **Metric conventions.** Macro-F1 averages over stages present in the true labels of the group
    scored; 4-class scores merge N1+N2 probabilities before the argmax; ECE is top-label with 15
    bins; CIs resample subjects (1,000 draws); personalization differences use a paired bootstrap.
    Hyperparameters optimize macro-F1, which favours rare stages over raw accuracy.
31. **The heart-rate sample count stays out of the features.** The dataset authors report that HR
    sampling frequency improves their model; Phase 1 found it slightly lower during Wake, i.e. a
    device artefact. Results here are therefore not directly comparable to theirs on accuracy.
