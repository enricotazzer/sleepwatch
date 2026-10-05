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
    time (measured with the packed-sequence implementation that decision 32 replaced).
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
32. **The GRU trains on CPU, with folds in parallel; MPS was measured and rejected.** On fold 0,
    one training epoch (batch of 8 nights) took 9.4 s on MPS against 2.4 s on one CPU thread, and
    MPS stayed slower at batches of 32 (3.0 vs 1.0 s) and 64 (1.8 vs 0.9 s): each night is ~1,000
    sequential steps, so the GPU waits on per-step dispatch. The bidirectional layer runs two
    forward GRUs, the second over each night reversed within its own length, instead of
    `nn.GRU(bidirectional=True)` on packed sequences: the same function (tested to 1e-6), 3x
    faster on CPU. One thread is as fast as four, so `--jobs 5` trains the five folds in
    parallel; each fold is seeded on its own, and a test checks that predictions don't depend on
    `--jobs`. `device: mps` remains available (same outputs to within 1e-6, not bitwise
    reproducible).
33. **Two nights with offset labels stay in, uncorrected.** `Bidslab42/1` and `Bidslab68/2`
    appear to have labels that start when the watch starts, 31 and 59 min after `recStart`
    (`docs/data.md`). This was found by scanning test-fold predictions, so correcting or dropping
    them now would change reported numbers on the strength of the test data. They stay in every
    reported result, and the effect of excluding them (kappa +0.003 to +0.005) is shown only as a
    diagnostic. Whether later phases correct them is open.
34. **Personalization is reported as a negative result, not redesigned.** No personalization
    variant beat the matched control. Changing the personal features after seeing these
    cross-validated results would be tuning on the test subjects. Any new design would be a new,
    pre-declared experiment that is reported alongside this one.

## Phase 2b: more personalization designs (protocol fixed before any run, 2026-10-01)

35. **Ground rules.** Three more personalization designs are tested with Phase 2's folds,
    seeds, evaluation nights (night 4 onward of the 43 subjects with at least 4 nights) and
    metrics. Each method's primary result is the pooled 5-class kappa difference against its
    control on the evaluation nights, with a paired subject bootstrap (1,000 draws). Every variant
    is reported whatever the outcome. No design changes after seeing test-fold results;
    hyperparameters are chosen on each fold's validation subjects only. The two offset nights
    stay in (decision 33). That makes about 12 comparisons with no multiplicity correction, so one
    CI excluding zero is weak evidence on its own.
36. **A · baseline from all earlier nights (label-free), `expanding_norm`.** For night *r*,
    `prior_norm`'s 12 features (decision 28's label-free counterpart) are computed from nights
    1..*r*−1 after the estimated onset. Night 1 has no baseline: NaN for the trees; zeros plus an
    availability flag for the GRU. Models train on all training nights, so the control is the
    population model trained on exactly the same nights. Both models; the trees reuse the fold's
    tuned grid point and the GRU uses Phase 2's settings.
37. **B · learned person summary (label-free for the person), `context`.**
    - **Encoder:** each earlier night's preprocessed epoch features go through a per-epoch
      linear layer (32 units, GELU), then mean and SD over the night, then a linear layer to 16
      numbers.
    - **Person summary:** the average over all strictly earlier nights. It is appended, with a
      has-context flag, to every epoch's input of the Phase 2 GRU (same size and training
      settings), and the whole model is trained end to end on training subjects' labels.
    - **Against memorizing training subjects:** dropout 0.3 on the summary, and during training
      each earlier night is dropped with probability 0.3.
    - **Controls:**
      - the population GRU;
      - `context_shuffled`: the same model given a different person's nights. Pairing is fixed and
        random, within the training subjects and within each fold's test subjects, using the
        partner's first *r*−1 nights or as many as they have.
38. **C · fine-tuning (uses labels; an upper bound), `finetune`.**
    - **Procedure:** for each test subject, a copy of the fold's population GRU is fine-tuned on
      their first N nights (N = 1, 2, 3) and scored on their evaluation nights. Separately, for
      each evaluation night *r*, a copy is fine-tuned on nights 1..*r*−1 (`all`).
    - **Training details:** full batch, AdamW, and no early stopping. Weight decay is replaced by
      0.01 × the squared distance from the population weights. Dropout is as in training.
    - **Grid:** learning rate {1e-4, 1e-3} × epochs {5, 20} × trained layers {output layer, all}.
      One grid point is picked per fold, by pooled 5-class kappa when the same procedure (all four
      variants) runs on the fold's validation subjects. The population GRU never fitted on them;
      they only set its early stopping.
    - **Controls:**
      - the population GRU;
      - `finetune_other`: fine-tuning on another test subject's nights of the same count. Pairing
        is fixed and random within the fold; the partner's first N or *r*−1 nights, or as many as
        they have.
39. **All Phase 2 experiments are rerun at the Phase 2b commit.** Phase 2b changes the experiment
    code. Rerunning puts every reported run on one commit, and the review notebook checks that
    the reruns reproduce the earlier Phase 2 predictions exactly.
40. **Phase 2b outcome (recorded after the runs; the protocol above was not changed).** Neither
    label-free design beat its control:
    - expanding baseline: trees −0.004 [−0.011, +0.004], GRU −0.007 [−0.019, +0.005];
    - learned summary: −0.006 [−0.025, +0.018].

    Fine-tuning on the person's labelled nights beat fine-tuning on another person's nights by
    +0.022 to +0.035, with all CIs above 0. Against the population GRU the gain was only +0.004
    to +0.014, with all CIs including 0. All Phase 2 reruns reproduced the earlier predictions
    exactly. Details are in `notebooks/02b_personalization.ipynb`.

## Phase 3: personalized anomaly detection (protocol fixed before any run, 2026-10-05)

41. **Ground rules.**
    - **Exclusions:** `Bidslab42/1` and `Bidslab68/2` are excluded everywhere (user decision; see
      decision 33).
    - **Scored nights and baselines:** a night is scored if its subject has at least 2 earlier
      usable nights, and its baseline uses only those earlier nights.
    - **What is learned from data, and where:** population priors, shrinkage strengths and alarm
      thresholds all come from the outer fold's training subjects. Results are reported on its
      test subjects.
    - **Thresholds see scores made the way test scores are:** an inner 5-fold split of the
      training subjects gives them out-of-sample stage predictions (inner GRUs) and out-of-sample
      priors. This adds 25 inner GRU trainings to the 5 outer ones in the plan.
    - **No tuning, everything reported:** nothing below is tuned on test subjects or on injected
      nights, and every condition is reported. There are no causal, psychological or HRV claims.
42. **Detector.**
    - **Signals:** epoch heart rate (`hr_mean`) and log₁₀ activity.
    - **Time of night:** bins of hours since the actigraphy onset: before onset, 0–1.5, 1.5–3,
      3–4.5, 4.5–6 and 6 or more.
    - **Expected value** for person *p*, stage *s* and time bin *t*: μ_pop(s,t) + b_p + b_{p,s}.
      - The *b* terms are averages of per-night mean residuals over the earlier nights, shrunk by
        N/(N+κ).
      - κ comes from the training subjects by method of moments.
      - z = residual / per-stage residual SD, where the SD comes from training subjects.
    - **Night channels**, all one-sided in the "worse" direction:
      1. the highest 30-min mean HR z in the sleep period;
      2. the mean HR z over the sleep period;
      3. wake bouts of 1 min or more per hour after sleep onset;
      4. onset latency, where onset is the first stretch of 10 min or more of continuous sleep.
    - **Channels 3–4 are z-scored against the person's earlier nights:** the mean is shrunk as
      above, and the SD is the pooled within-person SD between nights.
    - **The stage-free variant** treats all epochs as one stage. It counts movement bursts (runs
      with activity z > 3) instead of wake bouts, and uses the actigraphy onset.
    - **Alarm:**
      - each channel gets a tail probability under the null, which is the training subjects'
        clean nights;
      - a night's score is its smallest tail probability;
      - the night is flagged when that score is at or below its 5th percentile under the null,
        which makes the false-alarm rate 5% on training subjects.
43. **Variants.**
    - **Stages:** from the population GRU (primary), the expert labels (best case), or none.
    - **Baseline:** personal (primary) or population-only (b = 0).
    - That gives 6 detectors in all.
44. **Injections.**
    - **How:** anomalies are injected into the raw HR and accelerometer samples of test nights.
      Features are rebuilt with the unchanged pipeline, and stages are re-predicted by the fold's
      GRU.
    - **HR elevation:** +3, +6 or +10 bpm for 30 min, 2 h or the whole sleep period. Edges ramp
      over 2 min, and placement is random within the expert sleep period.
    - **Fragmentation:** 2, 4 or 8 awakenings of 3 min, at least 15 min apart. Their HR and
      accelerometer samples are copied from the night's own expert-Wake stretches (6+ epochs with
      at least 90% coverage), and the labels there become Wake.
    - **Delayed onset:** +20, +40 or +60 min after the expert persistent onset. That time is
      filled with wake copied from the same night, and the labels there become Wake.
    - **Versions and seeding:** each injected version holds one anomaly, and placement is seeded.
    - **Skipped nights:** nights without a usable wake stretch skip fragmentation and onset
      injections. There are 5 after the exclusions, and they are counted.
45. **Metrics.**
    - **Primary detector** (GRU stages + personal baseline), per type × size:
      - recall at the threshold;
      - AUROC of injected vs clean versions of the same nights;
      - 95% CIs by resampling subjects.
    - **False alarms:** the rate on clean test nights, against the nominal 5%.
    - **Precision** at an assumed 10% prevalence, derived from recall and the false-alarm rate.
    - **Localization:**
      - HR: whether the centre of the flagged 30-min window falls inside the injected window;
      - fragmentation: the share of injected awakenings that overlap a detected wake bout.
    - **Type:** whether the matching channel has the smallest tail probability.
    - **Comparisons:**
      - personal vs population-only baseline, paired;
      - GRU vs expert vs no stages;
      - recall by number of baseline nights.
    - **Real nights:** flagged clean test nights are described with no causal claims, and exported
      as structured summaries for Phase 4.
