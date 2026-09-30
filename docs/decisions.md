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
