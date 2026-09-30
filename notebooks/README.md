# Notebooks

- **Naming.** Use `NN_topic.ipynb` (for example `01_eda.ipynb`), numbered in reading order.
- **Loading data.** Go through the `sleepwatch` package; don't read raw files directly. That way
  notebooks and experiments see the same verified, aligned data.
- **Outputs.** Keep them, since they render on GitHub, but keep each notebook under 1 MB (a
  pre-commit hook enforces this). Prefer summary plots to raw dumps.
- **Results.** Notebooks explore and illustrate. Anything reported as a result comes from a
  config-driven experiment in `results/`.
- **Kernel.** Use the project's `.venv`: select `.venv/bin/python` in VS Code, or run
  `uv run python -m ipykernel install --user --name sleepwatch` once.
