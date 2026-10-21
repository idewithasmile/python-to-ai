# Data Directory

This directory is the local workspace for datasets and derived artifacts. Its contents are ignored by Git so that the repository stays small, reproducible, and safe to share.

## Rules

- Never commit datasets larger than 50 MB, model weights, or generated media.
- Never commit personal, confidential, or unlicensed data.
- Check and record the license and attribution for every dataset.
- Prefer download scripts or documented source links over committed copies.
- Keep a small, redistributable sample only when it is essential for tests.
- Store secrets and dataset credentials in environment variables, never in notebooks.

## Suggested Layout

```text
data/
├── raw/          # Original files; never edit in place
├── interim/      # Temporary cleaned or merged data
├── processed/    # Model-ready datasets
└── samples/      # Tiny redistributable examples
```

The `data/*` rule in `.gitignore` ignores everything except this README. Create the subdirectories locally as needed.

## Dataset Card

When a project uses a dataset, add a short dataset card to the project README with:

- Name, source, and version or access date.
- License and redistribution constraints.
- Target variable, features, and label definitions.
- Train, validation, and test split strategy.
- Known biases, missing values, and leakage risks.
- Preprocessing and download command.
- Checksums or file sizes for reproducibility.

## Model and Experiment Artifacts

Keep checkpoints, predictions, embeddings, and experiment outputs in ignored directories such as `models/`, `runs/`, or `outputs/`. Commit only lightweight reports, plots, metrics, and configuration files needed to reproduce the result.
