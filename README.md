# Interpreting Credit Risk - Experiment Code

Code for comparing SHAP and LIME on credit default prediction under strict timeline constraints.

## Why this setup

- Uses `uv` for fast, reproducible dependency management.
- Uses `Typer` for a clean multi-command CLI.
- Keeps artifacts and configs explicit for paper-ready reporting.

## Quick start

1. Install uv: <https://docs.astral.sh/uv/>
1. From this folder:

```bash
uv sync
```

1. Configure run settings:

```bash
cp .env.example .env
```

1. Run full pipeline:

```bash
uv run icr run-all --config configs/base.yaml
```

## Command overview

- `uv run icr prepare-data --config configs/base.yaml`
- `uv run icr train --config configs/base.yaml`
- `uv run icr evaluate --config configs/base.yaml`
- `uv run icr explain --config configs/base.yaml`
- `uv run icr stability --config configs/base.yaml`
- `uv run icr scalability --config configs/base.yaml`
- `uv run icr report --config configs/base.yaml`

## Data assumptions

Primary expected dataset: Give Me Some Credit CSV with target column `SeriousDlqin2yrs`.

Input CSV in `data/raw/` and filename in config.

## Output structure

- `artifacts/models/`: trained model and metadata
- `artifacts/metrics/`: predictive metrics and calibration
- `artifacts/explanations/`: SHAP/LIME exports
- `artifacts/stability/`: seed and perturbation analyses
- `artifacts/scalability/`: runtime and memory benchmarks
- `artifacts/reports/`: paper-ready markdown summary
