from __future__ import annotations

from pathlib import Path

import pandas as pd

from icr.reporting.tables import to_markdown_table

def build_report(
    out_path: Path,
    metrics_df: pd.DataFrame,
    explain_compare_df: pd.DataFrame,
    scalability_df: pd.DataFrame,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content = [
        "# Interpreting Credit Risk - Experiment Report",
        "",
        "## Predictive Metrics",
        to_markdown_table(metrics_df),
        "",
        "## SHAP vs LIME Agreement",
        to_markdown_table(explain_compare_df),
        "",
        "## Scalability",
        to_markdown_table(scalability_df),
        "",
        "## Notes",
        "- Replace placeholder metrics in paper after report generation.",
        "- Confirm final configuration and seeds in reproducibility section.",
    ]

    out_path.write_text("\n".join(content), encoding="utf-8")
