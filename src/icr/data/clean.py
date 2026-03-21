from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from icr.config import PipelineConfig


@dataclass
class CleaningArtifacts:
    numeric_medians: dict[str, float]


def fit_cleaning(df: pd.DataFrame, cfg: PipelineConfig) -> CleaningArtifacts:
    work = df.copy()
    if cfg.cleaning.age_zero_as_missing and "age" in work.columns:
        work.loc[work["age"] == 0, "age"] = np.nan

    numeric = work.select_dtypes(include=[np.number]).columns
    medians = {col: float(work[col].median()) for col in numeric}
    return CleaningArtifacts(numeric_medians=medians)


def transform_cleaning(
    df: pd.DataFrame,
    artifacts: CleaningArtifacts,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    out = df.copy()

    if cfg.cleaning.age_zero_as_missing and "age" in out.columns:
        out.loc[out["age"] == 0, "age"] = np.nan

    delinquency_cols = [
        c
        for c in out.columns
        if "delinquency" in c.lower()
        or "pastdue" in c.lower()
        or "late" in c.lower()
    ]

    for col in delinquency_cols:
        if cfg.cleaning.sensitive_delinquency_median_replace:
            invalid = out[col].isin(cfg.cleaning.delinquency_clip_values)
            if invalid.any():
                out.loc[invalid, col] = artifacts.numeric_medians.get(
                    col,
                    out[col].median(),
                )
        else:
            out[col] = out[col].replace(
                cfg.cleaning.delinquency_clip_values,
                cfg.cleaning.delinquency_clip_to,
            )

    for col, med in artifacts.numeric_medians.items():
        if col in out.columns:
            out[col] = out[col].fillna(med)

    return out
