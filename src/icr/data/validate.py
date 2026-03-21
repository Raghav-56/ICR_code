from __future__ import annotations

import pandas as pd

from icr.config import PipelineConfig



def validate_dataframe(df: pd.DataFrame, cfg: PipelineConfig) -> None:
    if cfg.data.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.data.target_col}' not found in dataset.")

    if df[cfg.data.target_col].nunique() != 2:
        raise ValueError("Target column must be binary for this pipeline.")
