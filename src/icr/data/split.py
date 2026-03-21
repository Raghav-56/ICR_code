from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from icr.config import PipelineConfig


@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame



def stratified_split(df: pd.DataFrame, cfg: PipelineConfig) -> DataSplits:
    y = df[cfg.data.target_col]

    train_val, test = train_test_split(
        df,
        test_size=cfg.data.test_size,
        random_state=cfg.project.random_seed,
        stratify=y,
    )

    rel_val_size = cfg.data.val_size / (1.0 - cfg.data.test_size)
    train, val = train_test_split(
        train_val,
        test_size=rel_val_size,
        random_state=cfg.project.random_seed,
        stratify=train_val[cfg.data.target_col],
    )

    return DataSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
