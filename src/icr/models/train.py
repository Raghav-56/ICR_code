from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from imblearn.over_sampling import SMOTE

from icr.config import PipelineConfig
from icr.models.calibrate import calibrate_sigmoid
from icr.models.registry import build_model


@dataclass
class TrainingArtifacts:
    model: object
    feature_columns: list[str]


def _split_xy(df: pd.DataFrame, target_col: str):
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> TrainingArtifacts:
    x_train, y_train = _split_xy(train_df, cfg.data.target_col)
    x_val, y_val = _split_xy(val_df, cfg.data.target_col)

    if cfg.models.use_smote_in_cv:
        x_train, y_train = SMOTE(
            random_state=cfg.project.random_seed
        ).fit_resample(x_train, y_train)

    model = build_model(cfg)
    model.fit(x_train, y_train)

    if cfg.models.calibrate and hasattr(model, "predict_proba"):
        model = calibrate_sigmoid(model, x_val, y_val)

    return TrainingArtifacts(
        model=model,
        feature_columns=list(x_train.columns),
    )
