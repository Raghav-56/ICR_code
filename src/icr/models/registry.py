from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from icr.config import PipelineConfig

def build_model(cfg: PipelineConfig) -> Any:
    name = cfg.models.model_name.lower()

    if name == "logistic":
        return LogisticRegression(
            class_weight=cfg.models.class_weight_strategy,
            max_iter=2000,
            random_state=cfg.project.random_seed,
            n_jobs=-1,
        )

    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=cfg.xgboost.n_estimators,
            max_depth=cfg.xgboost.max_depth,
            learning_rate=cfg.xgboost.learning_rate,
            subsample=cfg.xgboost.subsample,
            colsample_bytree=cfg.xgboost.colsample_bytree,
            reg_lambda=cfg.xgboost.reg_lambda,
            class_weight=cfg.models.class_weight_strategy,
            random_state=cfg.project.random_seed,
            n_jobs=-1,
            verbose=-1,
        )

    return XGBClassifier(
        n_estimators=cfg.xgboost.n_estimators,
        max_depth=cfg.xgboost.max_depth,
        learning_rate=cfg.xgboost.learning_rate,
        subsample=cfg.xgboost.subsample,
        colsample_bytree=cfg.xgboost.colsample_bytree,
        reg_lambda=cfg.xgboost.reg_lambda,
        random_state=cfg.xgboost.random_state,
        n_jobs=cfg.xgboost.n_jobs,
        tree_method=cfg.xgboost.tree_method,
        eval_metric="logloss",
    )
