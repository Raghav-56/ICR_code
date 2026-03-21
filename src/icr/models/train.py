from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score

from icr.config import PipelineConfig
from icr.models.calibrate import calibrate_sigmoid
from icr.models.registry import build_model


@dataclass
class TrainingArtifacts:
    model: object
    feature_columns: list[str]
    selected_strategy: str
    cv_best_score: float | None


def _split_xy(df: pd.DataFrame, target_col: str):
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


def _param_distributions(cfg: PipelineConfig, model_name: str | None = None) -> dict[str, list]:
    name = (model_name or cfg.models.model_name).lower()
    if name == "logistic":
        return {
            "C": [0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
        }
    if name == "catboost":
        return {
            "iterations": [300, 600, 900],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "l2_leaf_reg": [1.0, 3.0, 5.0],
        }
    return {
        "n_estimators": [300, 500, 700],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }


def _fit_one_strategy(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: PipelineConfig,
    strategy: str,
    model_name: str | None = None,
) -> tuple[object, float | None]:
    model = build_model(cfg, model_name=model_name)
    cv = StratifiedKFold(
        n_splits=cfg.models.cv_folds,
        shuffle=True,
        random_state=cfg.project.random_seed,
    )

    if strategy == "smote_inside_cv":
        estimator = ImbPipeline(
            steps=[
                ("smote", SMOTE(random_state=cfg.project.random_seed)),
                ("model", model),
            ]
        )
        params = {
            f"model__{k}": v
            for k, v in _param_distributions(cfg, model_name=model_name).items()
        }
    else:
        estimator = model
        params = _param_distributions(cfg, model_name=model_name)

    best_score: float | None = None
    fitted: object
    if cfg.models.tune_hyperparameters:
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=params,
            n_iter=cfg.models.tuning_iterations,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            random_state=cfg.project.random_seed,
            refit=True,
        )
        search.fit(x_train, y_train)
        fitted = search.best_estimator_
        best_score = float(search.best_score_)
    else:
        fitted = estimator.fit(x_train, y_train)

    return fitted, best_score


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: PipelineConfig,
    model_name: str | None = None,
) -> TrainingArtifacts:
    x_train, y_train = _split_xy(train_df, cfg.data.target_col)
    x_val, y_val = _split_xy(val_df, cfg.data.target_col)

    candidate_strategies = ["class_weight"]
    if cfg.models.compare_imbalance_strategies or cfg.models.use_smote_in_cv:
        candidate_strategies.append("smote_inside_cv")

    selected_model: object | None = None
    selected_strategy = candidate_strategies[0]
    selected_score: float | None = None

    for strategy in candidate_strategies:
        fitted, score = _fit_one_strategy(
            x_train,
            y_train,
            cfg,
            strategy,
            model_name=model_name,
        )
        if score is None and hasattr(fitted, "predict_proba"):
            val_prob = fitted.predict_proba(x_val)[:, 1]
            score = float(average_precision_score(y_val, val_prob))
        if selected_model is None:
            selected_model = fitted
            selected_strategy = strategy
            selected_score = score
            continue
        if score is not None and (
            selected_score is None or score > selected_score
        ):
            selected_model = fitted
            selected_strategy = strategy
            selected_score = score

    model = selected_model
    if model is None:
        raise RuntimeError("Training failed: no model was selected.")

    if cfg.models.calibrate and hasattr(model, "predict_proba"):
        model = calibrate_sigmoid(model, x_val, y_val)

    return TrainingArtifacts(
        model=model,
        feature_columns=list(x_train.columns),
        selected_strategy=selected_strategy,
        cv_best_score=selected_score,
    )
