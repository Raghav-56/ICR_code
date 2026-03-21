from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    run_name: str = "credit_shap_lime"
    random_seed: int = 42


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"


class DataConfig(BaseModel):
    input_file: str = "cs-training.csv"
    target_col: str = "SeriousDlqin2yrs"
    test_size: float = 0.15
    val_size: float = 0.15
    id_columns: list[str] = Field(default_factory=list)


class CleaningConfig(BaseModel):
    age_zero_as_missing: bool = True
    delinquency_clip_values: list[int] = Field(
        default_factory=lambda: [96, 98]
    )
    delinquency_clip_to: int = 12
    sensitive_delinquency_median_replace: bool = False


class ModelsConfig(BaseModel):
    model_name: str = "xgboost"
    class_weight_strategy: str = "balanced"
    use_smote_in_cv: bool = False
    cv_folds: int = 5
    calibrate: bool = True


class XGBoostConfig(BaseModel):
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"


class ExplainConfig(BaseModel):
    sample_size: int = 1000
    top_k: int = 10
    lime_num_features: int = 10
    lime_num_samples: int = 1000
    lime_kernel_width: float = 0.75
    lime_feature_selection: str = "auto"
    lime_discretize_continuous: bool = True
    lime_seeds: list[int] = Field(default_factory=lambda: [11, 19, 37, 71, 97])


class StabilityConfig(BaseModel):
    perturbation_sigma: float = 0.01
    perturbation_repeats: int = 3
    bootstrap_samples: int = 200


class ScalabilityConfig(BaseModel):
    sizes: list[int] = Field(default_factory=lambda: [100, 300, 500, 1000])


class PipelineConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    scalability: ScalabilityConfig = Field(default_factory=ScalabilityConfig)


def _deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str) -> PipelineConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        user_data = yaml.safe_load(f) or {}

    if "extends" in user_data:
        parent = Path(user_data.pop("extends"))
        if not parent.is_absolute():
            parent = path.parent / parent
        with parent.open("r", encoding="utf-8") as f:
            parent_data = yaml.safe_load(f) or {}
        final_data = _deep_merge(parent_data, user_data)
    else:
        final_data = user_data

    return PipelineConfig.model_validate(final_data)
