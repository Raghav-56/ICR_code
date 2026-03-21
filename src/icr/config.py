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
    model_name: str = "catboost"
    candidate_models: list[str] = Field(
        default_factory=lambda: ["logistic", "xgboost", "catboost"]
    )
    class_weight_strategy: str = "balanced"
    use_smote_in_cv: bool = False
    compare_imbalance_strategies: bool = True
    cv_folds: int = 5
    tune_hyperparameters: bool = True
    tuning_iterations: int = 20
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


class CatBoostConfig(BaseModel):
    iterations: int = 600
    depth: int = 6
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    random_seed: int = 42
    verbose: bool = False


class ExplainConfig(BaseModel):
    sample_size: int = 1000
    use_full_test_for_explanations: bool = True
    top_k: int = 3
    shap_kernel_background_size: int = 100
    shap_kernel_use_kmeans: bool = True
    lime_num_features: int = 10
    lime_num_samples: int = 1000
    lime_num_samples_sweep: list[int] = Field(default_factory=lambda: [500, 1000, 5000])
    lime_kernel_width: float | str = "sqrt_features"
    lime_feature_selection: str = "auto"
    lime_discretize_continuous: bool = True
    lime_seed_count: int = 50
    lime_seeds: list[int] = Field(
        default_factory=lambda: [
            11,
            19,
            37,
            71,
            97,
            101,
            103,
            107,
            109,
            113,
            127,
            131,
            137,
            139,
            149,
            151,
            157,
            163,
            167,
            173,
            179,
            181,
            191,
            193,
            197,
            199,
            211,
            223,
            227,
            229,
            233,
            239,
            241,
            251,
            257,
            263,
            269,
            271,
            277,
            281,
            283,
            293,
            307,
            311,
            313,
            317,
            331,
            337,
            347,
            349,
        ]
    )


class StabilityConfig(BaseModel):
    perturbation_sigma: float = 0.01
    perturbation_repeats: int = 3
    bootstrap_samples: int = 1000


class ScalabilityConfig(BaseModel):
    sizes: list[int] = Field(
        default_factory=lambda: [1, 10, 50, 100, 300, 500, 1000, 2500, 5000, 10000, 22500]
    )
    repeats: int = 5


class PipelineConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    catboost: CatBoostConfig = Field(default_factory=CatBoostConfig)
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
