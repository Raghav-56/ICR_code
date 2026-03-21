from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _unwrap_model(model):
    # CalibratedClassifierCV stores the fitted base estimator inside calibrated_classifiers_.
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        calibrated = model.calibrated_classifiers_[0]
        if hasattr(calibrated, "estimator"):
            return calibrated.estimator
    return model


def _is_tree_model(model) -> bool:
    mod = type(model).__module__.lower()
    return "xgboost" in mod or "lightgbm" in mod or "catboost" in mod


def explain_with_shap_local(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    *,
    kernel_background_size: int,
    kernel_use_kmeans: bool,
) -> pd.DataFrame:
    model = _unwrap_model(model)

    if _is_tree_model(model):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_sample)
    else:
        if kernel_use_kmeans:
            background = shap.kmeans(x_train, kernel_background_size)
        else:
            background = x_train.sample(
                n=min(kernel_background_size, len(x_train)),
                random_state=42,
            )
        explainer = shap.KernelExplainer(model.predict_proba, background)
        values = explainer.shap_values(x_sample, silent=True)

    arr = np.asarray(values)
    if isinstance(values, list):
        arr = np.asarray(values[1])
    elif arr.ndim == 3:
        arr = arr[:, :, 1]

    return pd.DataFrame(np.abs(arr), columns=x_sample.columns)


def summarize_local_attributions(local_abs: pd.DataFrame, method: str) -> pd.DataFrame:
    mean_abs = local_abs.mean(axis=0).to_numpy()
    out = pd.DataFrame(
        {
            "feature": local_abs.columns,
            "importance": mean_abs,
            "method": method,
        }
    ).sort_values("importance", ascending=False)
    return out.reset_index(drop=True)


def explain_with_shap(model, x_sample: pd.DataFrame) -> pd.DataFrame:
    local_abs = explain_with_shap_local(
        model,
        x_sample,
        x_sample,
        kernel_background_size=100,
        kernel_use_kmeans=True,
    )
    return summarize_local_attributions(local_abs, "shap")
