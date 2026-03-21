from __future__ import annotations

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


def _parse_feature_name(raw: str) -> str:
    separators = [" <=", " >=", " <", " >", " ="]
    for sep in separators:
        if sep in raw:
            return raw.split(sep, 1)[0].strip()
    return raw.strip()


def _resolve_kernel_width(kernel_width: float | str | None, n_features: int) -> float:
    if kernel_width is None:
        return float(np.sqrt(n_features))
    if isinstance(kernel_width, str):
        if kernel_width.lower() == "sqrt_features":
            return float(np.sqrt(n_features))
        return float(kernel_width)
    return float(kernel_width)


def explain_with_lime_local(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    *,
    num_features: int,
    num_samples: int,
    kernel_width: float | str | None,
    feature_selection: str,
    discretize_continuous: bool,
    random_state: int,
) -> pd.DataFrame:
    resolved_kernel_width = _resolve_kernel_width(kernel_width, x_train.shape[1])
    explainer = LimeTabularExplainer(
        training_data=x_train.values,
        feature_names=list(x_train.columns),
        class_names=["no_default", "default"],
        mode="classification",
        discretize_continuous=discretize_continuous,
        kernel_width=resolved_kernel_width,
        random_state=random_state,
        feature_selection=feature_selection,
    )

    local = pd.DataFrame(0.0, index=range(len(x_sample)), columns=x_train.columns)
    for i in range(len(x_sample)):
        instance = x_sample.iloc[i].values
        exp = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
        )
        for feat_name, weight in exp.as_list():
            col = _parse_feature_name(feat_name)
            if col in local.columns:
                local.loc[i, col] = abs(weight)

    return local



def explain_with_lime(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    *,
    num_features: int,
    num_samples: int,
    kernel_width: float | str | None,
    feature_selection: str,
    discretize_continuous: bool,
    random_state: int,
) -> pd.DataFrame:
    local = explain_with_lime_local(
        model,
        x_train,
        x_sample,
        num_features=num_features,
        num_samples=num_samples,
        kernel_width=kernel_width,
        feature_selection=feature_selection,
        discretize_continuous=discretize_continuous,
        random_state=random_state,
    )
    out = pd.DataFrame(
        {
            "feature": local.columns,
            "importance": np.asarray(local.mean(axis=0)),
        }
    )
    out["method"] = "lime"
    return out.sort_values("importance", ascending=False).reset_index(drop=True)
