from __future__ import annotations

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer



def explain_with_lime(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    *,
    num_features: int,
    num_samples: int,
    kernel_width: float,
    feature_selection: str,
    discretize_continuous: bool,
    random_state: int,
) -> pd.DataFrame:
    explainer = LimeTabularExplainer(
        training_data=x_train.values,
        feature_names=list(x_train.columns),
        class_names=["no_default", "default"],
        mode="classification",
        discretize_continuous=discretize_continuous,
        kernel_width=kernel_width,
        random_state=random_state,
        feature_selection=feature_selection,
    )

    agg = {col: [] for col in x_train.columns}

    for i in range(len(x_sample)):
        instance = x_sample.iloc[i].values
        exp = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
        )
        for feat_name, weight in exp.as_list():
            matched = [c for c in x_train.columns if c in feat_name]
            if matched:
                agg[matched[0]].append(abs(weight))

    rows = []
    for col, vals in agg.items():
        rows.append({"feature": col, "importance": float(np.mean(vals) if vals else 0.0)})

    out = pd.DataFrame(rows)
    out["method"] = "lime"
    return out.sort_values("importance", ascending=False).reset_index(drop=True)
