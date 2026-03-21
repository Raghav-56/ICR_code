from __future__ import annotations

import numpy as np
import pandas as pd
import shap



def explain_with_shap(model, x_sample: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.Explainer(model, x_sample)
    shap_values = explainer(x_sample)

    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    mean_abs = np.mean(np.abs(vals), axis=0)
    out = pd.DataFrame(
        {
            "feature": x_sample.columns,
            "importance": mean_abs,
            "method": "shap",
        }
    ).sort_values("importance", ascending=False)
    return out.reset_index(drop=True)
