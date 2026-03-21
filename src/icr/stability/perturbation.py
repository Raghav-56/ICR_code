from __future__ import annotations

import numpy as np
import pandas as pd



def perturb_numeric(x: pd.DataFrame, sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = x.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        std = float(out[col].std() or 1.0)
        noise = rng.normal(0.0, sigma * std, size=len(out))
        out[col] = out[col] + noise
    return out
