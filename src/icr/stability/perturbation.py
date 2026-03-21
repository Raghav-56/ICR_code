from __future__ import annotations

import numpy as np
import pandas as pd



def perturb_numeric(
    x: pd.DataFrame,
    sigma: float,
    seed: int,
    reference_std: pd.Series | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = x.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if reference_std is not None and col in reference_std.index:
            std = float(reference_std[col] or 1.0)
        else:
            std = float(out[col].std() or 1.0)
        noise = rng.normal(0.0, sigma * std, size=len(out))
        out[col] = out[col] + noise
    return out
