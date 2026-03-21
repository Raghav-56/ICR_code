from __future__ import annotations

import numpy as np
import pandas as pd



def bootstrap_topk_jaccard(df: pd.DataFrame, n_samples: int, seed: int = 42) -> dict[str, float]:
    if df.empty:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    vals = df["jaccard"].to_numpy()
    boots = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(vals), len(vals))
        boots.append(float(np.mean(vals[idx])))
    return {
        "mean": float(np.mean(vals)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }
