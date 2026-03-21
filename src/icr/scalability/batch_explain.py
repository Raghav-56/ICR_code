from __future__ import annotations

import pandas as pd



def sample_for_scalability(x: pd.DataFrame, max_size: int, seed: int = 42) -> pd.DataFrame:
    if len(x) <= max_size:
        return x
    return x.sample(n=max_size, random_state=seed).reset_index(drop=True)
