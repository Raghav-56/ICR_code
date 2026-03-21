from __future__ import annotations

import pandas as pd

from icr.explanation.compare_explanations import rank_correlation



def rank_corr_rows(base_df: pd.DataFrame, others: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for idx, other in enumerate(others):
        corr = rank_correlation(base_df, other)
        rows.append({"repeat": idx, **corr})
    return pd.DataFrame(rows)
