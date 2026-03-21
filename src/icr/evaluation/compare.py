from __future__ import annotations

import pandas as pd



def compare_metric_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(by="pr_auc", ascending=False)
