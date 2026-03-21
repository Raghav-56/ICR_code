from __future__ import annotations

import pandas as pd
from scipy.stats import kendalltau, spearmanr



def topk_jaccard(a: pd.DataFrame, b: pd.DataFrame, k: int) -> float:
    sa = set(a.head(k)["feature"])
    sb = set(b.head(k)["feature"])
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def rank_correlation(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    merged = a[["feature", "importance"]].merge(
        b[["feature", "importance"]], on="feature", suffixes=("_a", "_b")
    )
    if len(merged) < 2:
        return {"spearman": 0.0, "kendall": 0.0}
    spear = spearmanr(merged["importance_a"], merged["importance_b"]).statistic
    kend = kendalltau(merged["importance_a"], merged["importance_b"]).statistic
    return {"spearman": float(spear), "kendall": float(kend)}
