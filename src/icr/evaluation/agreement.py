from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _rank_positions(row: pd.Series) -> dict[str, int]:
    ordered = row.sort_values(ascending=False).index.tolist()
    return {f: i for i, f in enumerate(ordered)}


def per_instance_jaccard_at_k(a_local: pd.DataFrame, b_local: pd.DataFrame, k: int) -> list[float]:
    vals: list[float] = []
    for i in range(min(len(a_local), len(b_local))):
        sa = set(a_local.iloc[i].sort_values(ascending=False).head(k).index)
        sb = set(b_local.iloc[i].sort_values(ascending=False).head(k).index)
        vals.append(1.0 if (not sa and not sb) else len(sa & sb) / len(sa | sb))
    return vals


def per_instance_spearman(a_local: pd.DataFrame, b_local: pd.DataFrame) -> list[float]:
    vals: list[float] = []
    for i in range(min(len(a_local), len(b_local))):
        a_row = a_local.iloc[i]
        b_row = b_local.iloc[i]
        corr = spearmanr(a_row.to_numpy(), b_row.to_numpy()).statistic
        vals.append(float(0.0 if np.isnan(corr) else corr))
    return vals


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        boots.append(float(np.mean(arr[idx])))

    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }


def summarize_agreement_with_ci(
    shap_local: pd.DataFrame,
    lime_local: pd.DataFrame,
    *,
    top_k: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    j_vals = per_instance_jaccard_at_k(shap_local, lime_local, top_k)
    s_vals = per_instance_spearman(shap_local, lime_local)
    j_ci = bootstrap_ci(j_vals, n_boot=n_bootstrap, seed=seed)
    s_ci = bootstrap_ci(s_vals, n_boot=n_bootstrap, seed=seed)
    return {
        "jaccard_at_k_mean": j_ci["mean"],
        "jaccard_at_k_ci_low": j_ci["ci_low"],
        "jaccard_at_k_ci_high": j_ci["ci_high"],
        "spearman_mean": s_ci["mean"],
        "spearman_ci_low": s_ci["ci_low"],
        "spearman_ci_high": s_ci["ci_high"],
    }
