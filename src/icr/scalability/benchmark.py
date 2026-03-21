from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BenchResult:
    method: str
    size: int
    repeat: int
    seconds: float



def bench_callable(
    method: str,
    fn,
    x: pd.DataFrame,
    sizes: list[int],
    repeats: int,
) -> list[BenchResult]:
    rows: list[BenchResult] = []
    for n in sizes:
        subset = x.head(min(n, len(x)))
        for repeat in range(repeats):
            start = time.perf_counter()
            fn(subset)
            seconds = time.perf_counter() - start
            rows.append(
                BenchResult(
                    method=method,
                    size=len(subset),
                    repeat=repeat + 1,
                    seconds=seconds,
                )
            )
    return rows


def summarize_bench_medians(rows: list[BenchResult]) -> pd.DataFrame:
    raw = pd.DataFrame([r.__dict__ for r in rows])
    if raw.empty:
        return raw
    return (
        raw.groupby(["method", "size"], as_index=False)
        .agg(median_seconds=("seconds", lambda s: float(np.median(s))))
        .sort_values(["method", "size"])
    )
