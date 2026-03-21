from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd


@dataclass
class BenchResult:
    method: str
    size: int
    seconds: float



def bench_callable(method: str, fn, x: pd.DataFrame, sizes: list[int]) -> list[BenchResult]:
    rows: list[BenchResult] = []
    for n in sizes:
        subset = x.head(min(n, len(x)))
        start = time.perf_counter()
        fn(subset)
        seconds = time.perf_counter() - start
        rows.append(BenchResult(method=method, size=len(subset), seconds=seconds))
    return rows
