from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed() -> float:
    start = time.perf_counter()
    result = {"seconds": 0.0}
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
