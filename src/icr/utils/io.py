from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd



def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)



def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)



def save_model(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    joblib.dump(obj, path)



def load_model(path: Path) -> Any:
    return joblib.load(path)
