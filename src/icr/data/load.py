from __future__ import annotations

from pathlib import Path

import pandas as pd

from icr.config import PipelineConfig



def load_raw_dataframe(cfg: PipelineConfig) -> pd.DataFrame:
    path = Path(cfg.paths.raw_dir) / cfg.data.input_file
    if not path.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {path}. Put the dataset in data/raw and update config."
        )
    return pd.read_csv(path)
