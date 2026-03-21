import pandas as pd

from icr.config import PipelineConfig
from icr.data.split import stratified_split



def test_stratified_split_sizes():
    df = pd.DataFrame(
        {
            "x": list(range(100)),
            "SeriousDlqin2yrs": [0] * 90 + [1] * 10,
        }
    )
    cfg = PipelineConfig()
    splits = stratified_split(df, cfg)
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(df)
