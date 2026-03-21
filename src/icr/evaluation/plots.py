from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibrationDisplay



def save_calibration_plot(y_true, y_prob, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if isinstance(y_prob, Mapping):
        for label, prob in y_prob.items():
            CalibrationDisplay.from_predictions(y_true, prob, n_bins=10, ax=ax, name=label)
        ax.legend()
    else:
        CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, ax=ax)
    ax.set_title("Calibration Curve")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_runtime_plot(runtime_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method, grp in runtime_df.groupby("method"):
        ax.plot(grp["size"], grp["median_seconds"], marker="o", label=method)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Median runtime (s)")
    ax.set_title("Scalability: runtime by explanation method")
    ax.legend()
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
