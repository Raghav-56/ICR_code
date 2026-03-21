from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay



def save_calibration_plot(y_true, y_prob, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, ax=ax)
    ax.set_title("Calibration Curve")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
