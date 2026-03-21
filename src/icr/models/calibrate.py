from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV

def calibrate_sigmoid(model, x_val, y_val):
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(x_val, y_val)
    return calibrated
