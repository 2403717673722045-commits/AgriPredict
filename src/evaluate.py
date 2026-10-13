"""Metric helpers."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "MSE": mse, "RMSE": float(np.sqrt(mse)), "R2": float(r2_score(y_true, y_pred))}
