from typing import Any, Dict

import numpy as np
from sklearn import metrics


def get_evaluation_metrics(y_true: np.array, y_pred: np.array) -> Dict[str, Any]:
    eval_metrics = {
        "ME": np.mean(y_true - y_pred),
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "MAPE": metrics.mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        "R2": metrics.r2_score(y_true, y_pred),
    }
    return eval_metrics


def smape_loss(y_test, y_pred):
    """Symmetric mean absolute percentage error
    Addapted from https://github.com/alan-turing-institute/sktime/blob/15c5ccba8999ddfc52fe37fe4d6a7ff39a19ece3/sktime/performance_metrics/forecasting/_functions.py#L79

    Args:
        y_test ([type]): pandas Series of shape = (fh,) where fh is the forecasting horizon
            Ground truth (correct) target values.
        y_pred ([type]): pandas Series of shape = (fh,)
        Estimated target values.

    Returns:
        float: sMAPE loss
    """
    nominator = np.abs(y_test - y_pred)
    denominator = np.abs(y_test) + np.abs(y_pred)
    return np.mean(2.0 * nominator / denominator)
