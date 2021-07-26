import numpy.typing as npt
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn import metrics


def compute_evaluation_report(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, total: bool = True
) -> pd.DataFrame:
    """Compute evaluation report with metrics per bin of wcpm

    Args:
        y_true (npt.ArrayLike): [description]
        y_pred (npt.ArrayLike): [description]

    Returns:
        pd.DataFrame: [description]
    """
    results = pd.DataFrame({"y": y_true, "yhat": y_pred})
    results["bin"] = results["y"].apply(
        lambda x: "<75" if x < 75 else ("75-150" if x < 150 else "150+")
    )
    groups = results.groupby("bin")
    metrics_df = groups.apply(lambda x: pd.Series(get_evaluation_metrics(x["y"], x["yhat"])))
    metrics_df["n_samples"] = groups.size()
    metrics_df = metrics_df.reset_index()

    if total:
        total_df = pd.DataFrame(get_evaluation_metrics(y_true, y_pred), index=["Total"])
        total_df["n_samples"] = results.shape[0]
        total_df["bin"] = "Total"
    metrics_df = pd.concat([metrics_df, total_df]).set_index("bin")
    return metrics_df


def get_evaluation_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, decimals: Optional[int] = None
) -> Dict[str, Any]:
    """Compute metrics for a given y_true and y_pred

    Args:
        y_true (npt.ArrayLike): [description]
        y_pred (npt.ArrayLike): [description]

    Returns:
        Dict[str, Any]: Dict of metric_name: metric_value
    """
    eval_metrics = {
        "ME": np.mean(y_true - y_pred),
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "MAPE": metrics.mean_absolute_percentage_error(y_true, y_pred),
        "SMAPE": smape_loss(y_true, y_pred),
        "RMSE": np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        "R2": metrics.r2_score(y_true, y_pred),
    }

    if decimals is not None:
        eval_metrics = {k: round(v, decimals) for k, v in eval_metrics.items()}
    return eval_metrics


def smape_loss(y_test: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Symmetric mean absolute percentage error
    Addapted from https://github.com/alan-turing-institute/sktime/blob/15c5ccba8999ddfc52fe37fe4d6a7ff39a19ece3/sktime/performance_metrics/forecasting/_functions.py#L79

    Args:
        y_test (npt.ArrayLike): pandas Series of shape = (fh,) where fh is the forecasting horizon
            Ground truth (correct) target values.
        y_pred (npt.ArrayLike): pandas Series of shape = (fh,)
        Estimated target values.

    Returns:
        float: sMAPE loss
    """
    nominator = np.abs(y_test - y_pred)
    denominator = np.abs(y_test) + np.abs(y_pred)
    return np.mean(2.0 * nominator / denominator)
