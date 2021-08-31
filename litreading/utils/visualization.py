import numpy.typing as npt
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from sklearn.base import BaseEstimator

from litreading.utils.evaluation import get_evaluation_metrics

COLOR_PALETTE = "Set2"
plt.style.use("seaborn-darkgrid")


def plot_wcpm_scatter(stats, x: str, y: str) -> plt.Figure:
    """Scatter plot of x and y in stats to be choosen by user

    Args:
        stats ([type]): [description]
        x (str, optional): [description]. Defaults to HUMAN_WCPM_COL.
        y (str, optional): [description]. Defaults to "wcpm_estimation_error_%".
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    sns.scatterplot(data=stats, x=x, y=y)

    ax.set_title("Graph of %s" % y, fontsize=20, fontweight="bold")
    ax.set_xlabel(x, fontsize=16)
    ax.set_ylabel(y, fontsize=16)
    if "%" in y:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    return fig


def plot_feature_importance(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: Optional[pd.DataFrame] = None,
    top_n: int = 10,
    figsize: Tuple[int, int] = (8, 8),
    plot_error_bars: bool = True,
    print_table: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """plot feature importances of a tree-based sklearn estimator

    Args:
        estimator (BaseEstimator): sklearn-based estiamtor
        X_train (pd.DataFrame): training set features
        y_train (Optional[pd.DataFrame], optional): training set target values. Defaults to None.
        top_n (int, optional): top n feature importances to plot. Defaults to 10.
        figsize (Tuple[int, int], optional): Defaults to (8, 8).
        plot_error_bars (bool, optional): whether to plot error bars (std). Default to True.
        print_table (bool, optional): whether to print the table after the plot. Defaults to False.

    Raises:
        AttributeError: When feature_importances_ does not exists for the estimator

    Returns:
        plt.Figure: feature importances plot
        pd.DataFrame: df with feature name, importance, std based on trees
    """
    if not hasattr(estimator, "feature_importances_"):
        estimator.fit(X_train.values, y_train.values.ravel())
        if not hasattr(estimator, "feature_importances_"):
            raise AttributeError(
                f"{estimator.__class__.__name__} does not have feature_importances_ attribute"
            )

    feat_imp = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": estimator.feature_importances_,
            "std": np.std([tree.feature_importances_ for tree in estimator.estimators_], axis=0),
        }
    )
    feat_imp = feat_imp.sort_values(by="importance", ascending=False).iloc[:top_n]
    feat_imp = feat_imp.set_index("feature", drop=True)
    feat_imp = feat_imp.sort_values(by="importance", ascending=True)

    plot_kwargs = dict(
        title=f"Features Importances for {estimator.__class__.__name__}", figsize=figsize
    )
    if plot_error_bars is True:
        plot_kwargs["xerr"] = "std"
        fig = feat_imp.plot.barh(**plot_kwargs)
    else:
        fig = feat_imp.drop(columns=["std"]).plot.barh(**plot_kwargs)
    plt.xlabel("Feature Importance")

    if print_table is True:
        from IPython.display import display

        msg = f" Top {top_n} features in descending order of importance "
        print(f"\n{msg:-^100}\n")
        display(feat_imp.sort_values(by="importance", ascending=False))

    return fig, feat_imp


def plot_grid_search(
    cv_results: Dict[str, Any],
    x: str,
    hue: str = None,
    y: str = "mean_test_score",
    x_log_scale: bool = True,
) -> plt.Figure:
    """Plot grid search results for two parameters.
    x = param 1
    hue = param 2
    y = score

    Args:
        cv_results (dict): sklearn GridSearchCV.cv_results_ like
        x (str): [description]
        hue (str, optional): Defaults to None.
        y (str, optional): Defaults to "mean_test_score".
        x_log_scale (bool, optional): Plot x-axis on a log scale. Defaults to True.

    Returns:
        plt.Figure: [description]
    """
    cv_results_df = pd.DataFrame(cv_results)
    for col in [x, y, hue]:
        if col is not None:
            if col not in cv_results_df.columns:
                raise ValueError(f"cv_results does not have a '{col}' column.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    sns.lineplot(data=cv_results_df, x=x, y=y, hue=hue, palette=COLOR_PALETTE)

    ax.set_title("Grid Search Scores", fontsize=20, fontweight="bold")
    ax.set_xlabel(x, fontsize=16)
    ax.set_ylabel("CV Average Score", fontsize=16)
    ax.legend(loc="best", fontsize=15)
    if x_log_scale:
        ax.set_xscale("log")
    ax.grid("on")

    return fig


def plot_actual_vs_pred_scatter(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, title: str = "Actual v. Prediction"
) -> go.Figure:
    """Scatter plot of predictions on y-axis and ground truth on x-axis
    with optimal line x=y.

    Args:
        y_true (npt.ArrayLike): [description]
        y_pred (npt.ArrayLike): [description]
        title (str): [description]. Default to "Actual v. Prediction".

    Returns:
        go.Figure: [description]
    """
    fig = go.Figure(
        data=go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="values",
            marker=dict(color="#00828c", opacity=0.2),
        )
    )
    fig.add_trace(go.Scatter(x=y_true, y=y_true, name="optimal"))

    title += "\t\t\t" + " | ".join(
        f"{k}: {v}" for k, v in get_evaluation_metrics(y_true, y_pred, decimals=3).items()
    )
    fig.update_layout(title=title, xaxis_title="Actual", yaxis_title="Prediction")

    return fig
