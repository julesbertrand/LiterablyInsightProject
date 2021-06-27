from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import BaseEstimator

COLOR_PALETTE = "Set2"
plt.style.use("seaborn-darkgrid")


def plot_wcpm_distribution(stats, x: str, stat: str = "count", binwidth: float = 0.01):
    """Plot distribution of stats[stat] from x in bins of bin_width

    Args:
        stats ([type]): [description]
        x (str): [description]
        stat (str, optional): [description]. Defaults to "count".
        binwidth (float, optional): [description]. Defaults to 0.01.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    sns.histplot(ax=ax, data=stats, x=x, stat=stat, binwidth=binwidth)
    ax.set_title("Distribution of errors", fontsize=20, fontweight="bold")
    ax.set_xlabel(x, fontsize=16)
    if "%" in x:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel(stat, fontsize=16)
    plt.show()


def plot_wcpm_scatter(stats, x: str, y: str):
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


def feature_importance(
    estimator: BaseEstimator, feature_names: List[str], threshold: float = 0.001
) -> plt.Figure:
    """Compute and plot feature importances for tree based estimators

    Args:
        estimator (BaseEstimator): [description]
        feature_names (List[str]): [description]
        threshold (float, optional): [description]. Defaults to 0.001.

    Returns:
        plt.Figure: [description]
    """
    std = np.std([tree.feature_importances_ for tree in estimator.estimators_], axis=0)
    df = pd.DataFrame(
        {"feature_name": feature_names, "importance": estimator.feature_importances_, "std": std}
    )
    df = df.query(f"importance > {threshold}")
    df = df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, max(8, 0.2 * df.shape[0])))

    sns.barplot(data=df, x="importance", y="feature_name", color=sns.color_palette()[0])
    for i, val in enumerate(df.importance):
        ax.text(val + 0.01, i, s=f"{val:.3f}", ha="left", va="center")

    ax.set_title("Feature importance for current model", fontsize=16)
    ax.set_xlim(0, max(df.importance) + 0.03)

    return fig


def plot_grid_search(
    cv_results: dict,
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
    cv_results = pd.DataFrame(cv_results)
    for col in [x, y, hue]:
        if col not in cv_results.columns:
            raise ValueError(f"cv_results does not have a '{col}' column.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    sns.lineplot(data=cv_results, x=x, y=y, hue=hue, palette=COLOR_PALETTE)

    ax.set_title("Grid Search Scores", fontsize=20, fontweight="bold")
    ax.set_xlabel(x, fontsize=16)
    ax.set_ylabel("CV Average Score", fontsize=16)
    ax.legend(loc="best", fontsize=15)
    if x_log_scale:
        ax.set_xscale("log")
    ax.grid("on")

    return fig
