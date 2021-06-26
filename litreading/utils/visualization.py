from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns


def plot_wcpm_distribution(stats, x: str, stat: str = "count", binwidth: float = 0.01):
    """Plot distribution of stats[stat] from x in bins of bin_width

    Args:
        stats ([type]): [description]
        x (str): [description]
        stat (str, optional): [description]. Defaults to "count".
        binwidth (float, optional): [description]. Defaults to 0.01.
    """
    plt.style.use("seaborn-darkgrid")
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
    plt.style.use("seaborn-darkgrid")
    _, ax = plt.subplots(1, 1, figsize=(16, 6))
    sns.scatterplot(data=stats, x=x, y=y)
    ax.set_title("Graph of %s" % y, fontsize=20, fontweight="bold")
    ax.set_xlabel(x, fontsize=16)
    ax.set_ylabel(y, fontsize=16)
    if "%" in y:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.show()


def feature_importance(model, feature_names: List[str], threshold: float = 0.001):
    """Compute and plot feature importance for tree based methods from sklearn or similar

    Args:
        threshold (float, optional): minimum feature importance for the feature to be plotted. Defaults to 0.001.
    """
    # df = pd.Dataframe({feature_names})
    importances = dict(zip(feature_names, model.feature_importances_))
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    std = dict(zip(feature_names, std))
    idx = [x[0] for x in enumerate(importances) if x[1] > threshold]
    labels = model.features.columns[idx]
    importance = importances[idx]
    idx = np.argsort(importance)[::-1]

    plt.style.use("seaborn-darkgrid")
    _, ax = plt.subplots(1, 1, figsize=(8, max(8, 0.2 * len(idx))))
    sns.barplot(x=importance[idx], y=labels[idx], color=sns.color_palette()[0])
    for i, val in enumerate(importance[idx]):
        ax.text(val + 0.01, i, s="{:.3f}".format(val), ha="left", va="center")
    ax.set_title("Feature importance for current model", fontsize=16)
    ax.set_xlim(0, max(importance[idx]) + 0.03)
    plt.show()
