"""
ModelTrainer class to train models and / or tune hyperparameters 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer

from litreading.utils import logger, save_file, open_file, BaselineModel
from litreading.dataset import Dataset
from litreading.config import (
    MODELS_PATH,
    PREPROCESSING_STEPS,
    SEED,
    DEFAULT_MODEL_TYPE,
    DEFAULT_PARAMS,
)


class ModelTrainer(Dataset):
    """
    Train models with default params on new data
    Train models with personalized params
    Tune hyperparameters
    Methods: set_new_model, __set_estimator, get_model_params, set_model_params, save_model
            train: model.fit
            prepare_train_test_set: Dataset.preprocessing, Dataset.compute_features, remove outliers, train test split and scale features
            evaluate_model: Dataset.statistics and error count by magnitude (1% ,5%, 10%)
            grid_search: compute sklearn grid search on selected model with selected set of params and k-folds
            feature_importance: plot the feature importance for current model. Only for RF and XGB.
    Static methods: plot_grid_search: plot grid search results with seaborn with one param as x and one as hue
                    plot_distribution: plot distributon of errors
                    plot_scatter: scatter plot of y and x
    """

    def __init__(
        self,
        df,
        model_type=DEFAULT_MODEL_TYPE,
        prompt_col="prompt",
        asr_col="asr_transcript",
        human_col="human_transcript",
        duration_col="scored_duration",
        human_wcpm_col="human_wcpm",
    ):
        self.set_new_model(model_type)
        Dataset.__init__(
            self,
            df=df,
            prompt_col="prompt",
            asr_col="asr_transcript",
            human_col="human_transcript",
            duration_col="scored_duration",
            human_wcpm_col="human_wcpm",
            mode="train",
        )

    @property
    def model_type(self):
        return self.__model_type

    @staticmethod
    def __set_estimator(model_type):
        if model_type == "RF":
            estimator = RandomForestRegressor(random_state=SEED)
        elif model_type == "XGB":
            estimator = XGBRegressor(random_state=SEED)
        elif model_type == "KNN":
            estimator = KNeighborsRegressor()
        elif model_type == "Baseline":
            estimator = BaselineModel()
        else:
            raise KeyError("Sorry, this model type has not been implemented yet")
        return estimator

    def set_new_model(self, model_type, params={}, inplace=True):
        """ Set new models with chosen params """
        estimator = self.__set_estimator(model_type)
        self.__model_type = model_type
        if not inplace:
            return estimator
        self.__model = estimator
        self.set_model_params(params)
        logger.info("New model set: %s", model_type)

    def get_model_params(self):
        return self.__model.get_params()

    def set_model_params(self, params={}):
        """ default sklearn params, default config.py params of personalized params """
        if params == "config_params":
            params = DEFAULT_PARAMS[self.__model_type]
            self.__model.set_params(**params)
        elif isinstance(params, dict):
            if params != {}:
                self.__model.set_params(**params)
        else:
            raise TypeError(
                "Expected: 'config_params' or dictionnary of params names: params value"
            )

    def save_model(self, scaler=False, model=False, replace=False):
        """ Save both scaler and trained / untrained model with params in joblib in config.MODELS_PATH"""
        if scaler:
            try:
                save_file(
                    self.scaler,
                    path=MODELS_PATH,
                    file_name="standard_scaler.joblib",
                    replace=replace,
                )
            except AttributeError:
                logger.error(
                    "scaler not defined: Please fit a scaler before saving \
it by calling ModelTrainer.prepare_train_test_set()"
                )
        if model:
            save_file(
                self.__model,
                path=MODELS_PATH,
                file_name=self.__model_type + ".joblib",
                replace=replace,
            )

    def train(self):
        """ If test and train set were prepared before, will train current model with current params """
        try:
            self.X_train
        except AttributeError:
            logger.error(
                "X_train, Y_train not defined: Please prepare train and test \
set before training by calling ModelTrainer.prepare_train_test_set()"
            )
            return
        # reset model to blank
        params = self.__model.get_params()
        self.__model = self.__set_estimator(self.__model_type)
        self.set_model_params(params)
        # train
        logger.info("Training %s", self.__model_type)
        self.__model = self.__model.fit(self.X_train, self.Y_train)

    def prepare_train_test_set(
        self, remove_outliers=False, outliers_tol=0.1, test_set_size=0.2, inplace=True
    ):
        """
        if not Inplace, return X_train, X_test, Y_train, Y_test
        Preprocess data using Dataset.preprocess_data()
        Compute features using Dataset.compute_features()
        Remove outliers if remove_outliers = True (default: False) with tol=outliers_tol (default: .1)
        Split train and test set using sklearn.model_selection.train_test_split()
        Scale (fit transform) using sklearn.preprocessing.StandardScaler()
        """
        # Preprocessing data
        self.preprocess_data(**PREPROCESSING_STEPS, inplace=True)
        # create dataframe of features with Dataset.compute_features()
        self.compute_features(inplace=True)
        # removing outliers based on diff between human and asr transcript length
        if remove_outliers:
            mask = self.determine_outliers_mask(tol=outliers_tol)
            self.features = self.features[mask]
            self.datapoints = mask.sum()
            logger.info(
                "Removed %i outliers, %i datapoints remaining for training/testing",
                len(mask) - self.datapoints,
                self.datapoints,
            )
        else:
            self.datapoints = len(self.features.index)
        # train test split
        X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(
            self.features.drop(columns=["human_wcpm"]),
            self.features["human_wcpm"],
            test_size=test_set_size,
            random_state=SEED,
        )
        self.test_idx = X_test_raw.index
        logger.debug("Fit scaler to training set and transform training and test set")

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)
        if not inplace:
            return X_train, X_test, Y_train, Y_test
        else:
            self.X_train, self.X_test = X_train, X_test
            self.Y_train, self.Y_test = Y_train, Y_test

    def evaluate_model(self, visualize=True):
        """
        Return 3 DataFrames: statistics for each row, summary of statistics per bin, summary fo error counts per bin
        bins are <75 wcpm, 75-150 wcpm and 150+ wcpm
        """
        Y_pred = self.__model.predict(self.X_test)
        stats = self.compute_stats(Y_pred, self.test_idx)
        # creating 3 groups of datapoints based on human_wcpm
        assign_wcpm_bin = (
            lambda x: "<75" if x < 75 else ("75-150" if x < 150 else "150+")
        )
        stats["wcpm_bin"] = stats["human_wcpm"].apply(assign_wcpm_bin)
        # computing mean and std of stats for each bin
        stats_summary = stats.groupby("wcpm_bin").agg(["mean", "std"])
        stats_summary.loc["total"] = stats.agg(["mean", "std"]).unstack()
        # stats_summary = pd.concat([stats_summary, total], axis=0)
        stats_summary.drop(
            columns=["human_wcpm", "wcpm_estimation", "wcpm_estimation_error_%"],
            inplace=True,
        )
        cols = [
            ("wcpm_estimation_abs_error_%", "mean"),
            ("wcpm_estimation_abs_error_%", "std"),
        ]
        stats_summary[cols] = stats_summary[cols] * 100
        stats_summary = stats_summary.applymap(lambda x: round(x, 2))
        # computing # of error > 1%, 5%, 10% per bin
        d = {
            "Total Count": -1,
            "Error > 1%": 0.01,
            "Error > 5%": 0.05,
            "Error > 10%": 0.1,
        }
        for k, v in d.items():
            stats[k] = stats["wcpm_estimation_abs_error_%"] > v
        agg_funcs = ["sum", lambda x: round(100 * sum(x) / len(x), 1)]
        errors_summary = stats[list(d.keys()) + ["wcpm_bin"]].groupby("wcpm_bin")
        errors_summary = errors_summary.agg(agg_funcs)
        errors_summary.columns.set_levels(["count", "% of bin"], level=1, inplace=True)
        temp = (
            stats[list(d.keys())].agg(agg_funcs).unstack()
        )  # for total count and % of test set
        temp.index.set_levels(["count", "% of bin"], level=1, inplace=True)
        errors_summary.loc["total"] = temp
        if visualize:
            self.plot_wcpm_distribution(
                stats=stats, x="wcpm_estimation_error_%", stat="count", binwidth=0.01
            )
        return stats, stats_summary, errors_summary

    def grid_search(
        self, model_type, cv_params, cv_folds=5, verbose=2, scoring_metric="r2"
    ):
        """
        Perform sklearn.model_selection.GridSearch() on model model_type with params cv_params
        For other information see sklearn documentation
        """
        params = (
            dict()
        )  # for params in cv_params with unique value, set directly to estimator
        params_grid = dict()  # for params to be actually cross-validated
        for key, value in cv_params.items():
            if len(value) == 1:
                params[key] = value[0]
            else:
                params_grid[key] = value
        estimator = self.set_new_model(model_type=model_type, inplace=False)
        print("\n" + " Estimator: ".center(120, "-"))
        print(estimator.__class__.__name__)
        print("\n" + " Metric for evaluation: ".center(120, "-"))
        print(scoring_metric)
        if len(params.keys()) > 0:
            print("\n" + " Fixed params: ".center(120, "-"))
            [print(key, value) for key, value in params.items()]
        print("\n" + " Params to be tested: ".center(120, "-"))
        [print(key, value) for key, value in params_grid.items()]
        params_list = list(itertools.product(*params_grid.values()))
        n_combi = len(params_list)
        print(
            "\n"
            + " # of possible combinations to be cross-validated: {:d}".format(n_combi)
        )
        answer = input("\n" + "Continue with these c-v parameters ? (y/n)  ")
        if answer == "n" or answer == "no":
            print("Please redefine inputs.")
            return

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=params_grid,
            scoring=scoring_metric,
            cv=cv_folds,
            n_jobs=-1,
            verbose=verbose,
        )
        try:
            grid_search.fit(self.X_train, self.Y_train)
        except AttributeError:
            logger.error(
                "X_train, Y_train not defined: Please prepare train and test \
set before training by calling ModelTrainer.prepare_train_test_set()"
            )
        self.__model = grid_search.best_estimator_
        return grid_search

    @staticmethod
    def plot_grid_search(cv_results, x, hue=None, y="mean_test_score", log_scale=True):
        """
        Will plot the mean_test_score (or other metric if you choose)
        For parameters x and hue using seaborn
        log_scale = True to plot with a log_scale on x axis
        """
        # Get Test Scores Mean and std for each grid search
        cv_results = pd.DataFrame(cv_results)
        if hue is not None:
            hue = "param_" + hue
        x = "param_" + x
        # Plot Grid search scores
        plt.style.use("seaborn-darkgrid")
        _, ax = plt.subplots(1, 1, figsize=(10, 4))

        # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
        sns.lineplot(data=cv_results, x=x, y=y, hue=hue, palette="Set2")

        ax.set_title("Grid Search Scores", fontsize=20, fontweight="bold")
        ax.set_xlabel(x, fontsize=16)
        ax.set_ylabel("CV Average Score", fontsize=16)
        ax.legend(loc="best", fontsize=15)
        if log_scale:
            ax.set_xscale("log")
        ax.grid("on")
        plt.show()

    def feature_importance(self, threshold=0.001):
        """
        Compute and plot feature importance for tree based methods from sklearn or similar
        input: model already fitted
            features: names of the features
            threshold: minimum feature importance for the feature to be plotted
        """
        importance = self.__model.feature_importances_
        idx = [x[0] for x in enumerate(importance) if x[1] > threshold]
        labels = self.features.columns[idx]
        importance = importance[idx]
        idx = np.argsort(importance)[::-1]

        plt.style.use("seaborn-darkgrid")
        _, ax = plt.subplots(1, 1, figsize=(8, max(8, 0.2 * len(idx))))
        sns.barplot(x=importance[idx], y=labels[idx], color=sns.color_palette()[0])
        for i, val in enumerate(importance[idx]):
            ax.text(val + 0.01, i, s="{:.3f}".format(val), ha="left", va="center")
        ax.set_title("Feature importance for current model", fontsize=16)
        ax.set_xlim(0, max(importance[idx]) + 0.03)
        plt.show()

    @staticmethod
    def plot_wcpm_distribution(stats, x, stat="count", binwidth=0.01):
        """ Plot distribution of stats[stat] from x in bins of bin_width """
        plt.style.use("seaborn-darkgrid")
        _, ax = plt.subplots(1, 1, figsize=(16, 6))
        sns.histplot(ax=ax, data=stats, x=x, stat=stat, binwidth=binwidth)
        ax.set_title("Distribution of errors", fontsize=20, fontweight="bold")
        ax.set_xlabel(x, fontsize=16)
        if "%" in x:
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel("count", fontsize=16)
        plt.show()

    @staticmethod
    def plot_wcpm_scatter(stats, x="human_wcpm", y="wcpm_estimation_error_%"):
        """
        Scatter plot of x and y in stats to be choosen by user
        Default x='human_wcpm' and y='wcpm_estimation_error_%'
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
