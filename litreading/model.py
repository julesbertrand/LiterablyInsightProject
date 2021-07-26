import numpy.typing as npt
from typing import Any, Dict, List, Literal, Union

import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from loguru import logger

from sklearn import base
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from litreading.base import BaseModel, Dataset, load_model_from_file
from litreading.config import HUMAN_WCPM_COL, PREPROCESSING_STEPS, SEED
from litreading.preprocessor import LCSPreprocessor
from litreading.utils.evaluation import compute_evaluation_report
from litreading.utils.files import save_to_file
from litreading.utils.visualization import (
    plot_actual_vs_pred_scatter,
    plot_feature_importance,
    plot_grid_search,
)


class Model(BaseModel):

    _preprocessor = LCSPreprocessor(**PREPROCESSING_STEPS)

    def __init__(
        self,
        estimator: Union[str, BaseEstimator] = "default",
        scaler: Union[str, TransformerMixin] = "default",
        baseline_mode: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(baseline_mode=baseline_mode)
        self.verbose = verbose
        self._scaler = None
        self._estimator = None
        self._model = None
        self._build_model(scaler, estimator)

        self._dataset = None

    @property
    def dataset(self) -> Dataset:
        if not hasattr(self, "_dataset") or self._dataset is None:
            raise AttributeError(
                "training and test set not defined. Please start by using self._prepare_train_test_set"
            )
        return self._dataset

    def _build_model(
        self,
        scaler: Union[str, BaseEstimator],
        estimator: Union[str, TransformerMixin],
    ) -> None:
        if self.baseline_mode:
            msg = "Baseline mode -> Instanciating Baseline Model."
            msg += " Any scaler or estimator argument will be ignored."
            msg += "\nThe prediction is the word correct count based on differ list."
            logger.warning(msg)
            self._model = None
        else:
            self._check_estimator(estimator)
            self._check_scaler(scaler)
            self._model = Pipeline(
                [
                    ("scaler", self._scaler),
                    ("estimator", self._estimator),
                ],
                verbose=self.verbose,
            )

    def _check_estimator(self, estimator: Union[str, BaseEstimator]) -> None:
        if isinstance(estimator, str):
            raise NotImplementedError
        if isinstance(estimator, BaseEstimator):
            if base.is_classifier(estimator):
                raise TypeError("estimator must be a sklearn-like regressor")
            self._estimator = estimator
        else:
            raise TypeError("estimator must be either an str or a sklearn.base.BaseEstimator.")

    def _check_scaler(self, scaler: Union[str, TransformerMixin]) -> None:
        if isinstance(scaler, str):
            raise NotImplementedError
        if isinstance(scaler, TransformerMixin):
            if not hasattr(scaler, "fit_transform"):
                raise TypeError("scaler must be a sklearn-like classifier")
            self._scaler = scaler
        else:
            raise TypeError("scaler must be either an str or a sklearn.base.BaseEstimator.")

    def prepare_train_test_set(self, dataset: pd.DataFrame, test_set_size: float = 0.2) -> None:
        self._dataset = Dataset(
            *train_test_split(
                dataset.drop(columns=[HUMAN_WCPM_COL]),
                dataset[HUMAN_WCPM_COL],
                test_size=test_set_size,
                random_state=SEED,
            )
        )

    def fit(self):
        self._dataset.X_train = self.preprocessor.preprocess_data(
            self.dataset.X_train_raw, verbose=self.verbose
        )
        mask = (
            pd.DataFrame(self.dataset.X_train).isna().any(axis=1)
            | pd.Series(self.dataset.y_train).isna().any()
        )  # HACK
        self._dataset.X_train = self.dataset.X_train[~mask]
        self._dataset.y_train = self.dataset.y_train[~mask]

        if not self.baseline_mode:
            self.model.fit(self.dataset.X_train, self.dataset.y_train)
        return self

    def predict(self, X: pd.DataFrame) -> npt.ArrayLike:
        y_pred = self._predict(X)
        return y_pred

    def evaluate(self, X: npt.ArrayLike = None, y_true: npt.ArrayLike = None) -> pd.DataFrame:
        if X is None:
            X = self.dataset.X_test_raw
        if y_true is None:
            y_true = self.dataset.y_test

        y_pred = self.predict(X)
        metrics = compute_evaluation_report(y_true, y_pred)
        return metrics

    @staticmethod
    def __format_param_grid(
        mode=Literal["scaler", "estimator"], param_grid: Dict[str, List[Any]] = None
    ) -> Dict[str, List[Any]]:
        if param_grid is not None:
            param_grid = {f"{mode}__{k}": v for k, v in param_grid.items()}
        else:
            param_grid = {}
        return param_grid

    def grid_search(
        self,
        param_grid_scaler: Dict[str, List[Any]] = None,
        param_grid_estimator: Dict[str, List[Any]] = None,
        cv: int = 5,
        verbose: int = 5,
        scoring_metric: str = "r2",
        set_best_model: bool = True,
    ):
        param_grid = {}
        param_grid.update(self.__format_param_grid(mode="scaler", param_grid=param_grid_scaler))
        param_grid.update(
            self.__format_param_grid(mode="estimator", param_grid=param_grid_estimator)
        )
        if len(param_grid) == 0:
            raise ValueError("Please give at least one param to test")

        msg = f"\n{' Model: ' :-^120}\n"
        msg += f"{self.model}\n"
        msg += f"\n{' Params to be tested: ' :-^120}\n"
        msg += "\n".join([f"{key}: {value}" for key, value in param_grid.items()])
        n_combi = len(list(itertools.product(*param_grid.values())))
        msg += f"\n\n# of possible combinations for cross-validation: {n_combi}"
        msg += f"\nMetric for evaluation: {scoring_metric}"
        logger.info(msg)

        while True:
            answer = input("\nContinue with these Cross-validation parameters ? (y/n)")
            if answer not in ["y", "n"]:
                print("Possible answers: 'y' or 'n'")
                continue
            if answer == "y":
                break
            logger.error("Please redefine inputs.")
            return None

        verbose_model = self.model.verbose
        self._model.verbose = False

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring=scoring_metric,
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
        )

        self._dataset.X_train = self.preprocessor.preprocess_data(
            self.dataset.X_train_raw, verbose=verbose > 0
        )
        grid_search.fit(self.dataset.X_train, self.dataset.y_train)

        if set_best_model is True:
            self._model = grid_search.best_estimator_

        self._model.verbose = verbose_model

        return grid_search

    @staticmethod
    def plot_grid_search(
        cv_results: dict,
        x: str,
        y: str = "mean_test_score",
        hue: str = None,
        x_log_scale: bool = False,
    ) -> plt.Figure:
        fig = plot_grid_search(cv_results, x=x, y=y, hue=hue, x_log_scale=x_log_scale)
        return fig

    def plot_feature_importance(self, top_n: int = 10, print_table: bool = True) -> plt.Figure:
        fig, _ = plot_feature_importance(
            self.model["estimator"],
            self.dataset.X_train,
            self.dataset.y_train,
            top_n=top_n,
            print_table=print_table,
        )
        return fig

    def plot_scatter(self, X: npt.ArrayLike = None, y_true: npt.ArrayLike = None) -> go.Figure:
        if y_true is None:
            y_true = self.dataset.y_test
        if X is None:
            X = self.dataset.X_test_raw

        fig = plot_actual_vs_pred_scatter(y_true, self.predict(X))

        return fig

    def plot_wcpm_distribution(self) -> go.Figure:
        fig = ff.create_distplot(
            [self.dataset.y_train, self.dataset.y_test], ["train_set", "test_set"], bin_size=5
        )
        return fig

    def plot_feature_distribution(self, preprocess: bool = True) -> plt.Figure:
        if self.dataset.X_train is None and preprocess is not True:
            raise AttributeError(
                "Please start by preprocessing your raw data: \
either train a model or pass 'preprocess'=True"
            )

        if preprocess is True:
            _X_train = self._preprocessor.preprocess_data(self.dataset.X_train_raw, verbose=False)
            _X_test = self._preprocessor.preprocess_data(self.dataset.X_test_raw, verbose=False)
        else:
            _X_train = self.dataset.X_train.copy()
            _X_test = self.dataset.X_test.copy()

        _X_train["set"] = "train_set"
        _X_test["set"] = "test_set"
        _X = pd.concat([_X_train, _X_test])

        fig = sns.PairGrid(_X, hue="set", height=1)
        fig.map_upper(sns.scatterplot)
        fig.map_lower(sns.kdeplot, fill=True)
        fig.map_diag(sns.histplot, kde=True, common_norm=False)

        return fig

    @classmethod
    def load_from_file(cls, model_filepath: Union[str, os.PathLike]) -> Pipeline:
        model_ = load_model_from_file(model_filepath, estimator_ok=False)

        scaler = model_.steps[0][1]
        estimator = model_.steps[1][1]
        model = cls(estimator=estimator, scaler=scaler, baseline_mode=False)

        logger.info(f"Model loaded from {model_filepath}: \n{model_}")
        return model

    def save_model(
        self, filepath: Union[str, os.PathLike], version: bool = True, overwrite: bool = False
    ) -> None:
        if self.baseline_mode is True:
            raise ValueError("Cannot save baseline model as pickle file.")

        save_to_file(self.model, filepath, version=version, overwrite=overwrite, makedirs=True)
