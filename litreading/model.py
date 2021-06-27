from typing import Any, Dict, Union

import numpy.typing as npt
import pandas as pd
from loguru import logger

# from sklearn.preprocessing import FunctionTransformer
from sklearn import base
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from litreading.config import HUMAN_WCPM_COL, SEED
from litreading.preprocessor import LCSPreprocessor
from litreading.utils.evaluation import compute_evaluation_report


class Model:

    _preprocessor = LCSPreprocessor()

    def __init__(
        self,
        estimator: Union[str, BaseEstimator] = "default",
        scaler: Union[str, TransformerMixin] = "default",
        baseline_mode: bool = False,
    ) -> None:
        if not isinstance(baseline_mode, bool):
            raise TypeError("baseline_mode must be a bool")
        self._baseline_mode = baseline_mode
        self._build_model(scaler, estimator)

    @property
    def model(self) -> Union[Pipeline, TransformerMixin]:
        return self._model

    @property
    def preprocessor(self) -> LCSPreprocessor:
        return self._preprocessor

    @property
    def baseline_mode(self) -> bool:
        return self._baseline_mode

    def _build_model(
        self, scaler: Union[str, BaseEstimator], estimator: Union[str, TransformerMixin]
    ) -> None:
        if self._baseline_mode:
            msg = "Baseline mode -> Instanciating Baseline Model. Any scaler or estimator argument will be ignored."
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
                verbose=True,
            )

    def _check_estimator(self, estimator: Union[str, BaseEstimator]) -> None:
        if isinstance(estimator, str):
            raise NotImplementedError
        elif isinstance(estimator, BaseEstimator):
            if base.is_classifier(estimator):
                raise TypeError("estimator must be a sklearn-like regressor")
            self._estimator = estimator
        else:
            raise TypeError("estimator must be either an str or a sklearn.base.BaseEstimator.")

    def _check_scaler(self, scaler: Union[str, TransformerMixin]) -> None:
        if isinstance(scaler, str):
            raise NotImplementedError
        elif isinstance(scaler, TransformerMixin):
            if not hasattr(scaler, "fit_transform"):
                raise TypeError("scaler must be a sklearn-like classifier")
            self._scaler = scaler
        else:
            raise TypeError("scaler must be either an str or a sklearn.base.BaseEstimator.")

    def prepare_train_test_set(
        self,
        dataset: pd.DataFrame,
        test_set_size: float = 0.2,
    ) -> None:
        self._X_train_raw, self._X_test_raw, self.y_train, self.y_test = train_test_split(
            dataset.drop(columns=[HUMAN_WCPM_COL]),
            dataset[HUMAN_WCPM_COL],
            test_size=test_set_size,
            random_state=SEED,
        )
        self._test_idx = self._X_test_raw.index

    def fit(self):
        self.X_train = self.preprocessor.preprocess_data(self._X_train_raw)
        if not self._baseline_mode:
            mask = (
                pd.DataFrame(self.X_train).isna().any(axis=1)
                | pd.Series(self.y_train).isna().any()
            )  # HACK
            self.X_train, self.y_train = self.X_train[~mask], self.y_train[~mask]
            self._model.fit(self.X_train, self.y_train)
        return self

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        X_processed = self.preprocessor.preprocess_data(X)
        if self._baseline_mode:
            y_pred = X_processed["correct_words_pm"]
        else:
            y_pred = self._model.predict(X_processed)
        return y_pred

    def evaluate(self, X: npt.ArrayLike = None, y_true: npt.ArrayLike = None) -> pd.DataFrame:
        if X is None:
            X = self._X_test_raw
        if y_true is None:
            y_true = self.y_test

        y_pred = self.predict(X)
        metrics = compute_evaluation_report(y_true, y_pred)
        return metrics

    def grid_search(
        self,
        params_grid: Dict[str, Any],
        cv: int = 5,
        verbose: int = 2,
        scoring_metric: str = "r2",
    ):
        grid_search = GridSearchCV(
            estimator=self._model,
            param_grid=params_grid,
            scoring=scoring_metric,
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
        )
        return grid_search

    def plot_grid_search(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError
