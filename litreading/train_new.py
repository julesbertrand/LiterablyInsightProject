from typing import Any, Dict, Union

import numpy as np
import pandas as pd

# from sklearn.preprocessing import FunctionTransformer
from sklearn import base, metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from litreading.config import SEED
from litreading.preprocessor import LCSPreprocessor


class Model:
    def __init__(
        self,
        estimator: Union[str, BaseEstimator] = "default",
        scaler: Union[str, TransformerMixin] = "default",
    ) -> None:
        self._check_estimator(estimator)
        self._check_scaler(scaler)
        self._build_model()

    @property
    def model(self) -> Pipeline:
        return self._model

    @property
    def preprocessor(self) -> LCSPreprocessor:
        return self._preprocessor

    def _check_estimator(self, estimator: Union[str, BaseEstimator]):
        if isinstance(estimator, str):
            raise NotImplementedError
        elif isinstance(estimator, BaseEstimator):
            if base.is_classifier(estimator):
                raise TypeError("estimator must be a sklearn-like regressor")
            self._estimator = estimator
        else:
            raise TypeError("estimator must be either an str or a sklearn.base.BaseEstimator.")

    def _check_scaler(self, scaler: Union[str, TransformerMixin]):
        if isinstance(scaler, str):
            raise NotImplementedError
        elif isinstance(scaler, TransformerMixin):
            if not hasattr(scaler, "fit_transform"):
                raise TypeError("scaler must be a sklearn-like classifier")
            self._scaler = scaler
        else:
            raise TypeError("scaler must be either an str or a sklearn.base.BaseEstimator.")

    def _build_model(self) -> None:
        self._preprocessor = LCSPreprocessor()
        self._model = Pipeline(
            [
                (f"scaler: {self._scaler.__class__.__name__}", self._scaler),
                (f"estimator: {self._estimator.__class__.__name__}", self._estimator),
            ],
            verbose=True,
        )

    def prepare_train_test_set(
        self,
        df: pd.DataFrame,
        test_set_size: float = 0.2,
    ) -> None:
        self._X_train_raw, self._X_test_raw, self.y_train, self.y_test = train_test_split(
            df.drop(columns=["human_wcpm"]),
            df["human_wcpm"],
            test_size=test_set_size,
            random_state=SEED,
        )
        self._test_idx = self._X_test_raw.index

    def fit(self) -> None:
        self.X_train = self.preprocessor.preprocess_data(self._X_train_raw)
        mask = pd.DataFrame(self.X_train).isna().any(axis=1) | pd.Series(self.y_train).isna().any()
        self.X_train, self.y_train = self.X_train[~mask], self.y_train[~mask]
        self._model.fit(self.X_train, self.y_train)
        return self

    def predict(self, X: np.array) -> None:
        X_processed = self.preprocessor.preprocess_data(X)
        y_pred = self._model.predict(X_processed)
        return y_pred

    def evaluate(self, X: np.array = None, y_true: np.array = None) -> pd.DataFrame:
        if X is None:
            X = self._X_test_raw
        if y_true is None:
            y_true = self.y_test

        y_pred = self.predict(X)
        metrics = pd.Series(get_evaluation_metrics(y_true, y_pred))
        return metrics

    def grid_search(self):
        raise NotImplementedError

    def plot_grid_search(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError


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


if __name__ == "__main__":
    df = pd.read_csv("./data/test_data.csv")
    print(df.head())
    m = Model(estimator=LinearRegression(), scaler=StandardScaler())
    m.prepare_train_test_set(df)
    m = m.fit()
    perfs = m.evaluate()
    print(perfs)
