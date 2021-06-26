import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from litreading.train import Model


@pytest.mark.parametrize(
    "scaler, estimator, mode",
    [
        (StandardScaler(), LinearRegression(), "custom"),
        (StandardScaler(), LinearRegression(), "baseline"),
        (None, None, "baseline"),
    ],
)
def test_model_init(scaler, estimator, mode):
    Model(scaler=scaler, estimator=estimator, mode=mode)


@pytest.mark.parametrize(
    "scaler, estimator, mode",
    [
        (StandardScaler(), LinearRegression(), "custom"),
        (StandardScaler(), LinearRegression(), "baseline"),
        (None, None, "baseline"),
    ],
)
def test_model_fit(scaler, estimator, mode):
    test_dataset = pd.read_csv("tests/samples/test_data.csv")

    m = Model(scaler=scaler, estimator=estimator, mode=mode)
    m.prepare_train_test_set(test_dataset)
    m.fit()
