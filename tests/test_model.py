import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from litreading.model import Model


@pytest.mark.parametrize(
    "scaler, estimator, baseline_mode",
    [
        (StandardScaler(), LinearRegression(), False),
        (StandardScaler(), LinearRegression(), True),
        (None, None, True),
    ],
)
def test_model_init(scaler, estimator, baseline_mode):
    Model(scaler=scaler, estimator=estimator, baseline_mode=baseline_mode)


@pytest.mark.parametrize(
    "scaler, estimator, baseline_mode",
    [
        (StandardScaler(), LinearRegression(), False),
        (StandardScaler(), LinearRegression(), True),
        (None, None, True),
    ],
)
def test_model_fit(scaler, estimator, baseline_mode):
    test_dataset = pd.read_csv("tests/samples/test_data.csv")

    m = Model(
        scaler=scaler,
        estimator=estimator,
        baseline_mode=baseline_mode,
        verbose=True,
    )
    m.prepare_train_test_set(test_dataset)
    m.fit()
