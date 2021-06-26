import numpy as np
from sklearn.base import BaseEstimator

from litreading.config import MODELS_PATH
from litreading.utils.files import open_file


class BaselineModel(BaseEstimator):
    """
    Baseline Model for litreading is only longest common subsequence, no moedl fit after
    Therefore this class is here only to provide fit, predict and get or set_params methods to the baseline model
    """

    def __init__(self):
        self.name = "BaselineModel"
        self.scaler = open_file(MODELS_PATH / "standard_scaler.joblib")

    def fit(self, X_train: np.array, Y_train: np.array):
        return self

    def set_params(self, **params):
        pass

    def get_params(self, **params):
        return {}

    def predict(self, X_test: np.array) -> np.array:
        # prediction is the word correct count based on differ list
        unscaled = self.scaler.inverse_transform(X_test)
        return unscaled[:, 0]
