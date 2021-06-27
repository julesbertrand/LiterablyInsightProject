from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger
from sklearn.pipeline import Pipeline

from litreading.config import BASELINE_MODEL_PREDICTION_COL, MODELS_PATH
from litreading.preprocessor import LCSPreprocessor
from litreading.utils.files import open_file


class Grader:

    _preprocessor = LCSPreprocessor()

    def __init__(
        self, model_filepath: Union[str, Path] = None, baseline_mode: bool = False
    ) -> None:
        if not isinstance(baseline_mode, bool):
            raise TypeError("baseline_mode must be a boolean")
        self._baseline_mode = baseline_mode
        if not baseline_mode:
            self._model = self.__load_model_from_file(model_filepath)

    @property
    def model(self) -> Pipeline:
        return self._model

    @property
    def preprocessor(self) -> LCSPreprocessor:
        return self._preprocessor

    @property
    def baseline_mode(self) -> bool:
        return self._baseline_mode

    def grade(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = self.preprocessor.preprocess_data(X)
        if self.baseline_mode:
            grades = X_processed[BASELINE_MODEL_PREDICTION_COL].values
        else:
            grades = self.model.predict(X_processed)
        return grades

    @staticmethod
    def __load_model_from_file(model_filepath: Union[str, Path]) -> Pipeline:
        model_filepath = MODELS_PATH / model_filepath
        logger.info(f"Loading model from {model_filepath}")
        model = open_file(model_filepath)
        return model

    def train_model(self, dataset: pd.DataFrame) -> None:
        raise NotImplementedError
