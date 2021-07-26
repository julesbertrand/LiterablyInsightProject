"""This module implements a few classes and functions to be used for both the Grader and
the Model class, including the dataclass Dataset, a BaseModel with functions shared by
bosth Grader and Model, and a function to load a model from a file.
"""
import numpy.typing as npt
from typing import Union

import contextlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from litreading.config import (
    BASELINE_MODEL_PREDICTION_COL,
    PREPROCESSING_STEPS,
    SKLEARN_LOGLEVEL,
)
from litreading.preprocessor import LCSPreprocessor
from litreading.utils.files import open_file
from litreading.utils.logging import StreamToLogger


@dataclass
class Dataset:
    """Dataclass to always have access to raw, preprocessed and split data."""

    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train: pd.DataFrame = field(init=False, default=None)
    X_test: pd.DataFrame = field(init=False, default=None)
    train_test_ratio: float = field(init=False)

    def __post_init__(self):
        self.train_test_ratio = self.y_train.shape[0] / (
            self.y_train.shape[0] + self.y_test.shape[0]
        )


class BaseModel:
    _preprocessor = LCSPreprocessor(**PREPROCESSING_STEPS)

    def __init__(self, baseline_mode: bool = False) -> None:
        if not isinstance(baseline_mode, bool):
            raise TypeError("baseline_mode must be a boolean")
        self._baseline_mode = baseline_mode
        self._model = None  # Must be redefined in child class

    @property
    def model(self) -> Pipeline:
        return self._model

    @property
    def preprocessor(self) -> LCSPreprocessor:
        return self._preprocessor

    @property
    def baseline_mode(self) -> bool:
        return self._baseline_mode

    def _predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Simple function to preprocess data and predict using either a specific sklearn
        pipeline or the baseline

        Args:
            X (npt.ArrayLike): DataFrame or ndarray containing features for all
                datapoints to be predicted

        Returns:
            npt.ArrayLike: y_pred
        """
        X_processed = self.preprocessor.preprocess_data(X, verbose=False)
        if self.baseline_mode:
            y_pred = X_processed[BASELINE_MODEL_PREDICTION_COL].values
        else:
            logger.remove()
            logger.add(sys.__stdout__)
            stream = StreamToLogger(level=SKLEARN_LOGLEVEL)
            with contextlib.redirect_stdout(stream):
                y_pred = self.model.predict(X_processed)
        return X_processed, y_pred


class OutlierDetector:
    def __init__(self, epsilon: float) -> None:
        if epsilon < 0:
            raise ValueError("espilon tolerance must be a non-negative float.")
        self.epsilon = epsilon

    def detect(self, text_col_ref: npt.ArrayLike, text_col_alt: npt.ArrayLike) -> npt.ArrayLike:

        lengths_ref = np.array(list(map(lambda x: len(str(x)), text_col_ref)))
        lengths_alt = np.array(list(map(lambda x: len(str(x)), text_col_alt)))

        outliers = (lengths_ref > (1 + self.epsilon) * lengths_alt) | (
            lengths_alt > (1 + self.epsilon) * lengths_ref
        )

        return outliers


def load_model_from_file(
    model_filepath: Union[str, os.PathLike], estimator_ok: bool = True
) -> Union[Pipeline, BaseEstimator]:
    """Load model from file and raise and error if the format is incorrect

    Args:
        model_filepath (Union[str, os.PathLike]): model filepath as str o
        estimator_ok (bool): Make BaseEstimator a possible type fo type checking of the model.
            Default to True.

    Raises:
        FileNotFoundError: The file does not exists
        ValueError: The file is not a pickle file (suffix '.pkl')
        ValueError: The model is not of the expected type. Expected type is Pipeline or
            BaseEstimator if estimator_ok is True, only Pipeline otherwise

    Returns:
        Union[Pipeline, BaseEstimator]: sklearn model loaded from given pickle file
    """
    model_filepath = Path(model_filepath)
    if not Path(model_filepath).is_file():
        raise FileNotFoundError(model_filepath)
    if model_filepath.suffix != ".pkl":
        raise ValueError("Please give a path to pickle file")

    model = open_file(model_filepath)

    expected_type = Pipeline
    if estimator_ok is True:
        expected_type = (Pipeline, BaseEstimator)
    if not isinstance(model, expected_type):
        raise ValueError(
            f"Incompatible model: please give a filepath to a sklearn {expected_type} object"
        )

    return model
