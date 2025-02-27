import numpy.typing as npt
from typing import Optional, Union

import os

import pandas as pd
from loguru import logger

from litreading.base import BaseModel, load_model_from_file
from litreading.config import DEFAULT_MODEL_FILEPATHS


class Grader(BaseModel):
    def __init__(
        self, model_filepath: Optional[Union[str, os.PathLike]] = None, baseline_mode: bool = False
    ) -> None:
        super().__init__(baseline_mode)
        if not baseline_mode:
            if model_filepath is None:
                raise ValueError(
                    "Please provide a model_filepath when you're not using baseline mode."
                )
            self._model = load_model_from_file(model_filepath)
            logger.info(f"Model loaded from {model_filepath}: {self.model}")

    def grade(self, X: pd.DataFrame, return_processed_data: bool = False) -> npt.ArrayLike:
        """predict grades using BaseModel predict: will do raw data preprocessing + pipeline prediction

        Args:
            X (pd.DataFrame): [description]
            return_processed_data (bool, optional): [description]. Defaults to False.

        Returns:
            npt.ArrayLike: [description]
        """
        X_processed, y_pred = self._predict(X)
        if return_processed_data:
            return X_processed, y_pred
        return y_pred


def grade_wcpm(X, model_type: str = None, baseline_mode: bool = False):
    """shortcut to grade from a df

    Args:
        X ([type]): [description]
        model_type (str, optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if model_type is None:
        model_type = "default"

    if DEFAULT_MODEL_FILEPATHS.get(model_type) is None:
        msg = "This model type is not supported. Please choose one among:\n"
        msg += "\n".join(DEFAULT_MODEL_FILEPATHS.keys())
        raise ValueError(msg)

    grader = Grader(
        model_filepath=DEFAULT_MODEL_FILEPATHS[model_type], baseline_mode=baseline_mode
    )

    return grader.grade(X)
