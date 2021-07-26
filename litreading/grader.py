import numpy.typing as npt
from typing import Union

import os

import pandas as pd
from loguru import logger

from litreading.base import BaseModel, load_model_from_file
from litreading.config import DEFAULT_MODEL_FILEPATHS, DEFAULT_MODEL_TYPE


class Grader(BaseModel):
    def __init__(
        self, model_filepath: Union[str, os.PathLike] = None, baseline_mode: bool = False
    ) -> None:
        super().__init__(baseline_mode)
        if not baseline_mode:
            self._model = load_model_from_file(model_filepath)
            logger.info(f"Model loaded from {model_filepath}: {self._model}")

    def grade(self, X: pd.DataFrame) -> npt.ArrayLike:
        y_pred = self._predict(X)
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
        model_type = DEFAULT_MODEL_TYPE

    grader = Grader(
        model_filepath=DEFAULT_MODEL_FILEPATHS[model_type], baseline_mode=baseline_mode
    )

    return grader.grade(X)
