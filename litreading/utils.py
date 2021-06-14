"""
Helper functions: open_file and save_file
Classes: BaselineModel with methods get_params, set_params, fit and predict
"""
import logging
import os
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from litreading.config import MODELS_PATH

# Logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def open_file(filepath: Union[str, Path], sep: str = ";"):
    """ Function to open files from filepath, either cs or joblib or pkl """
    filepath = Path(filepath)
    extension = filepath.suffix
    if not filepath.exists():
        raise FileNotFoundError(filepath)
    if extension == "csv":
        f = pd.read_csv(filepath, sep=sep)
    else:
        f = joblib.load(filepath)
    return f


def save_file(file, dirpath: Union[str, Path], file_name: str, replace=False):
    """
    Save file with or without replacing previous versions, in cv or pkl
    input: file: python model or df to save
            path: path to save to
            file_name: name to give to the file, including extension
            replace: False if you do not want to delete and replace previous file with same name
    """
    dirpath = Path(dirpath)
    if not dirpath.exists():
        dirpath.mkdir(parents=True)
    file_name, extension = file_name.split(".")
    if replace:
        try:
            os.remove(dirpath / file_name)
        except OSError:
            pass
    else:
        i = 0
        while True:
            path = dirpath / ".".join((file_name + "_{:d}".format(i), extension))
            if not path.exists():
                break
            i += 1
        file_name += "_{:d}".format(i)
    file_name = ".".join((file_name, extension))
    if extension == "csv":
        file.to_csv(dirpath + file_name, index=False, sep=";", encoding="utf-8")
    elif extension == "joblib":
        joblib.dump(file, dirpath + file_name, compress=1)
    else:
        raise NotImplementedError(f"File type not handled: {extension}")
    logger.info(f"Saved file {file_name} in dir {dirpath}")


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
