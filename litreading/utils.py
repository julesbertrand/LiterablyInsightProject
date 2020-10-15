import os
import errno
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

import joblib

from litreading.config import MODELS_PATH

# Logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def open_file(file_path, sep=";"):
    _, extension = file_path.rsplit(".", 1)
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    if extension == "csv":
        f = pd.read_csv(file_path, sep=sep)
    else:
        f = joblib.load(file_path)
    return f


def save_file(file, path, file_name, replace=False):
    """save file with or without replacing previous versions, in cv or pkl
    input: file: python model or df to save
            path: path to save to
            file_name: name to give to the file, including extension
            replace: False if you do not want to delete and replace previous file with same name
    """
    if path[-1] != "/":
        path += "/"
    if not os.path.exists(path):
        raise FileNotFoundError
    file_name, extension = file_name.split(".")
    if replace:
        try:
            os.remove(file_name)
        except OSError:
            pass
    else:
        i = 0
        while os.path.exists(
            path + ".".join((file_name + "_{:d}".format(i), extension))
        ):
            i += 1
        file_name += "_{:d}".format(i)
    file_name = ".".join((file_name, extension))
    if extension == "csv":
        file.to_csv(path + file_name, index=False, sep=";", encoding="utf-8")
    else:
        joblib.dump(file, path + file_name, compress=1)
    logger.info("Saved file %s in dir %s", file_name, path)


class BaselineModel:
    def __init__(self):
        self.name = "BaselineModel"

    def fit(self, X_train, Y_train):
        return self

    def set_params(self, **params):
        logger.info("No params to be set in Baseline model")

    def get_params(self, **params):
        logger.info("No params in Baseline model")

    def predict(self, X_test):
        # prediction is the word correct count based on differ list
        self.scaler = open_file(MODELS_PATH + "standard_scaler.joblib")
        unscaled = self.scaler.inverse_transform(X_test)
        return unscaled[:, 0]
