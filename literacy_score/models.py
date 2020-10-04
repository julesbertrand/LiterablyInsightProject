import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

from literacy_score.grading_utils import Dataset, open_file, save_file
from literacy_score.config import MODELS_PATH, PREPROCESSING_STEPS

# Logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ModelTrainer():
    def __init__(self, df, model_name):
        if model_name == 'rf':
            self.model = RandomForestRegressor
        elif model_name == 'xgb':
            self.model = XGBRegressor
        self.dataset = Dataset(df)

    def compute_features(self):
        self.dataset.preprocess_data(**PREPROCESSING_STEPS,
                            inplace = True
                            )
        self.features = self.dataset.compute_features(inplace = False)  # created df self.features

    def train_model(self, remove_outliers = True, test_set_size = .2):
        return

    def evaluate_model(self, visualize = True):
        return

    def save_model(self, path):
        return

    def hyperparams_tuning(self):
        return

    def plot_grid_search(self):
        return

    def feature_importance(self, visualize = True):
        return

if __name__ == "__main__":
    pass