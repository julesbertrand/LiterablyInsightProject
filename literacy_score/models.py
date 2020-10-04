import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

from literacy_score.grading_utils import open_file, save_file
from literacy_score.grading_utils import Dataset
from literacy_score.config import MODELS_PATH, PREPROCESSING_STEPS, SEED

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
        self.model_name = model_name
        if self.model_name == 'RF':
            self.model = RandomForestRegressor(random_state=SEED)
        elif self.model_name == 'XGB':
            self.model = XGBRegressor(random_state=SEED)
        self.data = Dataset(df, mode = 'train')

    def save_model(self, replace = False):
        save_file(self.model,
                path = MODELS_PATH,
                file_name = self.model_name + '.joblib',
                replace = replace
                )

    def compute_features(self):
        self.data.preprocess_data(**PREPROCESSING_STEPS,
                            inplace = True
                            )
        self.features = self.data.compute_features(inplace = False)  # created df self.features

    def train(self, remove_outliers = True, outliers_tol = .1, test_set_size = .2, save_model = True):
        X_train, X_test, Y_train, Y_test = self.prepare_train_test_set(remove_outliers = remove_outliers,
                                                                        outliers_tol = outliers_tol,
                                                                        test_set_size = test_set_size,
                                                                        save_model = save_model
                                                                        )
        logging.info("Training %s", self.model_name)
        self.model = self.model.fit(X_train, Y_train)
        if save_model:
            self.save_model()

    def prepare_train_test_set(self, remove_outliers = False, outliers_tol = .1, test_set_size = .2, save_model = True):
        if remove_outliers:
            mask = self.data.determine_outliers_mask(tol = outliers_tol)
            self.features = self.features[mask]
            self.datapoints = mask.sum()
            logging.info("Removed %i outliers, %i datapoints remaining for training/testing",
                        len(mask) - self.datapoints,
                        self.datapoints
                        )
        else:
            self.datapoints = len(self.features.index)
        X_train_raw, X_test_raw, Y_train, Y_test = train_test_split(self.features.drop(columns = ['human_wc']),
                                                    self.features['human_wc'],
                                                    test_size = test_set_size,
                                                    random_state = SEED
                                                   )
        logging.info("Fit scaler to training set and transform training and test set")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        if save_model:
            save_file(scaler, path = MODELS_PATH, file_name =  'standard_scaler.joblib', replace = False)
        return X_train, X_test, Y_train, Y_test

    def evaluate_model(self, visualize = True):
        return

    def hyperparams_tuning(self):
        return

    def plot_grid_search(self):
        return

    def feature_importance(self, visualize = True):
        return

if __name__ == "__main__":
    df = pd.read_csv("./data/wcpm_more.csv")
    # print(df.head())
    trainer = ModelTrainer(df, model_name = "XGB")
    trainer.compute_features()
    trainer.train()