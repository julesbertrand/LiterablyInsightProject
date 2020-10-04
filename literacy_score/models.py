import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer 

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
    def __init__(self,
                df, 
                model_name,
                prompt_col = 'prompt', 
                asr_col = 'asr_transcript',
                human_col = 'human_transcript',
                duration_col = 'scored_duration',
                human_wcpm_col = 'human_wcpm'
                ):
        self.model_name = model_name
        if self.model_name == 'RF':
            self.model = RandomForestRegressor(random_state=SEED)
        elif self.model_name == 'XGB':
            self.model = XGBRegressor(random_state=SEED)
        self.data = Dataset(df,
                            prompt_col = 'prompt', 
                            asr_col = 'asr_transcript',
                            human_col = 'human_transcript',
                            duration_col = 'scored_duration',
                            human_wcpm_col = 'human_wcpm',
                            mode = 'train'
                            )

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

    def train(self, save_model = True):
        logging.info("Training %s", self.model_name)
        try:
            self.model = self.model.fit(self.X_train, self.Y_train)
        except AttributeError:
            logging.error("X_train not defined: Please prepare train and test set before training by calling ModelTrainer.prepare_train_test_set()")
        if save_model:
            self.save_model()

    def prepare_train_test_set(self, remove_outliers = False, outliers_tol = .1, test_set_size = .2, save_scaler = False, inplace = True):
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
        self.test_idx = X_test_raw.index
        logging.info("Fit scaler to training set and transform training and test set")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        if save_scaler:
            save_file(scaler, path = MODELS_PATH, file_name =  'standard_scaler.joblib', replace = False)
        if not inplace:
            return X_train, X_test, Y_train, Y_test
        else:
            self.X_train, self.X_test = X_train, X_test
            self.Y_train, self.Y_test = Y_train, Y_test

    def evaluate_model(self, visualize = True):
        Y_pred = self.model.predict(self.X_test)
        stats = self.data.compute_stats(Y_pred, self.test_idx)
        print("avg diff between human and asr wcpm: ",  stats['wcpm_estimation_error'].mean(axis=0))
        print("std of diff between human and asr wcpm: ",  stats['wcpm_estimation_error'].std(axis=0))
        print("\n")
        print("avg abs diff between human and asr wcpm: ", stats['wcpm_estimation_abs_error'].abs().mean(axis=0))
        print("std of abs diff between human and asr wcpm: ", stats['wcpm_estimation_abs_error'].abs().std(axis=0))
        print("\n")
        print("avg abs diff in % between human and asr wcpm: ", stats['wcpm_estimation_abs_error_%'].mean(axis=0) * 100)
        print("std of abs diff in % between human and asr wcpm: ", stats['wcpm_estimation_abs_error_%'].std(axis=0) * 100)
        if visualize:
            stats.groupby('wcpm_estimation_error').count()['human_wcpm'].plot(figsize=(16, 6))
            plt.show()
        return stats

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
    trainer.prepare_train_test_set(remove_outliers = True, outliers_tol = .1, )
    trainer.train()
    trainer.evaluate_model()