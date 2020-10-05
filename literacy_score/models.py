import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer 

from literacy_score.grading_utils import open_file, save_file, logger, Dataset
from literacy_score.config import MODELS_PATH, PREPROCESSING_STEPS, SEED, DEFAULT_MODEL

class ModelTrainer():
    def __init__(self,
                df, 
                model_name = DEFAULT_MODEL,
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

    def save_model(self, scaler = False, model = False, replace = False):
        """ Save both scaler and trained / untrained model
        """
        if scaler:
            try:
                save_file(self.scaler,
                        path = MODELS_PATH,
                        file_name =  'standard_scaler.joblib',
                        replace = replace)
            except AttributeError:
                logger.error("scaler not defined: Please fit a scaler before saving \
                            it by calling ModelTrainer.prepare_train_test_set()")
        if model:
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

    def train(self):
        logger.info("Training %s", self.model_name)
        try:
            self.model = self.model.fit(self.X_train, self.Y_train)
        except AttributeError:
            logger.error("X_train, Y_train not defined: Please prepare train and test \
                        set before training by calling ModelTrainer.prepare_train_test_set()")

    def prepare_train_test_set(self,
                            remove_outliers = False,
                            outliers_tol = .1, 
                            test_set_size = .2, 
                            inplace = True
                            ):
        if remove_outliers:
            mask = self.data.determine_outliers_mask(tol = outliers_tol)
            self.features = self.features[mask]
            self.datapoints = mask.sum()
            logger.info("Removed %i outliers, %i datapoints remaining for training/testing",
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
        logger.debug("Fit scaler to training set and transform training and test set")
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)
        if not inplace:
            return X_train, X_test, Y_train, Y_test
        else:
            self.X_train, self.X_test = X_train, X_test
            self.Y_train, self.Y_test = Y_train, Y_test

    def evaluate_model(self, visualize = True):
        Y_pred = self.model.predict(self.X_test)
        stats = self.data.compute_stats(Y_pred, self.test_idx)
        print("avg diff between human and asr wcpm: ",
                stats['wcpm_estimation_error'].mean(axis=0).round(2))
        print("std of diff between human and asr wcpm: ",
                stats['wcpm_estimation_error'].std(axis=0).round(2))
        print("\n")
        print("avg abs diff between human and asr wcpm: ",
                stats['wcpm_estimation_abs_error'].abs().mean(axis=0).round(2))
        print("std of abs diff between human and asr wcpm: ", 
                stats['wcpm_estimation_abs_error'].abs().std(axis=0).round(2))
        print("\n")
        print("avg abs diff in % between human and asr wcpm: ", 
                stats['wcpm_estimation_abs_error_%'].mean(axis=0).round(4) * 100)
        print("std of abs diff in % between human and asr wcpm: ", 
                stats['wcpm_estimation_abs_error_%'].std(axis=0).round(4) * 100)
        if visualize:
            plt.style.use("seaborn-darkgrid")
            plt.figure(figsize = (16, 6))
            sns.displot(x=stats['wcpm_estimation_error_%'], color=sns.color_palette()[0])
            plt.title("Distribution of errors %", fontsize=16)
            plt.show()
        return stats

    def grid_search(self, cv_params, cv_folds = 5, verbose = 2, scoring_metric = 'r2'):
        if self.model_name == 'RF':
            estimator = RandomForestRegressor(random_state = SEED)
        elif self.model_name == 'XGB':
            estimator = XGBRegressor(random_state = SEED)
        grid_search = GridSearchCV(estimator = estimator,
                                    param_grid = cv_params, 
                                    scoring = scoring_metric,
                                    cv = cv_folds,
                                    n_jobs = -1, 
                                    verbose = verbose
                                    )
        try:
            grid_search.fit(self.X_train, self.Y_train)
        except AttributeError:
            logger.error("X_train, Y_train not defined: Please prepare train and test \
                        set before training by calling ModelTrainer.prepare_train_test_set()")
        self.model  = grid_search.best_estimator_
        return grid_search

    def plot_grid_search(self, cv_results, x, hue=None, y='mean_test_score', log_scale=True):
        # Get Test Scores Mean and std for each grid search
        cv_results = pd.DataFrame(cv_results)
        hue = 'param_' + hue
        x = 'param_' + x
        # Plot Grid search scores
        plt.style.use("seaborn-darkgrid")
        _, ax = plt.subplots(1,1, figsize = (12, 8))

        # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
        sns.lineplot(data=cv_results, x=x, y=y, hue=hue, palette='Set2')

        ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
        ax.set_xlabel(x, fontsize=16)
        ax.set_ylabel('CV Average Score', fontsize=16)
        ax.legend(loc="best", fontsize=15)
        ax.set_xscale('log')
        ax.grid('on')
        plt.show()

    def feature_importance(self, threshold = 0.001):
        """
        Compute and plot feature importance for tree based methods from sklearn or similar
        input: model already fitted
            features: names of the features
            threshold: minimum feature importance for the feature to be plotted
        """
        importance = self.model.feature_importances_
        idx = [x[0] for x in enumerate(importance) if x[1] > threshold]
        labels = self.features.columns[idx]
        importance = importance[idx]
        idx = np.argsort(importance)[::-1]

        plt.style.use("seaborn-darkgrid")
        plt.figure(figsize = (8, max(8, 0.2 * len(idx))))
        sns.barplot(x=importance[idx], y=labels[idx], color=sns.color_palette()[0])
        for i, val in enumerate(importance[idx]):
            plt.text(val + 0.01, i, s="{:.3f}".format(val), ha='left', va='center')
        plt.title("Feature importance for current model", fontsize=16)
        plt.xticks(fontsize=14)
        plt.gca().set_xlim(0, max(importance[idx]) + 0.03)
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv("./data/wcpm_more.csv")
    # print(df.head())
    df = df.loc[:50]
    trainer = ModelTrainer(df, model_name = "XGB")
    trainer.compute_features()
    trainer.prepare_train_test_set(remove_outliers = True, outliers_tol = .1)
    trainer.train()
    trainer.feature_importance()
    # trainer.save_model(scaler = True, model = False)
    # gd = trainer.grid_search(cv_params={'learning_rate': [0.05, 0.1, 0.3],
    #                                     'n_estimators': list(np.arange(10, 500, 20))
    #                                     },
    #                         cv_folds=5,
    #                         scoring_metric = 'neg_mean_squared_log_error')
    # trainer.plot_grid_search(gd.cv_results_, x='n_estimators', hue='learning_rate')
    # trainer.evaluate_model(visualize = False)