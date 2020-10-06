import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer 

from literacy_score.utils import logger, save_file, open_file, BaselineModel
from literacy_score.dataset import Dataset
from literacy_score.config import MODELS_PATH, PREPROCESSING_STEPS, SEED, DEFAULT_MODEL_TYPE

class ModelTrainer():
    def __init__(self,
                df, 
                model_type = DEFAULT_MODEL_TYPE,
                prompt_col = 'prompt', 
                asr_col = 'asr_transcript',
                human_col = 'human_transcript',
                duration_col = 'scored_duration',
                human_wcpm_col = 'human_wcpm'
                ):
        self.set_new_model(model_type)
        self.data = Dataset(df,
                            prompt_col = 'prompt', 
                            asr_col = 'asr_transcript',
                            human_col = 'human_transcript',
                            duration_col = 'scored_duration',
                            human_wcpm_col = 'human_wcpm',
                            mode = 'train'
                            )
    
    def set_new_model(self, model_type, params = {}, inplace = True):
        if model_type == 'RF':
            estimator = RandomForestRegressor(random_state=SEED)
        elif model_type == 'XGB':
            estimator = XGBRegressor(random_state=SEED)
        elif model_type == 'KNN':
            estimator = KNeighborsRegressor()
        elif model_type == 'Baseline':
            estimator = BaselineModel()
        else:
            logger.error("Sorry, training for mode_type %s has not been implemented yet.", model_type)
            return
        self.model_type = model_type
        if not inplace:
            return estimator.set_params(**params)
        self.model = estimator
        if params != {}:
            self.set_model_params(params)
    
    def set_model_params(self, params = {}):
        self.model.set_params(**params)

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
                    file_name = self.model_type + '.joblib',
                    replace = replace
                    )

    def compute_features(self):
        self.data.preprocess_data(**PREPROCESSING_STEPS,
                            inplace = True
                            )
        # create dataframe of features
        self.features = self.data.compute_features(inplace = False)  

    def train(self, params = {}):
        self.model.set_params(**params)
        try:
            self.X_train
        except AttributeError:
            logger.error("X_train, Y_train not defined: Please prepare train and test \
                        set before training by calling ModelTrainer.prepare_train_test_set()")
            return
        logger.info("Training %s", self.model_type)
        self.model = self.model.fit(self.X_train, self.Y_train)

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
        n = len(self.test_idx)
        stats_summary = pd.DataFrame(
            data = {
                'Mean Error': [
                    stats['wcpm_estimation_error'].mean(axis=0).round(2),
                    stats['wcpm_estimation_error'].std(axis=0).round(2),
                    None, None
                ],
                'Mean Error %': [
                    stats['wcpm_estimation_error_%'].mean(axis=0).round(2),
                    stats['wcpm_estimation_error_%'].std(axis=0).round(2),
                    None, None
                ],
                'Mean abs. Error': [
                    stats['wcpm_estimation_abs_error'].abs().mean(axis=0).round(2),
                    stats['wcpm_estimation_abs_error'].abs().std(axis=0).round(2),
                    None, None
                ],
                'Mean abs. Error %': [
                    stats['wcpm_estimation_abs_error_%'].mean(axis=0).round(4) * 100,
                    stats['wcpm_estimation_abs_error_%'].std(axis=0).round(4) * 100,
                    None, None
                ],
                'RMSE': [
                    round(np.sqrt((stats['wcpm_estimation_error'] ** 2).mean(axis = 0)), 2),
                    round(np.sqrt((stats['wcpm_estimation_error'] ** 2).std(axis = 0)), 2),
                    None, None
                ],
                'Error > 1%': [None, None,
                    (stats['wcpm_estimation_abs_error_%'] > 0.01).sum().round(0),
                    round((stats['wcpm_estimation_abs_error_%'] > 0.01).sum() / n, 4) * 100
                ],
                'Error > 5%': [None, None,
                    (stats['wcpm_estimation_abs_error_%'] > 0.05).sum().round(0),
                    round((stats['wcpm_estimation_abs_error_%'] > 0.05).sum() / n, 4) * 100
                ],
                'Error > 10%': [None, None,
                    (stats['wcpm_estimation_abs_error_%'] > 0.1).sum().round(0),
                    round((stats['wcpm_estimation_abs_error_%'] > 0.1).sum() / n, 4) * 100
                ]
            },
            index = ['mean', 'std', 'absolute #', '% of test set']
        )
        if visualize:
            self.plot_wcpm_distribution(stats=stats,
                                        x='wcpm_estimation_error_%',
                                        stat='count',
                                        binwidth=.01
                                       )
        return stats, stats_summary

    def grid_search(self,
                    model_type,
                    cv_params, 
                    cv_folds = 5, 
                    verbose = 2, 
                    scoring_metric = 'r2'
                   ):
        params = dict()  # for params in cv_params with unique value, set directly to estimator
        params_grid = dict()  # for params to be actually cross-validated
        for key, value in cv_params.items():
            if len(value) == 1:
                params[key] = value[0]
            else:
                params_grid[key] = value
        estimator = self.set_new_model(model_type=model_type,
                                       params = params,
                                       inplace = False
                                      )
        print("\n" + " Estimator: ".center(120, "-"))
        print(estimator.__class__.__name__)
        print("\n" + " Metric for evaluation: ".center(120, "-"))
        print(scoring_metric)
        if len(params.keys()) > 0:
            print("\n" + " Fixed params: ".center(120, "-"))
            [print(key, value) for key, value in params.items()]
        print("\n" + " Params to be tested: ".center(120, "-"))
        [print(key, value) for key, value in params_grid.items()]
        params_list = list(itertools.product(*params_grid.values()))
        n_combi = len(params_list)
        print("\n" + " # of possible combinations to be cross-validated: {:d}".format(n_combi))
        answer = input("\n" + "Continue with these c-v parameters ? (y/n)  ")
        if answer == "n" or answer == "no":
            print("Please redefine inputs.")
            return

        grid_search = GridSearchCV(estimator = estimator,
                                    param_grid = params_grid,
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

    def plot_grid_search(self,
                         cv_results,
                         x,
                         hue=None,
                         y='mean_test_score',
                         log_scale=True
                        ):
        # Get Test Scores Mean and std for each grid search
        cv_results = pd.DataFrame(cv_results)
        if hue is not None:
            hue = 'param_' + hue
        x = 'param_' + x
        # Plot Grid search scores
        plt.style.use("seaborn-darkgrid")
        _, ax = plt.subplots(1,1, figsize = (10, 4))

        # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
        sns.lineplot(data=cv_results, x=x, y=y, hue=hue, palette='Set2')

        ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
        ax.set_xlabel(x, fontsize=16)
        ax.set_ylabel('CV Average Score', fontsize=16)
        ax.legend(loc="best", fontsize=15)
        if log_scale:
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
        _, ax = plt.subplots(1, 1, figsize=(8, max(8, 0.2 * len(idx))))
        sns.barplot(x=importance[idx], y=labels[idx], color=sns.color_palette()[0])
        for i, val in enumerate(importance[idx]):
            ax.text(val + 0.01, i, s="{:.3f}".format(val), ha='left', va='center')
        ax.set_title("Feature importance for current model", fontsize=16)
        ax.set_xlim(0, max(importance[idx]) + 0.03)
        plt.show()

    @staticmethod
    def plot_wcpm_distribution(stats, x, stat='count', binwidth = .01):
        plt.style.use("seaborn-darkgrid")
        _, ax = plt.subplots(1, 1,figsize = (16, 6))
        sns.histplot(ax=ax,
                        data=stats,
                        x=x,
                        stat=stat,
                        binwidth=binwidth
                    )
        ax.set_title("Distribution of errors",fontsize=20, fontweight='bold')
        ax.set_xlabel(x, fontsize=16)
        if '%' in x:
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_ylabel('count', fontsize=16)
        plt.show()

    @staticmethod
    def plot_wcpm_scatter(stats, y = 'wcpm_estimation_error_%'):
        plt.style.use("seaborn-darkgrid")
        _, ax = plt.subplots(1, 1, figsize=(16, 6))
        sns.scatterplot(data=stats, x='human_wcpm', y=y)
        ax.set_title('Graph of %s' % y, fontsize=20, fontweight='bold')
        ax.set_xlabel('human wcpm', fontsize=16)
        ax.set_ylabel(y, fontsize=16)
        if '%' in y:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

if __name__ == "__main__":
    df = pd.read_csv("./data/wcpm_more.csv")
    # print(df.head())
    # df = df.loc[:50]
    trainer = ModelTrainer(df, model_type = "Baseline")
    trainer.compute_features()
    trainer.prepare_train_test_set(remove_outliers = True, outliers_tol = .1)
    # trainer.train(params = {})
    # trainer.save_model(scaler = True, model = False)
    gd = trainer.grid_search(model_type = 'XGB',
                             cv_params={'learning_rate': [0.05],
                                        # 'n_estimators': list(np.arange(100, 500, 100))
                                        },
                            cv_folds=5,
                            scoring_metric = 'r2')
    # trainer.plot_grid_search(gd.cv_results_, x='n_estimators', hue=None)
    # trainer.feature_importance()
    stats, summary = trainer.evaluate_model(visualize = True)
    print(summary)