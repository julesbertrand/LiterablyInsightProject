""" This is a configuration file """

"""Package paths - Please update before installation!"""
# a path to a folder in which models are saved
MODELS_PATH = './literacy_score/models/'
# a path to a folder in which data is saved by default
DATA_PATH = './data/'
# test data path
TEST_DATA_PATH = './data/large_wcpm.csv'

# preprocessing steps
PREPROCESSING_STEPS = {
    'lowercase': True,
    'punctuation_free': True,
    'convert_num2words': True,
    'asr_string_recomposition': False,
}

# Default type of model used for predictions and training
DEFAULT_MODEL_TYPE = 'XGB'
# Default model files for each model
DEFAULT_MODEL_FILES = {
    'RF': 'RF.joblib',
    'XGB': 'XGB.joblib',
    'KNN': 'KNN.joblib',
    'StandardScaler': 'standard_scaler.joblib'
}
# Default parameters config for training each type of model:
DEFAULT_PARAMS = {
    'RF': {'max_features': 6,
           'n_estimators': 200,
           'max_depth': 20,
           'min_samples_split': 8,
           'min_samples_leaf': 2,
           'bootstrap': True
          },
    'XGB': {'n_estimators': 5000,
            'learning_rate': 0.02,
            'max_depth': 7,
            'subsample': 0.81,
            'colsample_bytree': 0.86,
            'gamma': 5
            },
    'KNN': {'n_neighbors': 6,
            'weights': 'distance'
            }
    }


# seed for training
SEED = 105