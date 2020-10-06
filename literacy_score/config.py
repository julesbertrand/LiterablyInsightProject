""" This is a configuration file """

"""Package paths - Please update before installation!"""
# a path to a folder in which models are saved
MODELS_PATH = './literacy_score/models/'
# a path to a folder in which data is saved by default
DATA_PATH = './data/'
# test data path
TEST_DATA_PATH = './data/large_wcpm.csv'

# the default type of model used for predictions and training
DEFAULT_MODEL_TYPE = 'XGB'

# Default model files for each model
DEFAULT_MODEL_FILES = {
    'RF': 'rf_hypertuned.pkl',
    'XGB': 'xgb.pkl',
    'KNN': None,
    'StandardScaler': 'standard_scaler.joblib'
}

# preprocessing steps
PREPROCESSING_STEPS = {
    'lowercase': True,
    'punctuation_free': True,
    'convert_num2words': True,
    'asr_string_recomposition': False,
}

# seed for training
SEED = 105