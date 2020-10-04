""" This is a configuration file """

"""Package paths - Please update before installation!"""
# a path to a folder in which models are saved
MODELS_PATH = './literacy_score/models/'
# a path to a folder in which data is saved by default
DATA_PATH = './data/'

# the default type of model used for predictions and training
DEFAULT_MODEL = 'XGB'

# preprocessing steps
PREPROCESSING_STEPS = {
    'lowercase': True,
    'punctuation_free': True,
    'convert_num2words': True,
    'asr_string_recomposition': False,
}