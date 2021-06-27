""" This is a configuration file """

from pathlib import Path

"""Package paths - Please update before installation!"""
# a path to a folder in which models are saved
MODELS_PATH = Path("./litreading/models/")

""" Column names in dataset """
PROMPT_TEXT_COL = "prompt"
ASR_TRANSCRIPT_COL = "asr_transcript"
HUMAN_TRANSCRIPT_COL = "human_transcript"
HUMAN_WCPM_COL = "human_wcpm"
DURATION_COL = "scored_duration"


""" Preprocessing steps - Please update considering your preprocessing"""
# preprocessing steps
PREPROCESSING_STEPS = {
    "lowercase": True,
    "punctuation_free": True,
    "convert_num2words": True,
    "asr_string_recomposition": False,
}

BASELINE_MODEL_PREDICTION_COL = "correct_words_pm"

""" Models available and config - Please update carefully or it could break the code """
# Available models for training and prediction
AVAILABLE_MODEL_TYPES = ["Baseline", "RF", "XGB", "KNN"]
# Default type of model used for predictions and training
DEFAULT_MODEL_TYPE = "XGB"
# Default model files for each model
DEFAULT_MODEL_FILES = {
    "RF": "RF.joblib",
    "XGB": "XGB.joblib",
    "KNN": "KNN.joblib",
    "StandardScaler": "standard_scaler.joblib",
}
# Default parameters config for training each type of model, can be updated.
DEFAULT_PARAMS = {
    "Baseline": {},
    "RF": {
        "max_features": 6,
        "n_estimators": 600,
        "max_depth": 10,
        "min_samples_split": 4,
        "min_samples_leaf": 1,
        "bootstrap": True,
    },
    "XGB": {
        "n_estimators": 800,
        "learning_rate": 0.02,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "gamma": 5,
    },
    "KNN": {"n_neighbors": 4, "weights": "distance"},
}


# seed for training
SEED = 105
