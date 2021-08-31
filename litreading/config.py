import os

""" General utils """
ROOT_PATH = "."

""" Logging """
STDOUT_LOGLEVEL = "INFO"


""" Column names in dataset """
PROMPT_TEXT_COL = "prompt"
ASR_TRANSCRIPT_COL = "asr_transcript"
HUMAN_TRANSCRIPT_COL = "human_transcript"
HUMAN_WCPM_COL = "human_wcpm"
DURATION_COL = "scored_duration"


""" Model general params """
PREPROCESSING_STEPS = dict(
    to_lowercase=True,
    remove_punctuation=True,
    convert_num2words=True,
    asr_string_recomposition=False,
)

SEED = 105
BASELINE_MODEL_PREDICTION_COL = "correct_words_pm"


""" Default models"""
DEFAULT_MODEL_TYPE = "xgboost"
DEFAULT_MODEL_FILEPATHS = {
    "xgboost": os.path.join(ROOT_PATH, "models/default_xgb.pkl"),
    "test": os.path.join(ROOT_PATH, "models/model_test.pkl"),
}

DEFAULT_PARAMS = {"xgboost": {}}
