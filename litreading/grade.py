"""
DataGrader class to predict WCPM
"""

import pandas as pd

from litreading.config import (
    ASR_TRANSCRIPT_COL,
    AVAILABLE_MODEL_TYPES,
    DEFAULT_MODEL_FILES,
    DEFAULT_MODEL_TYPE,
    DURATION_COL,
    MODELS_PATH,
    PREPROCESSING_STEPS,
    PROMPT_TEXT_COL,
)
from litreading.dataset import Dataset
from litreading.utils import BaselineModel, logger, open_file


# main function
def grade_wcpm(df: pd.DataFrame, only_wcpm: bool = False) -> pd.DataFrame:
    """ Instanciate Datagrader and grade """
    data = DataGrader(df)
    return data.grade_wcpm(only_wcpm=only_wcpm)


class DataGrader(Dataset):
    """
    Grader tool
    Methods: set_model
            grade_wcpm: will run Dataset.preprocessing, Dataset.compute_features and self.estimate_wcpm
            estimate_wcpm: use the model on Dataset.features to estimate wcpm
    Static methods: __load_model
    """

    def __init__(
        self,
        df,
        prompt_col=PROMPT_TEXT_COL,
        asr_col=ASR_TRANSCRIPT_COL,
        duration_col=DURATION_COL,
        model_type=DEFAULT_MODEL_TYPE,
    ):
        Dataset.__init__(
            self,
            df=df,
            prompt_col=prompt_col,
            asr_col=asr_col,
            duration_col=duration_col,
            mode="predict",
        )
        self.scaler = self.__load_model(DEFAULT_MODEL_FILES["StandardScaler"])
        self.model_type = None
        self.set_model(model_type)

    @staticmethod
    def __load_model(model_file, print_info=False):
        """ Used to load scaler or model using utils.open_file() """
        model_path = MODELS_PATH / model_file
        if print_info:
            logger.info("Loading model from %s", model_path)
        model = open_file(model_path)
        return model

    def set_model(self, model_type):
        """Change model to another one
        input: model_type: 'RF' or 'XGB'
        """
        if self.model_type is model_type:
            pass  # no changes if same model wanted
        else:  # to be improved
            if model_type == "Baseline":
                self.model = BaselineModel()
            elif model_type in AVAILABLE_MODEL_TYPES:
                self.model = self.__load_model(DEFAULT_MODEL_FILES[model_type], print_info=True)
            else:
                raise AttributeError(
                    "No such model is available. \
Please choose in '%s'."
                    % "', '".join(AVAILABLE_MODEL_TYPES)
                )
            self.model_type = model_type

    def grade_wcpm(self, only_wcpm=False):
        """preprocess, compute features and give grade all in one function """
        self.preprocess_data(**PREPROCESSING_STEPS, inplace=True)
        self.compute_features(inplace=True)
        self.estimate_wcpm(inplace=True)
        if only_wcpm:
            return self.data["wcpm_estimation"]
        return self.data

    def estimate_wcpm(self, inplace=False):
        """ Scale features, use current model to estimate the wcpm """
        logger.debug("Estimating wcpm")
        self._features = self.scaler.transform(self._features)
        wcpm = self.model.predict(self._features)
        wcpm = pd.Series(wcpm, name="wcpm_estimation")
        wcpm = wcpm.apply(lambda x: round(x, 1))
        if not inplace:
            return wcpm
        self.data["wcpm_estimation"] = wcpm
