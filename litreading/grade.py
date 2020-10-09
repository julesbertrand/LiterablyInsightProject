import os
import errno

import numpy as np
import pandas as pd

from litreading.utils import logger, save_file, open_file, BaselineModel
from litreading.dataset import Dataset
from litreading.config import MODELS_PATH, PREPROCESSING_STEPS, AVAILABLE_MODEL_TYPES, DEFAULT_MODEL_TYPE, DEFAULT_MODEL_FILES, DEFAULT_PARAMS

# main function
def grade_wcpm(df):
    data = DataGrader(df)
    return data.grade_wcpm()

class DataGrader(Dataset):
    def __init__(self,
                df, 
                prompt_col='prompt', 
                asr_col='asr_transcript', 
                duration_col='scored_duration',
                model_type=DEFAULT_MODEL_TYPE,
                ):
        Dataset.__init__(self,
                        df=df,
                        prompt_col=prompt_col,
                        asr_col=asr_col,
                        duration_col=duration_col,
                        mode='predict'
                        )
        self.scaler = self.__load_model(DEFAULT_MODEL_FILES['StandardScaler'])
        self.model_type = None
        self.set_model(model_type)

    def __load_model(self, model_file):
        model_path = MODELS_PATH + model_file
        logger.info("Loading model from %s", model_path)
        model = open_file(model_path)
        return model

    def set_model(self, model_type):
        """ Change model to another one
        input: model_type: 'RF' or 'XGB'
        """
        if self.model_type is model_type:
            pass  # no changes if same model wanted
        else:  # to be improved
            if model_type == 'Baseline':
                self.model = BaselineModel()
            elif model_type in AVAILABLE_MODEL_TYPES:
                self.model = self.__load_model(DEFAULT_MODEL_FILES[model_type])
            else:
                raise AttributeError("No such model is available. \
Please choose in '%s'." % "', '".join(AVAILABLE_MODEL_TYPES))
            self.model_type = model_type

    def grade_wcpm(self):
        """ preprocess, compute features and give grade all in one function
        """
        self.preprocess_data(**PREPROCESSING_STEPS,
                            inplace=True
                            )
        self.compute_features(inplace=True)
        self.estimate_wcpm(inplace=True)
        return self.get_data()

    def estimate_wcpm(self, inplace=False):
        """ take current model and estimate the wcpm with it
        """
        logger.debug("Estimating wcpm")
        self.features = self.scaler.transform(self.features)
        # wc = self.model.predict(self.features)
        # wc = pd.Series(wc, name='wc_estimations')
        # wcpm = wc.div(self.data[self.duration_col] /60, fill_value=0).round()
        # wcpm.rename('wcpm_estimations', inplace=True)
        wcpm = self.model.predict(self.features)
        wcpm = pd.Series(wcpm, name='wcpm_estimation')
        wcpm = wcpm.apply(lambda x: round(x, 1))
        if not inplace:
            return wcpm
        else:
            self.data['wcpm_estimation'] = wcpm



if __name__ == "__main__":
    df = pd.read_csv("./data/wcpm_more.csv")
    d = DataGrader(df.drop(columns='human_transcript').loc[:20], model_type='XGB')
    d.grade_wcpm()
    print(d.data.head(20))

