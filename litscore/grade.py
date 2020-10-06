import os
import errno

import numpy as np
import pandas as pd

from literacy_score.utils import logger, save_file, open_file, BaselineModel
from literacy_score.dataset import Dataset
from literacy_score.config import MODELS_PATH, DEFAULT_MODEL_TYPE, PREPROCESSING_STEPS, DEFAULT_MODEL_FILES

# main function
def grade_wcpm(df):
    data = DataGrader(df)
    return data.grade_wcpm()

class DataGrader(Dataset):
    def __init__(self,
                df, 
                prompt_col = 'prompt', 
                asr_col = 'asr_transcript', 
                duration_col = 'scored_duration',
                model_type = DEFAULT_MODEL_TYPE,
                ):
        Dataset.__init__(self,
                        df=df,
                        prompt_col = prompt_col,
                        asr_col = asr_col,
                        duration_col = duration_col,
                        mode = 'predict'
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
            if model_type in ['RF', 'XGB', 'KNN']:
                self.model = self.__load_model(DEFAULT_MODEL_FILES[model_type])
            elif model_type == 'Baseline':
                self.model = BaselineModel()
            else:
                logger.error("No such model is available: %s in %s. \
                            please choose between 'Baseline', 'RF', 'XGB' and 'KNN'.",
                            model_type,
                            MODELS_PATH
                            )
                return
            self.model_type = model_type

    def grade_wcpm(self):
        """ preprocess, compute features and give grade all in one function
        """
        self.preprocess_data(**PREPROCESSING_STEPS,
                            inplace = True
                            )
        self.features = self.compute_features(inplace = False)
        self.estimate_wcpm(inplace = True)
        return self.get_data()

    def estimate_wcpm(self, inplace = False):
        """ take current model and estimate the wcpm with it
        """
        logger.debug("Estimating wcpm")
        self.features = self.scaler.transform(self.features)
        wc = self.model.predict(self.features)
        wc = pd.Series(wc, name = 'wc_estimations')
        wcpm = wc.div(self.data[self.duration_col] /60, fill_value = 0).round()
        wcpm.rename('wcpm_estimations', inplace = True)
        if not inplace:
            return wcpm
        else:
            self.data['wcpm_estimations'] = wcpm



if __name__ == "__main__":
    df = pd.read_csv("./data/wcpm_more.csv")
    d = DataGrader(df.drop(columns = 'human_transcript').loc[:20], model_type = 'XGB')
    d.grade_wcpm()
    print(d.data.head(20))

