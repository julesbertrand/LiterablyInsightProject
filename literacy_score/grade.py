import os
import errno

import numpy as np
import pandas as pd

import joblib

import ast  # preprocessing ast to litteral
import re  # preprocessing
from num2words import num2words  # preprocessing 
import string # preprocessing punctuation

from literacy_score.grading_utils import open_file, save_file, logger, Dataset
from literacy_score.config import MODELS_PATH, DEFAULT_MODEL, PREPROCESSING_STEPS

# main function
def grade_wcpm(df):
    data = DataGrader(df)
    return data.grade_wcpm()

class DataGrader():
    def __init__(self,
                df, 
                prompt_col = 'prompt', 
                asr_col = 'asr_transcript', 
                duration_col = 'scored_duration',
                model_name = DEFAULT_MODEL,
                model_file = ''
                ):
        self.data = Dataset(df,
                            prompt_col = prompt_col,
                            asr_col = asr_col,
                            duration_col = duration_col,
                            mode = 'predict'
                            )

        self.scaler = self.__load_model('standard_scaler.joblib')
        if model_file == '':
            self.model = self.__load_model('XGB_0.joblib')
        else:
            self.model = self.__load_model(model_file)
        self.model_name = model_name

    def __load_model(self, model_file):
        model_path = MODELS_PATH + model_file
        logger.info("Loading model from %s", model_path)
        model = open_file(model_path)
        return model

    def set_model(self, model_name):
        """ Change model to another one
        input: model_name: 'RF' or 'XGB'
        """
        if self.model_name == model_name:
            pass  # no changes if same model wanted
        else:  # to be improved
            if model_name == "RF":
                self.model = self.__load_model('rf_hypertuned.pkl')
                self.model_name = model_name
            elif model_name == "XGB":
                self.model = self.__load_model('xgb_hypertuned.pkl')
                self.model_name = model_name
            else:
                logger.error("No such model is available: %s in %s. please choose between 'RF' and 'XGB'", model_name, MODELS_PATH)

    def grade_wcpm(self):
        """ preprocess, compute features and give grade all in one function
        """
        self.data.preprocess_data(**PREPROCESSING_STEPS,
                            inplace = True
                            )
        self.features = self.data.compute_features(inplace = False)  # created df self.features
        self.estimate_wcpm(inplace = True)
        return self.data.get_data()

    def estimate_wcpm(self, inplace = False, model = None):
        """ take current model and estimate the wcpm with it
        """
        if model is None:
            model = self.model
        else:
            self.set_model(model_name = model)
        logger.info("Estimating wcpm")
        self.features = self.scaler.transform(self.features)
        wc = self.model.predict(self.features)
        wc = pd.Series(wc, name = 'wc_estimations')
        wcpm = wc.div(self.data.data[self.data.duration_col] /60, fill_value = 0).round()
        wcpm.rename('wcpm_estimations', inplace = True)
        if not inplace:
            return wcpm
        else:
            self.data.data['wcpm_estimations'] = wcpm



if __name__ == "__main__":
    df = pd.read_csv("./data/wcpm_more.csv")
    d = DataGrader(df.drop(columns = 'human_transcript').loc[:20], model_file = 'xgb.pkl')
    d.grade_wcpm()
    print(d.data.data.head(20))

