import os
import errno
import logging

import numpy as np
import pandas as pd

import joblib

import ast  # preprocessing ast to litteral
import re  # preprocessing
from num2words import num2words  # preprocessing 
import string # preprocessing punctuation

from literacy_score.grading_utils import open_file, save_file,
from literacy_score.grading_utils import compare_text, get_errors_dict, avg_length_of_words, Dataset
from literacy_score.config import DATA_PATH, MODEL_PATH, DEFAULT_MODEL

# Logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def grade_wcpm(df):
    data = DataGrader(df)
    return data.grade_wcpm()

class DataGrader():
    prompt_col = 'prompt'
    asr_col = 'asr_transcript'
    duration_col = 'scored_duration'

    def __init__(self, df, prompt_col = prompt_col, asr_col = asr_col, duration_col = duration_col, model = 'xgb', model_file = ''):
        self.data = Dataset(df,
                            prompt_col = prompt_col,
                            asr_col = asr_col,
                            duration_col = duration_col
                            )

        self.scaler = self.__load_model('standard_scaler.joblib')
        self.model = self.__load_model('rf_hypertuned.pkl')
        self.model_name = "RF"

    def __load_model(self, model_name):
        model_path = MODEL_PATH + model_name
        logging.info("Loading model from %s", model_path)
        with open(model_path, 'rb') as f:
            model = joblib.load(f)  
        return model

    def set_model(self, model_name):
        if self.model_name == model_name:
            pass
        else:
            if model_name == "RF":
                self.model = self.__load_model('rf_hypertuned.pkl')
                self.model_name = model_name
            elif model_name == "XGB":
                self.model = self.__load_model('xgb_hypertuned.pkl')
                self.model_name = model_name
            else:
                logging.error("No such model is available: %s in %s. please choose between 'RF' and 'XGB'", model_name, MODEL_PATH)

    def grade_wcpm(self):
        self.data.preprocess_data(lowercase = True,
                            punctuation_free = True,
                            convert_num2words = True,
                            asr_string_recomposition = False,
                            inplace = True
                            )
        self.feaures = self.data.compute_features(inplace = False)  # created df self.features
        self.estimate_wcpm(inplace = True)
        return self.data.get_data()

    def estimate_wcpm(self, inplace = False, model = None):
        """ take current model and estimate the wcpm with it
        """
        if model is None:
            model = self.model
        else:
            self.set_model(model_name = model)
        logging.info("Estimating wcpm")
        self.features = self.scaler.transform(self.features)
        wc = self.model.predict(self.features)
        wc = pd.Series(wc, name = 'wc_estimations')
        wcpm = wc.div(self.data[self.duration_col] /60, fill_value = 0).apply(lambda x: round(x))
        wcpm.rename('wcpm_estimations', inplace = True)
        if not inplace:
            return wcpm
        else:
            self.data.data['wcpm_estimations'] = wcpm



if __name__ == "__main__":
    df = pd.read_csv("./data/wcpm_w_dur.csv")
    d = DataGrader(df.drop(columns = 'human_transcript').loc[:20])
    d.preprocess_data(inplace = True)
    d.compute_features(inplace = True)
    d.estimate_wcpm(inplace = True)
    print(d.data.head(20))

