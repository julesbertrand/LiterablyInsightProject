import numpy as np
import pandas as pd
import os
import errno

import joblib

import ast  # preprocessing ast to litteral
import re  # preprocessing
from num2words import num2words  # preprocessing 
import string # punctuation in preprocessing

from grading_utils import open_file, save_file, compare_text, get_errors_dict, avg_length_of_words

# Logging
# logger = logging.getLogger()
# handler = logging.StreamHandler()
# formatter = logging.Formatter(
#         '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.ERROR)

def grade_wcpm(df):
    data = DataGrader(df)
    return data.grade_wcpm()

class DataGrader():
    # def __init__(self, data_path):
    #     self.data_raw = self.__load_data(data_path)
    def __init__(self, df):
        self.data_raw = df
        self.data = self.data_raw.copy()
        self.prompt_col = 'prompt'
        self.asr_col = 'asr_transcript'
        self.duration_col = 'scored_duration'

        self.scaler = self.__load_model('standard_scaler.joblib')
        self.model = self.__load_model('rf_hypertuned.pkl')
        self.model_name = "random_forest"

    def __load_model(self, model_name):
        with open("./literacy_score/models/" + model_name, 'rb') as f:
            model = joblib.load(f)  
        return model

    # def __load_data(self, data_path, sep = ';'):
    #     if not os.path.isfile(data_path):
    #         raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_path)
    #     if data_path.split(".")[-1] != 'csv':
    #         print("This is not a csv file", sep = sep)
    #         return
    #     return pd.read_csv(data_path)

    def set_model(self, model_name):
        if self.model_name == model_name:
            pass
        else:
            self.model_name = model_name
            if model_name == "random_forest":
                self.model = self.__load_model('rf_hypertuned.pkl')
            elif model_name == "xgboost":
                self.model = self.__load_model('xgb_hypertuned.pkl')

    def grade_wcpm(self):
        self.preprocess_data(lowercase = True,
                            punctuation_free = True,
                            convert_num2words = True,
                            asr_string_recomposition = False,
                            inplace = True
                            )
        self.compute_features(inplace = True)  # created df self.features
        self.estimate_wcpm(inplace = True)
        return self.data

    def preprocess_data(self,
                        lowercase = True,
                        punctuation_free = True,
                        convert_num2words = True,
                        asr_string_recomposition = False,
                        inplace = False
                        ):
        prompt = self.data[self.prompt_col]
        asr_transcript = self.data[self.asr_col]
        if asr_string_recomposition:
            # if data is string-ed list of dict, get list of dict
            asr_transcript = asr_transcript.apply(lambda x: ast.literal_eval(x))
            asr_transcript = asr_transcript.apply(lambda x: " ".join([e['text'] for e in x]))
        if lowercase:
            # convert text to lowercase
            prompt = prompt.str.lower()
            asr_transcript = asr_transcript.str.lower()
        if convert_num2words:
            def converter(s):
                if len(s) == 4:
                    return re.sub('\d+', lambda y: num2words(y.group(), to='year'), s)
                return re.sub('\d+', lambda y: num2words(y.group()), s)
            prompt = prompt.apply(lambda x: re.sub('\d+', lambda y: converter(y.group()), x))
        if punctuation_free:
            # remove punctuation
            translater = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            prompt = prompt.str.translate(translater)
            prompt = prompt.str.split().str.join(' ') 
            asr_transcript = asr_transcript.str.translate(translater)
            asr_transcript = asr_transcript.str.split().str.join(' ')            
        df = self.data.copy()
        df[self.prompt_col] = prompt.fillna(" ")
        df[self.asr_col] = asr_transcript.fillna(" ")

        if not inplace:
            df = self.data.copy()
            df[self.prompt_col] = prompt.fillna(" ")
            df[self.asr_col] = asr_transcript.fillna(" ")
            return df
        else:
            self.data[self.prompt_col] = prompt.fillna(" ")
            self.data[self.asr_col] = asr_transcript.fillna(" ")

    def compute_differ_list(self, col_1, col_2, inplace = False):
        """ apply _compare_text to two self.df columns 
        and creates a new column in df for the number of common words
        """
        if not (isinstance(col_1, str) and isinstance(col_2, str)):
            raise TypeError("col_1 and col_2 should be strings from data columns headers")
        temp = self.data.apply(lambda x: compare_text(x[col_1], x[col_2]), axis=1)

        if not inplace:
            return pd.Series(temp, name = 'differ_list')
        else:
            self.data['differ_list'] = temp

    def compute_features(self, inplace = False):
        diff_list = self.compute_differ_list(col_1 = self.prompt_col,
                                            col_2 = self.asr_col,
                                            inplace = False
                                            )
        temp = diff_list.apply(lambda x: get_errors_dict(x))
        temp = pd.DataFrame(temp.to_list(), columns = ["correct_words",
                                                        "added_words",
                                                        "removed_words",
                                                        "replaced_words",
                                                        "errors_dict"
                                                    ])
        temp.drop(columns = ['errors_dict'], inplace = True)
        temp['asr_word_count'] = self.data[self.asr_col].apply(lambda x: len(x.split()))
        temp['prompt_avg_word_length'] = self.data[self.prompt_col].apply(lambda x: avg_length_of_words(x))
        temp['asr_avg_word_length'] = self.data[self.asr_col].apply(lambda x: avg_length_of_words(x))
        # temp['human_wc'] = self.data['human_wcpm'].mul(self.data['scored_duration'] / 60, fill_value = 0)
        if not inplace:
            return temp
        else:
            self.features = temp

    def estimate_wcpm(self, inplace = False, model = None):
        if model is None:
            model = self.model
        else:
            self.set_model(model_name = model)
        self.features = self.scaler.transform(self.features)
        wc = self.model.predict(self.features)
        wc = pd.Series(wc, name = 'wc_estimations')
        wcpm = wc.div(self.data[self.duration_col] /60, fill_value = 0).apply(lambda x: round(x))
        wcpm.rename('wcpm_estimations', inplace = True)
        if not inplace:
            return wcpm
        else:
            self.data['wcpm_estimations'] = wcpm



if __name__ == "__main__":
    df = pd.read_csv("./data/wcpm_w_dur.csv")
    d = DataGrader(df.drop(columns = 'human_transcript').loc[:20])
    d.preprocess_data(inplace = True)
    d.compute_features(inplace = True)
    d.estimate_wcpm(inplace = True)
    print(d.data.head(20))

