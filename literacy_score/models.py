import logging

import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
# import gensim.downloader as api

# Logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ModelTrainer():
    def __init__(self, df, model_name):
        if model_name == 'rf':
            self.model = RandomForestRegressor
        elif model_name == 'xgb':
            self.model = XGBRegressor
        self.dataset = Dataset()

    def compute_features(self):
        self.dataset.preprocess_data(**PREPROCESSING_STEPS,
                            inplace = True
                            )
        self.features = self.dataset.compute_features(inplace = False)  # created df self.features

    def train_model(self, remove_outliers = True, test_set_size = .2):
        return

    def evaluate_model(self, visualize = True):
        return

    def save_model(self, path):
        return

    def hyperparams_tuning(self):
        return

    def plot_grid_search(self):
        return

    def feature_importance(self, visualize = True):
    






class glove_vectorizer():
    def __init__(self, model_name="glove100"):
        self.model_name = model_name
        if self.model_name == "glove100":
            self.word2vec_model_file  = './literacy_score/models/glove.6B.100d.txt.word2vec'
            self.__open_model_file()
        else:
            self.model = None
        
    def __glove2word2vec__(self):
        glove_input_file = './models/glove.6B.100d.txt'
        word2vec_output_file = 'glove.6B.100d.txt.word2vec'
        glove2word2vec(glove_input_file, word2vec_output_file)

    def __open_model_file(self):
        self.model = KeyedVectors.load_word2vec_format(self.word2vec_model_file, binary=False)
        
    def embed_errors(self, data):
        embeddings = []
        words_list = data['words'].to_list()
        for words in words_list:
#             print(words[0], words[1])
            try:
                embeddings.append(self.model.get_vector(words[0]) - self.model.get_vector(words[1]))
            except KeyError:
                embeddings.append(np.zeros(100))  # Out ot vocabulary words
        return embeddings


if __name__ == "__main__":
    pass
    

