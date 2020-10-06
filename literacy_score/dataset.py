import numpy as np
import pandas as pd

import joblib

import ast  # preprocessing ast to litteral
import re  # preprocessing
from num2words import num2words  # preprocessing 
import string  # preprocessing punctuation
import difflib  # text comparison

from literacy_score.config import DATA_PATH, MODELS_PATH
from literacy_score.utils import logger, save_file, open_file, BaselineModel

class Dataset():
    def __init__(self,
                df,
                prompt_col = 'prompt',
                asr_col = 'asr_transcript',
                human_col = 'human_transcript',
                duration_col = 'scored_duration',
                human_wcpm_col = 'human_wcpm',
                mode = 'predict'
                ):
        self.data_raw = df
        self.data = self.data_raw.copy()
        # columns names
        self.prompt_col = prompt_col
        self.asr_col = asr_col
        self.human_col = human_col
        self.duration_col = duration_col
        self.human_wcpm_col = human_wcpm_col
        # mode can be train or predict, used for labeled handling when features are computed
        self.mode = mode

    def get_data(self):
        return self.data

    def get_features(self):
        return self.features

    def save_data(self, filename, path = DATA_PATH):
        save_file(self.data, path, filename, replace = False)

    def print_row(self, col_names=[], index = -1):
        if len(col_names) == 0:
            col_names = self.data.columns
        if index != -1:
            for col in col_names:
                print(col)
                print(self.data[col].iloc[index])
                print("\n")
        else:
            print(self.data[col_names])

    def preprocess_data(self,
                        lowercase = True,
                        punctuation_free = True,
                        convert_num2words = True,
                        asr_string_recomposition = False,
                        inplace = False
                        ):
        """ Preprocessing data to make it standard and comparable
        """
        columns = [self.prompt_col, self.asr_col]
        if self.mode == 'train':
            columns.append(self.human_col)
        df = self.data[columns].copy()
        if asr_string_recomposition:
            logger.debug("Recomposing ASR string from dict")
            df = df.applymap(lambda x: ast.literal_eval(x))
            df = df.applymap(lambda x: " ".join([e['text'] for e in x]))
        if lowercase:
            logger.debug("Converting df to lowercase")
            df = df.applymap(lambda x: str(x).lower())
        if convert_num2words:
            logger.debug("Converting numbers to words")
            def converter(s):
                if len(s) == 4:
                    return re.sub('\d+', lambda y: num2words(y.group(), to='year'), s)
                return re.sub('\d+', lambda y: num2words(y.group()), s)
            df = df.applymap(converter)
        if punctuation_free:
            # remove punctuation
            logger.debug("Removing punctuation")
            t = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            def remove_punctuation(s, translater):
                s = s.translate(translater)
                return str(" ".join(s.split()))
            df.applymap(lambda x: remove_punctuation(x, t))
        df.fillna(" ", inplace = True)

        if not inplace:
            return df
        else:
            self.data[columns] = df

    @staticmethod
    def compare_text(string_a, string_b, split_car = " "):
        """ compare string a and b split by split_care, default split by word, remove text surplus at the end
        Used in self.compute_differ_list()
        """
        differ_list = difflib.Differ().compare(str(string_a).split(split_car), str(string_b).split(split_car))
        differ_list = list(differ_list)
        
        # if a lot characters at the end were added or removed from prompt
        # then delete them from differ list 
        to_be_removed = differ_list[-1][0]
        if to_be_removed != " ":
            while differ_list[-1][0] == to_be_removed and len(differ_list) >= 1:
                differ_list.pop()
        return differ_list

    def compute_differ_list(self, col_1, col_2, inplace = False):
        """ apply _compare_text to two self.df columns 
        and creates a new column in df for the number of common words
        """
        logger.debug("Comparing %s to %s", col_1, col_2)
        if not (isinstance(col_1, str) and isinstance(col_2, str)):
            logger.error("col_1 and col_2 should be strings from data columns headers")
        temp = self.data.apply(lambda x: self.compare_text(x[col_1], x[col_2]), axis=1)

        if not inplace:
            return pd.Series(temp, name = 'differ_list')
        else:
            self.data['differ_list'] = temp

    @staticmethod
    def get_errors_dict(differ_list):
        """ computes number of correct, added, removed, replaced words in
        the difflib differ list and computes the list of replaced words detected 
        Used in self.compute_features()
        """
        counter = 0
        errors_dict = {'prompt': [], 'transcript': []}
        skip_next = 0
        n = len(differ_list)
        add = 0
        sub = 0
        for i, word in enumerate(differ_list):
            if skip_next > 0:
                skip_next -= 1
                pass  # when the word has already been added to the error dict
            if word[0] == " ":
                counter += 1  # + 1 if word correct 
            elif i < n - 2:  # keep track of errors and classify them later
                if word[0] == "+":
                    add += 1
                elif word[0] == "-":
                    sub += 1
                j = 1
                while i+j < n and differ_list[i + j][0] == "?":  # account for ? in skip_next
                    j += 1
                plus_minus = (word[0] == "+" and differ_list[i + j][0] == "-")
                minus_plus = (word[0] == "-" and differ_list[i + j][0] == "+")
                skip_next = (plus_minus or minus_plus) * j
                if plus_minus:
                    errors_dict['prompt'] += [word.replace("+ ", "")]
                    errors_dict['transcript'] += [differ_list[i + j].replace("- ", "")]
                elif minus_plus:
                    errors_dict['prompt'] += [word.replace("- ", "")]
                    errors_dict['transcript'] += [differ_list[i + j].replace("+ ", "")]
        replaced = len(errors_dict['prompt'])
        return counter, add, sub, replaced, errors_dict

    @staticmethod
    def avg_length_of_words(s, sep = " "):
        """ takes a string s and gives the avg length of words in it
        """
        s = s.split(sep)
        n = len(s)
        if n == 0:
            return 0
        return sum(len(word) for word in s) / n

    def compute_features(self, inplace = False):
        """ compute differ list with difflib, then count words and add feautres for wcpm estimation
        """
        diff_list = self.compute_differ_list(col_1 = self.prompt_col,
                                            col_2 = self.asr_col,
                                            inplace = False
                                            )
        logger.debug("Computing features")
        temp = diff_list.apply(lambda x: self.get_errors_dict(x))
        temp = pd.DataFrame(temp.to_list(), columns = ["correct_words",
                                                        "added_words",
                                                        "removed_words",
                                                        "replaced_words",
                                                        "errors_dict"
                                                    ])
        temp.drop(columns = ['errors_dict'], inplace = True)
        temp['asr_word_count'] = self.data[self.asr_col].apply(lambda x: len(x.split()))
        temp['prompt_avg_word_length'] = self.data[self.prompt_col].apply(lambda x: self.avg_length_of_words(x))
        temp['asr_avg_word_length'] = self.data[self.asr_col].apply(lambda x: self.avg_length_of_words(x))
        if self.mode == 'train':
            temp['human_wc'] = self.data['human_wcpm'].mul(self.data[self.duration_col] / 60, fill_value = 0)
        self.features = temp
        if not inplace:
            return self.features

    def determine_outliers_mask(self, tol = .2):
        """ For train mode, determine what rows have a too big difference 
        between human and asr transcript length to be taken into account
        Input: tol: % of diff between len(asr_transcript) adn len(human_transcript) \
                    above which the row is considered an outlier
        """
        if self.mode != 'train':
            logger.error("You need to be in 'train' mode to determine outliers. 'predict' was passed.")
            return
        def determine_outlier(row, tol):
            len_h = len(str(row[self.human_col]).split())
            len_a = len(str(row[self.asr_col]).split()) 
            # if diff between lengths > tol * mean of lengths
            if len_h > (1+tol) * len_a or len_a > (1+tol) * len_h:
                return False
            return True
        return self.data.apply(lambda x: determine_outlier(x, tol), axis=1)

    def compute_stats(self, Y_pred, test_idx):
        """ Computes statistics about the wcpm estimation for train mode
        input: asr_wc_estimation: prediction on test set of words correct 
                        test_idx: index of test set in all data 
        """
        if self.mode != 'train':
            logger.error("You need to be in 'train' mode to compute statistics about the wcpm estimation.\
                 'predict was passed")
            return
        stats = pd.DataFrame(Y_pred, columns = ['asr_wc_estimation'], index = test_idx)
        stats['human_wcpm'] = self.data[self.human_wcpm_col].loc[test_idx]
        stats['wcpm_estimation'] = stats['asr_wc_estimation'].div(self.data[self.duration_col].loc[test_idx] / 60, fill_value = 0)
        stats['wcpm_estimation_error'] = stats['human_wcpm'] - stats['wcpm_estimation']
        stats['wcpm_estimation_abs_error'] = stats['wcpm_estimation_error'].abs()
        stats['wcpm_estimation_error_%'] = np.where(stats['human_wcpm'] != 0,
                                                        stats['wcpm_estimation_error'] / stats['human_wcpm'],
                                                        0
                                                        ) 
        stats['wcpm_estimation_abs_error_%'] = stats['wcpm_estimation_error_%'].abs()
        return stats


if __name__ == "__main__":
    df = pd.read_csv("./data/wcpm_w_dur.csv")
    d = Dataset(df)
    d.preprocess_data(inplace = True)
    d.compute_features(inplace = True)
    print(d.determine_outliers_mask(tol = .1))
    print(d.data.head(6))