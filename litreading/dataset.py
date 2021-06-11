"""
Dataset Class to open, preprocess and transform data
"""

import ast  # preprocessing ast to litteral
import difflib  # text comparison
import re  # preprocessing
import string  # preprocessing punctuation

import numpy as np
import pandas as pd
from num2words import num2words  # preprocessing

from litreading.utils import logger, save_file


class Dataset:
    """
    To be used as data for training and prediction
    Methods: get_data, get_features, save_data, print_rows
            preprocess_data: for preprocessing (punctuation, lowercase, num2words
            compute_differ_lists: compute comparison of asr and prompt for each row
            compute_features: calculate features (number of added, replaced, removed words, avg length, etc)
            determine_outliers_mask: in train mode, used to remove rows with huge length diff between human and asr transcripts
            compute_stats: in train mode, used to compute MAE, RMSE, etc
    Static methods: longest_common_subsequence, get_errors_dict, stats_length_of_words
    """

    def __init__(
        self,
        df,
        prompt_col="prompt",
        asr_col="asr_transcript",
        human_col="human_transcript",
        duration_col="scored_duration",
        human_wcpm_col="human_wcpm",
        mode="predict",
    ):
        self.data_raw = df
        self.data = self.data_raw.copy()
        # columns names
        self.prompt_col = prompt_col
        self.asr_col = asr_col
        self.human_col = human_col
        self.duration_col = duration_col
        self.human_wcpm_col = human_wcpm_col
        # mode can be train or predict, influence on how certain methods will be used
        self.mode = mode

    def get_data(self):
        return self.data

    def get_features(self):
        return self.features

    def save_data(self, filename, path):
        """ Sav data using utils.save_file() """
        save_file(self.data, path, filename, replace=False)

    def print_row(self, col_names=[], index=-1):
        """ Print desired columns col_names at desired index (if -1, all data is printed)"""
        if len(col_names) == 0:
            col_names = self.data.columns
        if index != -1:
            for col in col_names:
                print(col)
                print(self.data[col].iloc[index])
                print("\n")
        else:
            print(self.data[col_names])

    def preprocess_data(
        self,
        lowercase=True,
        punctuation_free=True,
        convert_num2words=True,
        asr_string_recomposition=False,
        inplace=False,
    ):
        """
        Preprocessing data to make it standard and comparable
        If not inplace, return pd.DataFrame with preprocessed data
        4 possible actions:
            lowercase, default True
            remove punctuation, default True
            convert numbers to words (2 -> 'two'), default True
            asr_string_recomposition: if asr is given as list of string, recompose it, default False
        inplace: if inplace, will not return anything, default False
        """
        columns = [self.prompt_col, self.asr_col]
        if self.mode == "train":
            columns.append(self.human_col)
        df = self.data[columns].copy()
        if asr_string_recomposition:
            logger.debug("Recomposing ASR string from dict")
            df = df.applymap(ast.literal_eval)
            df = df.applymap(lambda x: " ".join([e["text"] for e in x]))
        if lowercase:
            logger.debug("Converting df to lowercase")
            df = df.applymap(lambda x: str(x).lower())
        if convert_num2words:
            logger.debug("Converting numbers to words")

            def converter(s):
                if len(s) == 4:
                    return re.sub("\d+", lambda y: num2words(y.group(), to="year"), s)  # noqa
                return re.sub("\d+", lambda y: num2words(y.group()), s)  # noqa

            df = df.applymap(converter)
        if punctuation_free:
            # remove punctuation
            logger.debug("Removing punctuation")
            t = str.maketrans(string.punctuation, " " * len(string.punctuation))

            def remove_punctuation(s, translater):
                s = s.translate(translater)
                return str(" ".join(s.split()))

            df.applymap(lambda x: remove_punctuation(x, t))
        df.fillna(" ", inplace=True)

        if not inplace:
            return df
        else:
            self.data[columns] = df

    @staticmethod
    def longest_common_subsequence(string_a, string_b, split_car=" "):
        """
        Return differ_list: difflib.Differ.compare() output
        Compare string a and b split by split_care
        Default split_car: " " (split by word)
        Remove text surplus at the end.
        Used in self.compute_differ_lists()
        """
        differ_list = difflib.Differ().compare(
            str(string_a).split(split_car), str(string_b).split(split_car)
        )
        differ_list = list(differ_list)

        # if a lot characters at the end were added or removed from prompt
        # then delete them from differ list
        to_be_removed = differ_list[-1][0]
        if to_be_removed != " ":
            while differ_list[-1][0] == to_be_removed and len(differ_list) >= 1:
                differ_list.pop()
        return differ_list

    def compute_differ_lists(self, col_1, col_2, inplace=False):
        """
        Return pandas series of differ_lists
        Apply self.longest_common_subsequence() to two self.df columns
        Create a new column in df for the number of common words.
        """
        logger.debug("Comparing %s to %s", col_1, col_2)
        if not (isinstance(col_1, str) and isinstance(col_2, str)):
            logger.error("col_1 and col_2 should be strings from data columns headers")
        temp = self.data.apply(
            lambda x: self.longest_common_subsequence(x[col_1], x[col_2]), axis=1
        )
        if not inplace:
            return pd.Series(temp, name="differ_list")
        else:
            self.data["differ_list"] = temp

    @staticmethod
    def get_errors_dict(differ_list):
        """
        Return number of correct, added, removed, replaced words and dict of errors
        Compute the list of replaced words detected (errors_dict)
        Used in self.compute_features()
        """
        counter = 0
        errors_dict = {"prompt": [], "transcript": []}
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
            elif i < n - 2:  # keep track of mistakes
                if word[0] == "+":
                    add += 1
                elif word[0] == "-":
                    sub += 1
                j = 1
                while i + j < n and differ_list[i + j][0] == "?":  # account for ? in skip_next
                    j += 1
                # two cases for replaced words: + - or - +
                plus_minus = word[0] == "+" and differ_list[i + j][0] == "-"
                minus_plus = word[0] == "-" and differ_list[i + j][0] == "+"
                skip_next = (plus_minus or minus_plus) * j
                if plus_minus:
                    errors_dict["prompt"] += [word.replace("+ ", "")]
                    errors_dict["transcript"] += [differ_list[i + j].replace("- ", "")]
                elif minus_plus:
                    errors_dict["prompt"] += [word.replace("- ", "")]
                    errors_dict["transcript"] += [differ_list[i + j].replace("+ ", "")]
        replaced = len(errors_dict["prompt"])
        return counter, add, sub, replaced, errors_dict

    @staticmethod
    def stats_length_of_words(s, sep=" "):
        """ Return the avg length of words in a string s, with separator sep. """
        s = s.split(sep)
        n = len(s)
        if n == 0:
            return 0
        s = [len(word) for word in s]
        mean = round(np.mean(s), 3)
        std = round(np.std(s), 3)
        return mean, std

    def compute_features(self, inplace=False):
        """
        if not inplace, return features DataFrame
        Use longest_common_subsequence with selfcompute_differ_lists()
        Count words using self.get_errors_dict()
        Compute avg and std of lengt of words
        Divide by scored_duration column to get 'per minute' figure
        """
        diff_list = self.compute_differ_lists(
            col_1=self.prompt_col, col_2=self.asr_col, inplace=False
        )
        logger.debug("Computing features")
        temp = diff_list.apply(self.get_errors_dict)
        features = pd.DataFrame(
            temp.to_list(),
            columns=[
                "correct_words",
                "added_words",
                "removed_words",
                "replaced_words",
                "errors_dict",
            ],
        )
        features.drop(columns=["errors_dict"], inplace=True)
        features["asr_word_count"] = self.data[self.asr_col].apply(lambda x: len(x.split()))
        features = features.div(self.data[self.duration_col] / 60, axis=0)
        features = features.add_suffix("_pm")
        temp_prompt = self.data[self.prompt_col].apply(self.stats_length_of_words)
        temp_prompt = pd.DataFrame(
            temp_prompt.to_list(),
            columns=["prompt_avg_word_length", "prompt_std_word_length"],
        )
        temp_asr = self.data[self.asr_col].apply(self.stats_length_of_words)
        temp_asr = pd.DataFrame(
            temp_asr.to_list(), columns=["asr_avg_word_length", "asr_std_word_length"]
        )
        features = pd.concat([features, temp_prompt, temp_asr], axis=1)
        if self.mode == "train":
            features["human_wcpm"] = self.data[self.human_wcpm_col]
        self.features = features
        if not inplace:
            return self.features

    def determine_outliers_mask(self, tol=0.2):
        """
        Return a mask with 0 for outliers and 1 for correct data
        In train mode, determine what rows have a too big difference \
between human and asr transcript length to be taken into account
        Input: tol: % of diff between len(asr_transcript) and \
len(human_transcript) above which the row is considered an outlier
        """
        if self.mode != "train":
            logger.error(
                "You need to be in 'train' mode to determine outliers. 'predict' was passed."
            )
            return

        def determine_outlier(row, tol):
            """ Determine if row is an outlier with regards to tol """
            len_h = len(str(row[self.human_col]).split())
            len_a = len(str(row[self.asr_col]).split())
            # if diff between lengths > tol * mean of lengths
            if len_h > (1 + tol) * len_a or len_a > (1 + tol) * len_h:
                return False
            return True

        return self.data.apply(lambda x: determine_outlier(x, tol), axis=1)

    def compute_stats(self, Y_pred, test_idx):
        """
        Return a DataFrame of statistics about the wcpm estimation for train mode.
        Input: Y_pred: prediction on test set of words correct
                test_idx: index of test set in all data
        """
        if self.mode != "train":
            logger.error(
                "You need to be in 'train' mode to compute statistics about the wcpm estimation.\
                 'predict was passed"
            )
            return
        stats = pd.DataFrame(Y_pred, columns=["wcpm_estimation"], index=test_idx)
        stats["human_wcpm"] = self.data[self.human_wcpm_col].loc[test_idx]
        stats["wcpm_estimation_error"] = stats["human_wcpm"] - stats["wcpm_estimation"]
        stats["wcpm_estimation_abs_error"] = stats["wcpm_estimation_error"].abs()
        stats["wcpm_estimation_error_%"] = np.where(
            stats["human_wcpm"] != 0,
            stats["wcpm_estimation_error"] / stats["human_wcpm"],
            0,
        )
        stats["wcpm_estimation_abs_error_%"] = stats["wcpm_estimation_error_%"].abs()
        stats["RMSE"] = stats["wcpm_estimation_error"] ** 2
        return stats
