"""
Dataset Class to open, preprocess and transform data
"""

import ast  # preprocessing ast to litteral
import difflib  # text comparison
import re  # preprocessing
import string  # preprocessing punctuation
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from num2words import num2words  # preprocessing

from litreading.config import (
    ASR_TRANSCRIPT_COL,
    DURATION_COL,
    HUMAN_TRANSCRIPT_COL,
    HUMAN_WCPM_COL,
    PROMPT_TEXT_COL,
)
from litreading.utils.files import save_file


class Dataset:
    """
    To be used as data for training and prediction
    Methods: get_data, get_features, save_data, print_rows
            preprocess_text: for preprocessing (punctuation, lowercase, num2words
            compute_differ_lists: compute comparison of asr and prompt for each row
            compute_features: calculate features (number of added, replaced, removed words, avg length, etc)
            determine_outliers_mask: in train mode, used to remove rows with huge length diff between human and asr transcripts
            compute_stats: in train mode, used to compute MAE, RMSE, etc
    Static methods: longest_common_subsequence, get_errors_dict, stats_length_of_words
    """

    def __init__(
        self,
        df: pd.DataFrame,
        prompt_col: str = PROMPT_TEXT_COL,
        asr_col: str = ASR_TRANSCRIPT_COL,
        human_col: str = HUMAN_TRANSCRIPT_COL,
        duration_col: str = DURATION_COL,
        human_wcpm_col: str = HUMAN_WCPM_COL,
        mode: str = "predict",
    ):
        self.__data_raw = df
        self._data = self.__data_raw.copy()
        # columns names
        self.prompt_col = prompt_col
        self.asr_col = asr_col
        self.human_col = human_col
        self.duration_col = duration_col
        self.human_wcpm_col = human_wcpm_col
        if mode not in ["train", "predict"]:
            raise ValueError("mode must be train or predict")
        self.mode = mode

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def features(self) -> pd.DataFrame:
        return self._features

    def save_data(self, dirpath: str, filename: str):
        """ Sav data using utils.save_file() """
        save_file(self.data, dirpath, filename, replace=False)

    def print_row(self, col_names: List[str] = None, index: int = -1):
        """ Print desired columns col_names at desired index (if -1, all data is printed)"""
        if col_names is None:
            col_names = list(self.data.columns)
        if index == -1:
            print(self.data[col_names])
        elif index >= 0:
            for col in col_names:
                print(col)
                print(self.data[col].iloc[index])
                print("\n")
        else:
            raise ValueError(f"index must be -1 or >= 0. Current: {index}")

    def preprocess_text(
        self,
        lowercase: bool = True,
        punctuation_free: bool = True,
        convert_num2words: bool = True,
        asr_string_recomposition: bool = False,
        inplace: bool = False,
    ) -> Union[None, pd.DataFrame]:
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
        self.data[columns] = df

    def compute_features(self, inplace: bool = False) -> Union[None, pd.DataFrame]:
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
            features[HUMAN_WCPM_COL] = self.data[self.human_wcpm_col]

        if not inplace:
            return features
        self._features = features

    def compute_differ_lists(self, col_1: str, col_2: str, inplace: bool = False) -> List[str]:
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
    def longest_common_subsequence(str_a: str, str_b: str, split_car: str = " "):
        """
        Return differ_list: difflib.Differ.compare() output
        Compare string a and b split by split_care
        Default split_car: " " (split by word)
        Remove text surplus at the end.
        Used in self.compute_differ_lists()
        """
        if not isinstance(str_a, str) or not isinstance(str_b, str):
            raise TypeError("Compared strings must be of string type")
        differ_list = difflib.Differ().compare(str_a.split(split_car), str_b.split(split_car))
        differ_list = list(differ_list)

        # if a lot characters at the end were added or removed from prompt
        # then delete them from differ list
        to_be_removed = differ_list[-1][0]
        if to_be_removed != " ":
            while differ_list[-1][0] == to_be_removed and len(differ_list) >= 1:
                differ_list.pop()
        return differ_list

    @staticmethod
    def get_errors_dict(differ_list: List[str]) -> Tuple[int, int, int, int, Dict[str, Any]]:
        """
        Return number of correct, added, removed, replaced words and dict of errors
        Compute the list of replaced words detected (errors_dict)
        Used in self.compute_features()
        """
        counter = 0
        errors_dict = {PROMPT_TEXT_COL: [], "transcript": []}
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
                    errors_dict[PROMPT_TEXT_COL] += [word.replace("+ ", "")]
                    errors_dict["transcript"] += [differ_list[i + j].replace("- ", "")]
                elif minus_plus:
                    errors_dict[PROMPT_TEXT_COL] += [word.replace("- ", "")]
                    errors_dict["transcript"] += [differ_list[i + j].replace("+ ", "")]
        replaced = len(errors_dict[PROMPT_TEXT_COL])
        return counter, add, sub, replaced, errors_dict

    @staticmethod
    def stats_length_of_words(s: str, sep: str = " ") -> Tuple[int, int]:
        """ Return the avg length of words in a string s, with separator sep. """
        s = s.split(sep)
        n = len(s)
        if n == 0:
            return 0
        s = [len(word) for word in s]
        mean = round(np.mean(s), 3)
        std = round(np.std(s), 3)
        return mean, std

    def determine_outliers_mask(self, tol: float = 0.2):
        """
        Return a mask with 0 for outliers and 1 for correct data
        In train mode, determine what rows have a too big difference \
between human and asr transcript length to be taken into account
        Input: tol: % of diff between len(asr_transcript) and \
len(human_transcript) above which the row is considered an outlier
        """
        if self.mode != "train":
            raise ValueError(
                "You need to be in 'train' mode to determine outliers. 'predict' was passed."
            )

        if not (0 <= tol):
            raise ValueError(f"Invalid value for tol (must be float, >= 0). Current: {tol}")

        def determine_outlier(row: int, tol: float) -> bool:
            """ Determine if row is an outlier with regards to tol """
            len_h = len(str(row[self.human_col]).split())
            len_a = len(str(row[self.asr_col]).split())
            # if diff between lengths > tol * mean of lengths
            if len_h > (1 + tol) * len_a or len_a > (1 + tol) * len_h:
                return False
            return True

        return self.data.apply(lambda x: determine_outlier(x, tol), axis=1)
