import ast
import difflib
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import num2words
import numpy as np
import pandas as pd

from litreading.utils import logger

# from sklearn.base import TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import FunctionTransformer


class LCSPreprocessor:
    """[summary]

    Raises:
        TypeError: [description]
        TypeError: [description]
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """

    # default_lowercase = True
    # default_remove_punctuation = True
    # default_num2_words = True
    # default_asr_recomp = False

    def __init__(
        self,
        asr_string_recomposition: bool = False,
        to_lowercase: bool = True,
        remove_punctuation: bool = True,
        convert_num2words: bool = True,
        outlier_detector_type: Optional[str] = None,
        outlier_detector_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.preprocesssing_steps = {
            "asr_string_recomposition": asr_string_recomposition,
            "to_lowercase": to_lowercase,
            "remove_punctuation": remove_punctuation,
            "convert_num2words": convert_num2words,
        }
        self.outlier_detector_type = outlier_detector_type
        self.outlier_detector_params = outlier_detector_params
        self._init_outlier_detector()
        self._init_steps()

    def _init_outlier_detector(self):
        self._outlier_detector = None
        if self.outlier_detector_type is not None:
            self._outlier_detector = OutlierDetector(
                self.outlier_detector_type, self.outlier_detector_params
            )

    def _init_steps(self) -> None:
        self._steps = []
        for k, v in self.preprocesssing_steps.items():
            if v:
                self._steps.append(k)
        self._steps.append("compute_numerical_features")
        if self._outlier_detector is not None:
            self._steps.append("remove_outliers")

    def _init_preprocessing_pipeline(self):
        raise NotImplementedError

    def _compute_step_msg(self) -> str:
        step_no, step_name = next(self.__steps_iter)
        msg = f"[Preprocessing] (step {step_no + 1} of {len(self._steps)}): {step_name}"
        return msg

    def preprocess_data(
        self,
        df: pd.DataFrame,
        prompt_col: str = "prompt",
        asr_transcript_col: str = "asr_transcript",
        human_transcript_col: str = "human_transcript",
        duration_col: str = "scored_duration",
    ):
        self.__steps_iter = iter(enumerate(self._steps))
        data_ = df.copy()
        # text_cols = {
        #     "prompt_col": prompt_col,
        #     "asr_transcript_col": asr_transcript_col,
        #     "human_transcript_col": human_transcript_col,
        # }
        text_cols = [prompt_col, asr_transcript_col, human_transcript_col]
        data_[text_cols] = self.preprocess_text(data_[text_cols], **self.preprocesssing_steps)
        features = self.compute_numerical_features(
            data_, prompt_col, asr_transcript_col, duration_col
        )

        if self._outlier_detector is not None:
            features = self._remove_outliers(features)

        return features

    def preprocess_text(
        self,
        data,
        to_lowercase: bool = True,
        remove_punctuation: bool = True,
        convert_num2words: bool = True,
        asr_string_recomposition: bool = False,
    ) -> pd.DataFrame:
        if asr_string_recomposition:
            logger.info(self._compute_step_msg())
            data = data.applymap(recompose_asr_string_from_dict)

        if to_lowercase:
            logger.info(self._compute_step_msg())
            data = data.applymap(lambda x: str(x).lower())

        if convert_num2words:
            logger.info(self._compute_step_msg())
            data = data.applymap(numbers_to_literals)

        if remove_punctuation:
            logger.info(self._compute_step_msg())
            data = data.applymap(remove_punctuation_from_string)

        data = data.fillna(" ")
        return data

    def compute_numerical_features(
        self, data: pd.DataFrame, prompt_col: str, asr_transcript_col: str, duration_col: str
    ) -> pd.DataFrame:
        logger.info(self._compute_step_msg())
        diff_list_df = self.compute_differ_lists(data, col_1=prompt_col, col_2=asr_transcript_col)
        words_count = diff_list_df.apply(lambda x: pd.Series(self.get_words_count(x)))

        features = pd.DataFrame.from_dict(words_count).add_suffix("_words")
        features["asr_word_count"] = data[asr_transcript_col].apply(lambda x: len(x.split()))

        features = features.div(data[duration_col] / 60, axis=0)
        features = features.add_suffix("_pm")

        for col in [prompt_col, asr_transcript_col]:
            words_length_stats = data[col].apply(
                lambda x: pd.Series(self.get_words_length_stats(x))
            )
            words_length_stats.columns = [f"{col}_word_length_avg", f"{col}_word_length_std"]
            features = pd.concat([features, words_length_stats], axis=1)

        return features

    def compute_differ_lists(self, data: pd.DataFrame, col_1: str, col_2: str) -> pd.DataFrame:
        if not (isinstance(col_1, str) and isinstance(col_2, str)):
            raise TypeError("col_1 and col_2 should be strings from data columns headers")

        logger.debug("Computing differences list for %s v. %s", col_1, col_2)
        differ_list_df = data.apply(
            lambda x: self.longest_common_subsequence(x[col_1], x[col_2]), axis=1
        )
        return differ_list_df

    @staticmethod
    def longest_common_subsequence(str_a: str, str_b: str, split_car: str = " "):
        if not isinstance(str_a, str) or not isinstance(str_b, str):
            raise TypeError("Compared strings must be of string type")

        differ_list = difflib.Differ().compare(str_a.split(split_car), str_b.split(split_car))
        differ_list = list(differ_list)

        # if a lot of characters at the end were added or removed from prompt
        # then delete them from differ list
        to_be_removed = differ_list[-1][0]
        if to_be_removed != " ":
            while differ_list[-1][0] == to_be_removed and len(differ_list) >= 1:
                differ_list.pop()

        return differ_list

    @staticmethod
    def get_words_count(differ_list: List[str]) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """
        Return number of correct, added, removed, replaced words and dict of errors
        Compute the list of replaced words detected (errors_dict)
        Used in self.compute_features()
        """
        correct = 0
        errors_dict = {"prompt": [], "transcript": []}
        skip_next = 0
        n = len(differ_list)
        added = 0
        removed = 0
        for i, word in enumerate(differ_list):
            if skip_next > 0:
                skip_next -= 1
                pass  # when the word has already been added to the error dict
            if word[0] == " ":
                correct += 1  # + 1 if word correct
            elif i < n - 2:  # keep track of mistakes
                if word[0] == "+":
                    added += 1
                elif word[0] == "-":
                    removed += 1
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

        words_count = {
            "correct": correct,
            "added": added,
            "removed": removed,
            "replaced": len(errors_dict["prompt"]),
        }
        return words_count  # , errors_dict

    @staticmethod
    def get_words_length_stats(s: str, sep: str = " ") -> Tuple[int, int]:
        """ Return the avg length of words in a string s, with separator sep. """
        s = s.split(sep)
        if len(s) == 0:
            return 0
        s = [len(word) for word in s]
        mean = np.mean(s)
        std = np.std(s)
        return mean, std

    def _remove_outliers(self):
        logger.info(self._compute_step_msg())
        raise NotImplementedError


def numbers_to_literals(s: str) -> str:
    if len(s) == 4:
        return re.sub("\d+", lambda y: num2words.num2words(y.group(), to="year"), s)  # noqa
    return re.sub("\d+", lambda y: num2words.num2words(y.group()), s)  # noqa


def recompose_asr_string_from_dict(s: str) -> str:
    s = ast.literal_eval(s)
    s = " ".join([e["text"] for e in s])
    return s


def remove_punctuation_from_string(s: str) -> str:
    # return re.sub(r" +", " ", s.translate(translater))
    return re.sub(r"[^\w\s]", "", s)


class OutlierDetector:
    """[summary]"""

    def __init__(
        self, detector_type=Literal["default", "localof"], params: Dict[str, Any] = None
    ) -> None:
        if detector_type == "default":
            raise NotImplementedError
        elif detector_type == "localof":
            raise NotImplementedError
        else:
            raise ValueError
