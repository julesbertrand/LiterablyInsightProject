import ast
import re

import num2words


def numbers_to_literals(s: str) -> str:
    """convert all numbers to litterals"""
    if len(s) == 4:
        return re.sub(r"\d+", lambda y: num2words.num2words(y.group(), to="year"), s)  # noqa
    return re.sub(r"\d+", lambda y: num2words.num2words(y.group()), s)  # noqa


def recompose_asr_string_from_dict(s: str) -> str:
    """recompose asr string from dict with words"""
    s = ast.literal_eval(s)
    s = " ".join([e["text"] for e in s])
    return s


def remove_punctuation_from_string(s: str) -> str:
    """reomve punctuation from string"""
    return re.sub(r"[^\w\s]", "", s)
