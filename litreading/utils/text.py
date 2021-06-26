import ast
import re

import num2words


def numbers_to_literals(s: str) -> str:
    if len(s) == 4:
        return re.sub("\d+", lambda y: num2words.num2words(y.group(), to="year"), s)  # noqa
    return re.sub("\d+", lambda y: num2words.num2words(y.group()), s)  # noqa


def recompose_asr_string_from_dict(s: str) -> str:
    s = ast.literal_eval(s)
    s = " ".join([e["text"] for e in s])
    return s


def remove_punctuation_from_string(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s)
