import os

import pandas as pd

from litreading.utils.files import open_file


def test_utils_open_file():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    test_path = "/test_data/test_data.csv"
    df = open_file(dir_path + test_path, sep=",")
    df_test = pd.read_csv(dir_path + test_path)
    assert df.equals(df_test)
