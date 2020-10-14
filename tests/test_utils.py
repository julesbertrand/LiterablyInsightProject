import numpy as np
import pandas as pd 
import os
import pytest

from litreading import BaselineModel, open_file

def test_utils_Baseline():
    obj = BaselineModel()

def test_utils_open_file():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    test_path = '/test_data/test_data.csv'
    df = open_file(dir_path + test_path, sep=',')
    df_test = pd.read_csv(dir_path + test_path)
    assert df.equals(df_test)
