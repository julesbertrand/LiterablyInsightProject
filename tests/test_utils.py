import os

import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression

from litreading.utils.cli import Instanciator
from litreading.utils.files import open_file


def test_utils_open_file():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    test_path = "/samples/test_data.csv"
    df = open_file(dir_path + test_path, sep=",")
    df_test = pd.read_csv(dir_path + test_path)
    assert df.equals(df_test)


def test_instanciator_init():
    available_objects = dict(default=LinearRegression)
    Instanciator(available_objects, "test object")


@pytest.mark.parametrize("parameters", [None, {}, {"normalize": True}])
def test_instanciator_instanciate(parameters):
    available_objects = dict(default=LinearRegression)
    Instanciator(available_objects, "test object").instanciate("default", parameters)


def test_instanciator_instanciate_error():
    available_objects = dict(default=LinearRegression)
    i = Instanciator(available_objects, "test object")
    with pytest.raises(ValueError):
        i.instanciate("key_not_in_dict")
