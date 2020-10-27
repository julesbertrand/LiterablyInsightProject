import pandas as pd
import os
import pytest
import logging
import ast

from litreading.dataset import Dataset
from litreading.utils import logger

logger.setLevel(logging.CRITICAL)

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = "/test_data/test_data.csv"
DF_TEST = pd.read_csv(ABS_PATH + TEST_PATH)


def test_dataset_class():
    obj = Dataset(DF_TEST)


def test_dataset_get_data():
    obj = Dataset(DF_TEST)
    assert DF_TEST.equals(obj.get_data())


def test_dataset_preprocess_data():
    obj = Dataset(DF_TEST)
    df = obj.preprocess_data(
        lowercase=True,
        punctuation_free=True,
        convert_num2words=True,
        asr_string_recomposition=False,
        inplace=False,
    )
    df_test = pd.read_csv(ABS_PATH + "/test_data/test_data_preprocessed.csv", sep=";")
    print(df_test)
    assert df_test.equals(df[["prompt", "asr_transcript"]])


def test_dataset_LCS_1():
    obj = Dataset(DF_TEST)
    string_a = "I am testing this algorithm and I hope it will no raise an error biip"
    string_b = "I am testin this algorithm and I hopit will no raise an aerror"
    result = [
        "  I",
        "  am",
        "- testing",
        "?       -\n",
        "+ testin",
        "  this",
        "  algorithm",
        "  and",
        "  I",
        "+ hopit",
        "- hope",
        "- it",
        "  will",
        "  no",
        "  raise",
        "  an",
        "- error",
        "+ aerror",
        "? +\n",
    ]
    assert result == obj.longest_common_subsequence(string_a, string_b, split_car=" ")


def test_dataset_LCS_2():
    obj = Dataset(DF_TEST)
    string_a = "I am testing this algorithm and I will no raise an aerror biip"
    string_b = ""
    result = ["+ "]
    assert result == obj.longest_common_subsequence(string_a, string_b, split_car=" ")


def test_dataset_differ_list():
    obj = Dataset(DF_TEST)
    obj.preprocess_data(inplace=True)
    df = obj.compute_differ_lists(col_1="prompt", col_2="asr_transcript", inplace=False)
    df_test = pd.read_csv(ABS_PATH + "/test_data/test_differ_lists.csv", sep=";")
    df_test = df_test.applymap(ast.literal_eval)
    assert df_test["differ_list"].equals(df)


def test_dataset_features():
    obj = Dataset(DF_TEST)
    obj.preprocess_data(inplace=True)
    features = obj.compute_features(inplace=False)
    features_test = pd.read_csv(ABS_PATH + "/test_data/test_data_features.csv")
    pytest.approx(features.to_numpy(), features_test.to_numpy())


def test_dataset_outliers_mask_predict_mode():
    obj = Dataset(DF_TEST, mode="predict")
    assert None is obj.determine_outliers_mask(tol=0.2)


def test_dataset_outliers_mask_train_mode():
    obj = Dataset(DF_TEST, mode="train")
    outliers = obj.determine_outliers_mask(tol=0.2)
    result = [True] * 49
    result[14], result[22] = False, False
    assert outliers.to_list() == result


def test_dataset_compute_stats_predict_mode():
    obj = Dataset(DF_TEST, mode="predict")
    Y_pred = DF_TEST["human_wcpm"] + 5
    assert None is obj.compute_stats(Y_pred, test_idx=DF_TEST.index)


def test_dataset_compute_stats_train_mode():
    obj = Dataset(DF_TEST, mode="train")
    Y_pred = DF_TEST["human_wcpm"] + 5
    stats = obj.compute_stats(Y_pred, test_idx=DF_TEST.index)
    stats_test = pd.read_csv(ABS_PATH + "/test_data/test_data_stats.csv")
    pytest.approx(stats.to_numpy(), stats_test.to_numpy())
