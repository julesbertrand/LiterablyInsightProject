import os

import pandas as pd

from litreading.grade import DataGrader, grade_wcpm

# logger.setLevel(logging.CRITICAL)

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = "/test_data/test_data.csv"
DF_TEST = pd.read_csv(ABS_PATH + TEST_PATH)


def test_grade_datagrader_class():
    for model_type in ["Baseline", "RF", "KNN", "XGB"]:
        DataGrader(DF_TEST, model_type=model_type)


def test_grade_set_model():
    obj = DataGrader(DF_TEST, model_type="XGB")
    obj.set_model("Baseline")
    assert obj.model.__class__.__name__ == "BaselineModel"


def test_grade_estimate_wcpm():
    obj = DataGrader(DF_TEST, model_type="XGB")
    features = pd.read_csv(ABS_PATH + "/test_data/test_data_features.csv", sep=";")
    obj.features = features
    estimations = obj.estimate_wcpm(inplace=False)
    estimations_test = pd.read_csv(ABS_PATH + "/test_data/test_data_wcpm_estimations.csv", sep=";")
    estimations.equals(estimations_test["wcpm_estimation"])


def test_grade_grade_wcpm():
    grades = grade_wcpm(DF_TEST, only_wcpm=True)
    grades_test = pd.read_csv(ABS_PATH + "/test_data/test_data_wcpm_estimations.csv", sep=";")
    grades.equals(grades_test["wcpm_estimation"])
