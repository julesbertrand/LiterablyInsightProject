import numpy as np
import numpy.testing as npt
import pandas as pd
import os

import literacy_score.data_cleaning as data_cleaning

def data_cleaning_smoke():
    data_table = pd.read_csv(os.getcwd() + "/data/test_data.csv", sep=",")
    obj = data_cleaning.data(data_table)

# def data_cleaning_ASR_string_recomposition():
#     # test ASR_string_recomposition_function
#     data_table = pd.read_csv(os.getcwd() + "/data/test_data.csv", sep=",")
#     ASR_text = """
# you probably know me already in every story you've ever been told someone like me exists
# a figure in the background barely noticed by the main players a talentless unwanted child
# the ugly one the ugly one only gets in the way she is as out of place as a sparrow in the
# clutch of swans this was the role i had in my father's hall it was the role my father
# gave me i have a memory it's smudgy almost faded into nothing now it's a memory of my
# father i can remember him picking me up in his big arms nd whirling me around until i
# shrieked with laughter that's the last the only time i can ever remember him holding
# me i don't know what changed maybe it was me i'm not like my brothers whom it must
# be said he did love a great deal i must have been a great disappointment as a king's
# daughter i could not be married off to his advantage for who would want such who
# would want to wed a creature so plain and i was a strange little girl always talking
# to things other people couldn't see running off on my own never listening to""".replace("\n", " ")
#     obj = data_cleaning.data(data_table)
#     obj.ASR_string_recomposition(col_names=["ASR_data"])
#     npt.assert_equal(obj.df["ASR_data_text"].iloc[0], ASR_text)

def data_cleaning_compare_text():
    a = "I am a data scientist testing early literacy and reading skills"
    b = "I am machine learning engineer testin early literacy and reading skill s"
    obj = data_cleaning.data(pd.DataFrame.from_dict({'a': [a], 'b': [b]}))
    npt.assert_equal(obj._compare_text(obj.df['a'].iloc[0], obj.df['b'].iloc[0]), 6)


if __name__ == "__main__":
    # data_cleaning_ASR_string_recomposition()
    data_cleaning_compare_text()
