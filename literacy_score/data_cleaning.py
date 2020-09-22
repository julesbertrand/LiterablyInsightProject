import os  # for parent path
import pandas as pd  # data management format
import ast  # for literal evaluation of ASR data string --> list of dict
import string
import difflib  # string comparison


class data():
    def __init__(self, data_table):
        self.data_table = data_table

    def ASR_string_recomposition(self, col_names = []):
        if not isinstance(col_names, list):
            raise TypeError("col_names should be a list")
        for name in col_names:
            self.data_table[name] = data_table[name].apply(lambda x: ast.literal_eval(x))
            self.data_table[name + "_text"] = data_table[name].apply(lambda x: " ".join([e['text'] for e in x])) 
        return data_table


if __name__ == "__main__":
    data_table = pd.read_csv(os.getcwd() + "/data/data.csv", sep=",")
    data = data(data_table)
    data.ASR_string_recomposition(col_names=["ASR_data"])
