import os  # for parent path
import pandas as pd  # data management format
import ast  # for literal evaluation of ASR data string --> list of dict
import string
import difflib  # string comparison


class data():
    def __init__(self, data_table):
        self.df = data_table

    def ASR_string_recomposition(self, col_names = []):
        if not isinstance(col_names, list):
            raise TypeError("col_names should be a list")
        for name in col_names:
            self.df[name] = self.df[name].apply(lambda x: ast.literal_eval(x))
            self.df[name + "_text"] = self.df[name].apply(lambda x: " ".join([e['text'] for e in x])) 

    def text_cleaning(self, col_names = []):
        if not isinstance(col_names, list):
            raise TypeError("col_names should be a list")

        def remove_punctuation(text):
            text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
            return " ".join(text.split())

        for name in col_names:
            self.df[name] = self.df[name].apply(lambda x: remove_punctuation(x))
            self.df[name] = self.df[name].apply(lambda x: x.lower()) 

    def _compare_text(self, a, b, split_car = " "):
        # comparing string a and b split by split_care, default split by word
        differ_list = difflib.Differ().compare(a.split(split_car), b.split(split_car))
        counter = 0
        for word in differ_list:
            if word[0] == " ":
                counter += 1
        return counter

    def count_words_correct(self, col_name_1, col_name_2, new_col_name = "words_correct"):
        if not (isinstance(col_name_1, str) and isinstance(col_name_2, str)):
            raise TypeError("col_name_1 and col_names_2 should be strings from data columns headers")
        self.df[new_col_name] = self.df.apply(lambda x: self._compare_text(x[col_name_1], x[col_name_2]), axis = 1)



if __name__ == "__main__":
    data_table = pd.read_csv(os.getcwd() + "/data/data.csv", sep=",")
    data = data(data_table)
    data.ASR_string_recomposition(col_names=["ASR_data"])
    data.text_cleaning(col_names = ["Original_text", "Human_transcript ", "ASR_data_text"])
    data.count_words_correct("Original_text", "Human_transcript ")
    print(data.df)
