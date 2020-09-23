import pandas as pd  # data management format
import ast  # for literal evaluation of ASR data string --> list of dict
import string  # for 
import difflib  # string comparison


class data():
    def __init__(self, data_table):
        self.df = data_table

    def ASR_string_recomposition(self, col_names = []):
        """ convert designtad columns from string of list of dict to list of dict with ASR algorithm output.
        create a column with the full text from the ASR.
        """
        if not isinstance(col_names, list):
            raise TypeError("col_names should be a list")
        for name in col_names:
            self.df[name] = self.df[name].apply(lambda x: ast.literal_eval(x))
            self.df[name + "_text"] = self.df[name].apply(lambda x: " ".join([e['text'] for e in x])) 

    def text_cleaning(self, col_names = []):
        """ Remove double spaces, punctuation and lowering all words to enable conprehensive comparison of texts
        """
        if not isinstance(col_names, list):
            raise TypeError("col_names should be a list")

        def remove_punctuation(text):
            text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
            return " ".join(text.split())

        for name in col_names:
            self.df[name] = self.df[name].apply(lambda x: remove_punctuation(x))
            self.df[name] = self.df[name].apply(lambda x: x.lower()) 

    def _compare_text(self, a, b, split_car = " "):
        """ Given two strings and how to separate them (new line, space, dot), 
        use difflib.Differ to give a list of common and different parts and give the number of common words among the two strings
        """
        # comparing string a and b split by split_care, default split by word
        differ_list = difflib.Differ().compare(a.split(split_car), b.split(split_car))
        counter = 0
        for word in differ_list:
            if word[0] == " ":
                counter += 1
        return counter

    def count_words_correct(self, col_name_1, col_name_2, new_col_name = "words_correct"):
        """ apply _compare_text to two self.df columns and creates a new column in df for the number of common words
        """
        if not (isinstance(col_name_1, str) and isinstance(col_name_2, str)):
            raise TypeError("col_name_1 and col_names_2 should be strings from data columns headers")
        self.df[new_col_name] = self.df.apply(lambda x: self._compare_text(x[col_name_1], x[col_name_2]), axis=1)



if __name__ == "__main__":
    import os
    data_table = pd.read_csv(os.getcwd() + "/data/test_data.csv", sep=",")
    data = data(data_table)
    data.ASR_string_recomposition(col_names=["ASR_data"])
    data.text_cleaning(col_names = ["Original_text", "Human_transcript ", "ASR_data_text"])
    data.count_words_correct("Original_text", "Human_transcript ")
    print(data.df)
