import pandas as pd  # data management format
import ast  # for literal evaluation of ASR data string --> list of dict
import string  # for 
import difflib  # string comparison
import os


class Dataset():
    def __init__(self, file_path, lowercase=True, punctuation_free=True, asr_string_recomposition=False):
        self.file_path = file_path
        self.df_raw = self.__read_data__()
        self.processed_df = self.__preprocess_data__(
            lowercase=lowercase,
            punctuation_free=punctuation_free,
            asr_string_recomposition=asr_string_recomposition
            )
        self.df = self.processed_df.copy()

    def __read_data__(self):
        path = self.file_path
        # logger.info("Loading data from %s", path)
        if not (os.path.isfile(path)):
            raise ValueError(path)
        return pd.read_csv(path)

    def get_df(self):
        return self.df
    
    def save_df(self, path):
        self.df.to_csv(path + "processed_wcpm.csv", index=False)

    def change_size_data(self, size='all'):
        """ work with only a part of data for tests purposes"""
        if size == 'all':
            self.df = self.processed_df.copy()
        elif isinstance(size, int):
            self.df = self.processed_df[:size].copy()

    def __preprocess_data__(self, lowercase=True, punctuation_free=True, asr_string_recomposition=False):
        prompt = self.df_raw['prompt']
        human_transcript = self.df_raw['human_transcript']
        asr_transcript = self.df_raw['asr_transcript']
        if asr_string_recomposition:
            # if data is stringed list of dict, get list of dict
            asr_transcript = asr_transcript.apply(lambda x: ast.literal_eval(x))
            asr_transcript = asr_transcript.apply(lambda x: " ".join([e['text'] for e in x]))
        if lowercase:
            # convert text to lowercase
            prompt = prompt.str.lower()
            human_transcript = human_transcript.str.lower()
            asr_transcript = asr_transcript.str.lower()
        if punctuation_free:
            # remove punctuation
            translater = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
            prompt = prompt.str.translate(translater)
            human_transcript = human_transcript.str.translate(translater)
            human_transcript = human_transcript.str.split().str.join(' ')
            asr_transcript = asr_transcript.str.translate(translater)
            asr_transcript = asr_transcript.str.split().str.join(' ')
        df = self.df_raw.copy()
        df['prompt'] = prompt.fillna(" ")
        df['human_transcript'] = human_transcript.fillna(" ")
        df['asr_transcript'] = asr_transcript.fillna(" ")
        return df

    def count_words_in_transcript(self, col_names = []):
        for col in col_names:
            self.df[col.split("_")[0] + "_count"] = self.df[col].apply(lambda x: len(x.split()))

    def count_words_correct(self, col_name_1, col_name_2, new_col_name = ""):
        """ apply _compare_text to two self.df columns and creates a new column in df for the number of common words
        """
        if not (isinstance(col_name_1, str) and isinstance(col_name_2, str)):
            raise TypeError("col_name_1 and col_names_2 should be strings from data columns headers")

        def compare_text(a, b, split_car = " "):
            """ comparing string a and b split by split_care, default split by word"""
            differ_list = difflib.Differ().compare(str(a).split(split_car), str(b).split(split_car))
            differ_list = list(differ_list)
            to_be_removed = differ_list[-1][0]
            if to_be_removed != " ":
                while differ_list[-1][0] == to_be_removed and len(differ_list) >= 1:
                    differ_list.pop()
            counter = 0
            error_list = []
            skip_next = False
            n = len(differ_list)
            for i, word in enumerate(differ_list):
                if skip_next:
                    skip_next = False
                    pass  # when the word has already been added to the eror list
                if word[0] == " ":
                    counter += 1  # + 1 word correct 
                elif i < n - 1:  # keep track of errors and classify them later
                    plus_minus = (word[0] == "+" and differ_list[i + 1][0] == "-")
                    minus_plus = (word[0] == "-" and differ_list[i + 1][0] == "+")
                    skip_next = plus_minus or minus_plus
                    if plus_minus:  # added word always first
                        error_list += [(word.replace("+ ", ""), differ_list[i + 1].replace("- ", ""))]
                    elif minus_plus:
                        error_list += [(differ_list[i + 1].replace("+ ", ""), word.replace("- ", ""))]
            return counter, error_list
        
        if new_col_name == "":
            new_col_name = col_name_1.split("_")[0] + "_" + col_name_2.split("_")[0]
        temp = self.df.apply(lambda x: compare_text(x[col_name_1], x[col_name_2]), axis=1)
        self.df[[new_col_name + "_wc", new_col_name + "_errors"]] = pd.DataFrame(temp.to_list(), index = self.df.index)
        self.df[new_col_name + "_wcpm"] = self.df[new_col_name + "_wc"].div(self.df["scored_duration"] / 60, fill_value = 0)            

if __name__ == "__main__":
    pass
