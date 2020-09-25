import pandas as pd  # data management format
import ast  # for literal evaluation of ASR data string --> list of dict
import string  # for 
import difflib  # string comparison
import os


class Dataset():
    def __init__(self, file_path, lowercase=True, punctuation_free=True, asr_string_recomposition=False):
        self.file_path = file_path
        self.df_raw = self.__read_data__()
        self.df = self.__preprocess_data__(
            lowercase=lowercase,
            punctuation_free=punctuation_free,
            asr_string_recomposition=asr_string_recomposition
            )

    def __read_data__(self):
        path = self.file_path
        # logger.info("Loading data from %s", path)
        if not (os.path.isfile(path)):
            raise ValueError(path)
        return pd.read_csv(path)

    def get_df(self):
        return self.df

    def use_small_df(self, n):
        self.small_df = self.df.iloc[:n]

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
        df['prompt'] = prompt
        df['human_transcript'] = human_transcript
        df['asr_transcript'] = asr_transcript
        return df

    def count_words_correct(self, col_name_1, col_name_2, new_col_name = "words_correct"):
        """ apply _compare_text to two self.df columns and creates a new column in df for the number of common words
        """
        if not (isinstance(col_name_1, str) and isinstance(col_name_2, str)):
            raise TypeError("col_name_1 and col_names_2 should be strings from data columns headers")

        def compare_text(a, b, split_car = " "):
            # comparing string a and b split by split_care, default split by word
            differ_list = difflib.Differ().compare(str(a).split(split_car), str(b).split(split_car))
            counter = 0
            for word in differ_list:
                if word[0] == " ":
                    counter += 1
            return counter
        self.df[new_col_name] = self.df.apply(lambda x: compare_text(x[col_name_1], x[col_name_2]), axis=1)



if __name__ == "__main__":
    pass
