import numpy as np
import pandas as pd  # data management format
import ast  # for literal evaluation of ASR data string --> list of dict
import string
import difflib  # string comparison
import os
import jellyfish  # for phonetic transcriptions
import pickle  # save model 
import re  # df.apply regex
from num2words import num2words


def save_to_file(file, path, replace=False):
    path, file_name = path.rsplit("/", 1)
    path += "/"
    file_name, extension = file_name.split(".")
    if replace:
        try:
            os.remove(file_name)
        except OSError: pass
    else:
        i = 0
        while os.path.exists(path + ".".join((file_name + '_{:d}'.format(i), extension))):
            i += 1
        file_name += '_{:d}'.format(i)
    if extension == 'csv':
        file.to_csv(path + ".".join((file_name, extension)), index=False, sep=';', encoding='utf-8')
    else:
        pickle.dump(file, path + ".".join((file_name, extension)), compress = 1)


class Dataset():
    def __init__(self, file_path, lowercase=True, punctuation_free=True, asr_string_recomposition=False, convert_num2words=True):
        self.file_path = file_path
        self.df_raw = self.__read_data__()
        self.processed_df = self.__preprocess_data__(
            lowercase=lowercase,
            punctuation_free=punctuation_free,
            asr_string_recomposition=asr_string_recomposition,
            convert_num2words = convert_num2words
            )
        self.df = self.processed_df.copy()

    def __read_data__(self):
        path = self.file_path
        # logger.info("Loading data from %s", path)
        if not (os.path.isfile(path)):
            raise ValueError(path)
        return pd.read_csv(path)
    
    def __preprocess_data__(self,
                            lowercase=True,
                            punctuation_free=True,
                            asr_string_recomposition=False,
                            convert_num2words=True
                           ):
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
        if convert_num2words:
            def converter(s):
                if len(s) == 4:
                    return re.sub('\d+', lambda y: num2words(y.group(), to='year'), s)
                return re.sub('\d+', lambda y: num2words(y.group()), s)
            prompt = prompt.apply(lambda x: re.sub('\d+', lambda y: converter(y.group()), x))
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

    def get_df(self):
        return self.df
    
    def save_df(self, path):
        save_to_file(self.df, path)

    def change_data_size(self, size='all'):
        """ work with only a part of data for tests purposes"""
        if size == 'all':
            self.df = self.processed_df.copy()
        elif isinstance(size, int):
            self.df = self.processed_df[:size].copy()
            
    def print_row(self, col_names=[], index = -1):
        if len(col_names) == 0:
            col_names = self.df.columns
        if index != -1:
            for col in col_names:
                print(col)
                print(self.df[col].iloc[index])
                print("\n")
        else:
            print(self.df[col_names]) 
    
    def determine_outliers(self, tol = .2):
        def determine_outlier(human_transcript, asr_transcript, tol):
            len_h = len(human_transcript.split())
            len_a = len(asr_transcript.split()) 
            # if diff between lengths > tol * mean of lengths
            if len_h > (1+tol) * len_a or len_a > (1+tol) * len_h:
                return False
            return True
        return self.df.apply(lambda x: determine_outlier(x['human_transcript'], x['asr_transcript'], tol), axis=1)
    
    def compute_wcpm_with_details(self, prompt_col, transcript_col):
        new_col_name = prompt_col.split("_")[0] + "_" + transcript_col.split("_")[0]
        inplace = True
        # create differ list with difflib
        self.compute_differ_lists(prompt_col, transcript_col, new_col_name = new_col_name, inplace = inplace)
        # word count and dict of detected errors
        self.compute_word_count(new_col_name + "_differ_list", new_col_name = new_col_name, inplace = inplace)
        # phonetic comparison of error words
#         self.phonetic_comparison_of_errors(errors_col = new_col_name + "_errors", inplace = inplace)
        # add the current wc and correct word from phonetics
        self.df[new_col_name + "_wc_final"] = self.df[new_col_name + "_wc"] #+ self.df[new_col_name + "_phonetic_count"]
        # get wcpm
        self.compute_wcpm_from_wc(wc_col = new_col_name + "_wc_final", inplace = inplace)
        
    def count_words_in_transcript(self, col_names = []):
        for col in col_names:
            self.df[col.split("_")[0] + "_count"] = self.df[col].apply(lambda x: len(x.split()))
            
    def compute_differ_lists(self, prompt_col, transcript_col, new_col_name="", inplace=False):
        """ apply _compare_text to two self.df columns and creates a new column in df for the number of common words
        """
        if not (isinstance(prompt_col, str) and isinstance(transcript_col, str)):
            raise TypeError("prompt_col and transcript_col should be strings from data columns headers")

        def compare_text(prompt, transcript, split_car = " "):
            """ compare string a and b split by split_care, default split by word, remove text surplus at the end
            """
            differ_list = difflib.Differ().compare(str(prompt).split(split_car), str(transcript).split(split_car))
            differ_list = list(differ_list)
            
            to_be_removed = differ_list[-1][0]
            if to_be_removed != " ":
                while differ_list[-1][0] == to_be_removed and len(differ_list) >= 1:
                    differ_list.pop()
            return differ_list
        
        if new_col_name == "":
            new_col_name = prompt_col.split("_")[0] + "_" + transcript_col.split("_")[0]
        temp = self.df.apply(lambda x: compare_text(x[prompt_col], x[transcript_col]), axis=1)
        if not inplace:
            return temp
        self.df[new_col_name + "_differ_list"] = temp
        
    def compute_word_count(self, differ_list_col, new_col_name = "", inplace = False):
        """ Uses differ_list column in dataframe to compute word_count and erro_dict
        """
        def get_errors_dict(differ_list):
            counter = 0
            errors_dict = {'prompt': [], 'transcript': []}
            skip_next = 0
            n = len(differ_list)
            for i, word in enumerate(differ_list):
                if skip_next > 0:
                    skip_next -= 1
                    pass  # when the word has already been added to the eror list
                if word[0] == " ":
                    counter += 1  # + 1 word correct 
                elif i < n - 2:  # keep track of errors and classify them later
                    j = 1
                    while differ_list[i + j][0] == "?":
                        j += 1
                    plus_minus = (word[0] == "+" and differ_list[i + j][0] == "-")
                    minus_plus = (word[0] == "-" and differ_list[i + j][0] == "+")
                    skip_next = (plus_minus or minus_plus) * j
                    if plus_minus:  # added word always first
                        errors_dict['prompt'] += [word.replace("+ ", "")]
                        errors_dict['transcript'] += [differ_list[i + j].replace("- ", "")]
                    elif minus_plus:
                        errors_dict['prompt'] += [word.replace("- ", "")]
                        errors_dict['transcript'] += [differ_list[i + j].replace("+ ", "")]
            return counter, errors_dict
        
        if new_col_name == "":
            new_col_name = differ_list_col.replace("_differ_list", "")
        temp = self.df[differ_list_col].apply(lambda x: get_errors_dict(x))
        temp = pd.DataFrame(temp.to_list(), columns = [new_col_name + "_wc", new_col_name + "_errors"], index = self.df.index)
        if not inplace:
            return temp
        self.df = pd.concat([self.df, temp], axis=1)
        
    def compute_wcpm_from_wc(self, wc_col, inplace = False):
        new_col_name = wc_col.replace("_wc", "_wcpm")
        temp = self.df[wc_col].div(self.df["scored_duration"] / 60, fill_value = 0)
        if not inplace:
            return temp.rename(new_col_name)
        self.df[new_col_name] = temp
        
    def compute_wc_from_wcpm(self, wcpm_col, inplace=False):
        new_col_name = wcpm_col.replace("_wcpm", "_wc")
        temp = self.df[wcpm_col].mul(self.df["scored_duration"] / 60, fill_value = 0)
        if not inplace:
            return temp.rename(new_col_name)
        self.df[new_col_name] = temp
        
    def phonetic_comparison_of_errors(self, errors_col, inplace = False):
        """ Perform phonetic comparison of errors to know if they are to be counted as a corretc word for the student"""
        def phonetic_comparison(errors_dict):
            counter = 0
            for i in range(len(errors_dict['prompt'])):
                if jellyfish.match_rating_comparison(errors_dict['prompt'][i], errors_dict['transcript'][i]):
                    counter += 1
            return counter
        new_col_name = errors_col.replace('_errors', '_phonetic_count')
        temp = self.df[errors_col].apply(lambda x: phonetic_comparison(x))
        if not inplace:
            return temp.rename(new_col_name)
        else:
            self.df[new_col_name] = temp
            
    def label_errors(self, save = False):
        words = []
        labels = []
        idxs = []
        for i in self.df.index:
            prompt_words_asr = self.df['prompt_asr_errors'].loc[i]['prompt']
            if len(prompt_words_asr) == 0:
                pass
            prompt_words_human = self.df['prompt_human_errors'].loc[i]['prompt']
            asr_errors = self.df['prompt_asr_errors'].loc[i]['transcript']
            for j, word in enumerate(prompt_words_asr):
                words.append([word, asr_errors[j].replace("+ ", "").replace("- ", "")])
                idxs.append(i)
                if word in prompt_words_human:  # if the error was detected by the human --> mistake by student
                    labels.append(False)
                else:
                    labels.append(True)  # mistake by asr
        labeled_errors = pd.DataFrame({'indexes': idxs, 'words': words, 'labels': labels})
        if save:
            path = "./data/labeled_data.csv"
            save_to_file(labeled_errors, path, replace=False)
        return labeled_errors

if __name__ == "__main__":
    pass
