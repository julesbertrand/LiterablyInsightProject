import numpy as np
import pandas as pd 
import joblib

import os
import difflib

def open_file(file_path, sep = ';'):
    _, extension = file_path.rsplit(".", 1)
    if extension == 'csv':
        f = pd.read_csv(file_path, sep=sep)
    elif extension == 'pkl':
        with open(file_path, 'rb') as f:
            f = pickle.load(f)
    else:
        print("No such csv or pikle file to open")  # logger
        return
    return f


def save_file(file, path, file_name, replace=False):
    """ save file with or without replacing previous versions, in cv or pkl
    input: file: python model or df to save
            path: path to save to
            file_name: name to give to the file, including extension
            replace: False if you do not want to delete and replace previous file with same name
    """
    if path[-1] != "/":
        path += "/"
    if not os.path.exists(path):
        raise FileNotFoundError
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
        joblib.dump(file, path + ".".join((file_name, extension)), compres = 1)


def avg_length_of_words(s, sep = " "):
    """ takes a string s and gives the avg length of words in it
    """
    s = s.split(sep)
    n = len(s)
    if n == 0:
        return 0
    return sum(len(word) for word in s) / n


def compare_text(prompt, transcript, split_car = " "):
    """ compare string a and b split by split_care, default split by word, remove text surplus at the end
    """
    differ_list = difflib.Differ().compare(str(prompt).split(split_car), str(transcript).split(split_car))
    differ_list = list(differ_list)
    
    # if a lot characters at the end were added or removed from prompt
    # then delete them from differ list 
    to_be_removed = differ_list[-1][0]
    if to_be_removed != " ":
        while differ_list[-1][0] == to_be_removed and len(differ_list) >= 1:
            differ_list.pop()
    return differ_list


def get_errors_dict(differ_list):
    """ computes number of correct, added, removed, replaced words in
     the difflib differ list and computes the list of replaced words detected 
    """
    counter = 0
    errors_dict = {'prompt': [], 'transcript': []}
    skip_next = 0
    n = len(differ_list)
    add = 0
    sub = 0
    for i, word in enumerate(differ_list):
        if skip_next > 0:
            skip_next -= 1
            pass  # when the word has already been added to the error dict
        if word[0] == " ":
            counter += 1  # + 1 if word correct 
        elif i < n - 2:  # keep track of errors and classify them later
            if word[0] == "+":
                add += 1
            elif word[0] == "-":
                sub += 1
            j = 1
            while i+j < n and differ_list[i + j][0] == "?":  # account for ? in skip_next
                j += 1
            plus_minus = (word[0] == "+" and differ_list[i + j][0] == "-")
            minus_plus = (word[0] == "-" and differ_list[i + j][0] == "+")
            skip_next = (plus_minus or minus_plus) * j
            if plus_minus:
                errors_dict['prompt'] += [word.replace("+ ", "")]
                errors_dict['transcript'] += [differ_list[i + j].replace("- ", "")]
            elif minus_plus:
                errors_dict['prompt'] += [word.replace("- ", "")]
                errors_dict['transcript'] += [differ_list[i + j].replace("+ ", "")]
    replaced = len(errors_dict['prompt'])
        
    return counter, add, sub, replaced, errors_dict