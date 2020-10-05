import streamlit as st
import numpy as np 
import difflib  # string comparison

from grade import grade

def compare_text(a, b, split_car = " "):
    """ comparing string a and b split by split_care, default split by word"""
    differ_list = difflib.Differ().compare(str(a).split(split_car), str(b).split(split_car))
    differ_list = list(differ_list)
    
    to_be_removed = differ_list[-1][0]
    if to_be_removed != " ":
        while differ_list[-1][0] == to_be_removed and len(differ_list) >= 1:
            differ_list.pop()
    return differ_list

def get_error_list(differ_list):
    counter = 0
    error_dict = {'prompt': [], 'transcript': []}
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
                error_dict['prompt'] += [word.replace("+ ", "")]
                error_dict['transcript'] += [differ_list[i + j].replace("- ", "")]
            elif minus_plus:
                error_dict['prompt'] += [word.replace("- ", "")]
                error_dict['transcript'] += [differ_list[i + j].replace("+ ", "")]
    return counter, error_dict

def main():
    f = open("./literacy_score/resources/header.md", 'r')
    header = st.markdown(f.read())
    f.close()
    prompt = st.text_area("Input the original text")
    transcript = st.text_area("Input the transcript from the audio")
    duration = st.text_input("Input the duration of the recording in seconds")

    grade = grade()
    if st.button("Grade"):
        st.write("Your wcpm is %i" % counter)


if __name__ == "__main__":
    main()

