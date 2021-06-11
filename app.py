import pandas as pd
import streamlit as st

from litreading.grade import grade_wcpm


def main():
    f = open("./resources/appheader.md", "r")
    st.markdown(f.read())
    f.close()
    prompt = st.text_area("Input the original text")
    transcript = st.text_area("Input the transcript from the audio")
    duration = st.text_input("Input the duration of the recording in seconds")
    if st.button("Grade"):
        if prompt == "" or transcript == "" or duration == "":
            st.error("Please fill all fields before grading")
        else:
            df = pd.DataFrame(
                {
                    "prompt": [prompt],
                    "asr_transcript": [transcript],
                    "scored_duration": [float(duration)],
                }
            )
            grade = grade_wcpm(df)
            st.write("Your wcpm is %i" % grade["wcpm_estimation"])


if __name__ == "__main__":
    main()
