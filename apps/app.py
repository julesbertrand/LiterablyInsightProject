import pandas as pd
import streamlit as st
from loguru import logger

from litreading import grade_wcpm
from litreading.config import ASR_TRANSCRIPT_COL, DURATION_COL, PROMPT_TEXT_COL


def main():
    f = open("./resources/appheader.md", "r")
    st.markdown(f.read())
    f.close()
    prompt = st.text_area("Input the original text")
    transcript = st.text_area("Input the transcript from the audio")
    duration = st.text_input("Input the duration of the recording in seconds")

    logger.disable("litreading")

    if st.button("Grade"):

        if prompt == "" or transcript == "" or duration == "":
            st.error("Please fill all fields before grading")

        else:
            df = pd.DataFrame(
                {
                    PROMPT_TEXT_COL: [prompt],
                    ASR_TRANSCRIPT_COL: [transcript],
                    DURATION_COL: [float(duration)],
                }
            )

            grade = grade_wcpm(df, model_type="test", baseline_mode=False)

            st.write(f"Your wcpm is {grade[0]}")


if __name__ == "__main__":
    main()
