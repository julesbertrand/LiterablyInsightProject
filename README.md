# LitReadingAI
### Automated assessment of students' reading skills and literacy
This is a consulting project done as part of Insight Artificial Intelligence Fellowship.

![GitHub](https://img.shields.io/github/license/julesbertrand/litreading-insight-project)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://pypi.org/project/black/)
![CI](https://github.com/julesbertrand/litreading-insight-project/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/julesbertrand/litreading-insight-project/branch/develop/graph/badge.svg?token=ULW5SHSJSC)](https://codecov.io/gh/julesbertrand/litreading-insight-project)

---

**Source Code**: <a href="https://github.com/julesbertrand/litreading-insight-project" target="_blank">https://github.com/julesbertrand/litreading-insight-project</a>

---

Litreading aims at predicting the actual number of correct words spoken by a student reading aloud per minute (Word Correct Per Minute, WCPM), using a speech-recognition-generated transcript of the student reading aloud and the original passage the student is supposed to read.
This is a python module that can accept a speech recognition-generated transcript, the original text that the student was supposed to read and the recorded duration of the reading in order to estimate the number of words that the student has read correctly and thus the number of words that he or she can read correctly per minute.

The key Features are:
* **trainer**: a trainer class that will preprocess your data, and then pass it to a scikit-learn pipeline including a scaler and a Machine Learning model (linear regression, random forest, boosting, ...).
* **grader**: a grader class to use an already existing pipeline and make predictions.
* **preprocessor**: a preprocessor class to clean text data and useful compute numerical features from it.

---

## Table of Contents
* [Installation](#installation)
    * [Install litreading](#install-litreading)
    * [Install litreading with trained models](#install-litreading-with-trained-models)
    * [Run Streamlit app](#run-streamlit-app-for-fun)
* [Quickstart](#quickstart)
   * [Data](#data)
   * [Command line](#command-line)
   * [In python script](#in-python-script)
* [Algorithm and Performance so far](#algorithm-and-performance-so-far)
* [Repo Directory structure](#repo-directory-structure)

## Installation

### Install litreading

```
pip install git+https://github.com/julesbertrand/litreading-insight-project
```

Please check the `litreading/config.py` file to avoid any issues. Note that you will have to train models as first thing.

> If you want to have access to already trained models that work well, you can either clone the repo (see below) or download the `models/`folders from the repo.

### Install litreading with trained models

This will install the code, the models, and some other dev files. It also allows you ti have acess to everything needed to contribute to the project.

1. Clone the repo:
```
git clone https://github.com/julesbertrand/litreading-insight-project
cd ./litreading-insight-project
```

2. update the config file
Please go to your package directory and check if the paths variables are ok (especially MODELS_PATH).
You can also modify the preprocessing steps you want to run by default on your data. In that case, you may want to retrain the model to be sure to have great results. This can modify features or make them incompatible with sklearn model.

3. Install as package
To make it accessible as a python package, Run in terminal
```
pip install .
```

### Run streamlit app (for fun)
You will need to clone the repo as stated above. Then install [streamlit](https://www.streamlit.io/), go to the directory and run the app::
```
pip install streamlit
cd ./litreading-insight-project
streamlit run apps/app.py
```

Then, give a passage, a interpretation and a scored duration, and it will give you a WCPM grade.


## Quickstart


### Data

Your data for prediction must be a pandas DataFrame with one columns for the original text, the asr transcript and the scored duration. The default names for these columns are `'prompt'`, `'asr_transcript'`, `'scored_duration'`. They can be adapted to your data by specifying the right name for each column in `litreading/config.py`. For training, you will also need a `'human_wcpm'` column for labels.

### Command line

Litreading is available in cli using python. There are 3 main commands: grade, train and gridsearch.

To grade you need to have a model: you can either download a pre-trained model from the repo or train your own Pipeline. Then, type this in a terminal:
```bash
python -m litreading grade mydir/mymodel.pkl data/my_data.csv
```

### In python script

Once your data is in the form of a DataFrame `df` with the right column headers, you can either call the function `grade_wcpm()` to grade with default params or instanciate `Grader()` to do preprocessing, feature computation and prediction step by step ith customized params.

```python
from litreading import grade_wcpm

df = pd.read_csv('file_to_grade.csv')
grades = grade_wcpm(df, model_type="default")
```

```python
from litreading import Grader

df = pd.read_csv('file_to_grade.csv')
g = Grader(model_filepath="models/model_test.pkl")
grades = g.grade(df)
```

To train a model, you will have to instanciate a Modeltrainer. You can either choose to retrain the same model with default params (already tuned for this problem for KNN, RF and XGBoost) or try a gridsearch for hyperparameters tuning.

```python
from sklearn.ensemble import RandomForestRegressor
from litreading import ModelTrainer
from litreading.config import DEFAULT_MODEL_SCALER

df = pd.read_csv('file_to_grade.csv')
m = ModelTrainer(
    estimator=RandomForestRegressor(),
    scaler=DEFAULT_MODEL_SCALER(),
    baseline_mode=False,  # default to False
    verbose=True,
)
m.prepare_train_test_set(df, test_size=.2)
m = m.fit()
metrics = m.evaluate()
m.save_model(f"models/model_rf.pkl", overwrite=False)
```

Please refer to the [examples jupyter notebook](https://github.com/julesbertrand/litreading-insight-project/blob/master/tutorial_litgrade.ipynb) in the github repo for more details.

## Algorithm and Performance so far

The model and data pipeline can be visualized as follows:

![](resources/model_pipeline_final.png?raw=true)

The algorithm is using Difflib to make a word to word comparison between the original text and teh ASR transcript. If the original text was not fully read, then the end is deleted to ensure a fair count. Then features are computed by counting the number of words correct (same word, same place), replaced(wrong word, same place), added, or removed, as well as the number of words in each transcript and the mean and standard deviation in word length for each text. An XGBoost finetuned model is then run on this features to have an estimation of the WCPM of the students.

Currently with XGBoost, the MAE is 2.86% on with 3000 datapoint used for training/test. The RMSE is 3.9, to compare to an average wcpm of around 120. Here is the distribution of errors for XGB, and a scatter plot with y=estimations and x=labels=human_wcpm.
![](resources/scatter_xgb.png?raw=true)

We can see that the etsimations are very good, however more data is needed to train an accurate algorithm for big values of wcpm.

## Repo Directory structure

The package is located in the `litreading/` folder. All the modules are located in it along side:
- a configuration file `config.py` where paths to the required files and model training default config are defined.
- The `litreading/models/` folder holds the models which are used for prediction and newly trained models - make sure that the MODELS_PATH variable in `config.py` is updated and points to the location of the `litreading/models/` folder!
- the `grader.py` file hosts the Grader class for wcpm estimation
- the `trainer.py` file hosts the ModelTrainer class for training default model, new models, or run a gridsearch.

Unit tests are located in the `tests/` folder and for them to run properly, the variable TEST_FOLDER in config.py should point to the `tests/samples/` folder.

The streamlit app code is located in `app.py`.

A tutorial for this package can be found and downloaded in the form of a jupyter notebook: [`tutorial_litreading.ipynb`](https://github.com/julesbertrand/litreading-insight-project/blob/master/tutorial_litreading.ipynb).

```
.
├── .github
│   └── workflows
│       ├── ci.yml
│       └── codecov.yml
├── apps
│   └── app.py
├── bin
│   └── install.sh
├── data/
├── litreading
│   ├── utils
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── evaluation.py
│   │   ├── files.py
│   │   ├── logging.py
│   │   ├── text.py
│   │   └── visualization.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── base.py
│   ├── config.py
│   ├── grader.py
│   ├── main.py
│   ├── preprocessor.py
│   └── trainer.py
├── models/
├── resources/
├── tests/
├── .flake8
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
├── requirements_dev.in
├── requirements_dev.txt
├── requirements.in
├── requirements.txt
├── setup.py
└── tutorial_litreading.ipynb
```
