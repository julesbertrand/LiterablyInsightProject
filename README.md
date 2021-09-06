# LitReadingAI
### Automated assessment of students' reading skills and literacy
This is a consulting project done as part of Insight Artificial Intelligence Fellowship.

![GitHub](https://img.shields.io/github/license/julesbertrand/litreading-insight-project)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://pypi.org/project/black/)
![CI](https://github.com/julesbertrand/litreading-insight-project/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/julesbertrand/litreading-insight-project/branch/develop/graph/badge.svg?token=ULW5SHSJSC)](https://codecov.io/gh/julesbertrand/litreading-insight-project)



## Background
We are trying to predict the true number of words correct spoken by a student reading out loud per minute (Word Correct Per Minute, WCPM), using a speech recognition-generated transcript of the student reading out loud and the original passage that the student is supposed to read.
This is a python module that can accept an ASR-generated transcript, the original text the student was supposed to read and the scored duration of the reading in order to estimate the number of words the student read correctly and therefore the number of words he can read correctly per minute.

## Table of Contents

* [Background](#background)
* [Installation](#installation)
    * [Install with models](#install-with-models)
    * [Run Streamlit app](#run-streamlit-app-for-fun)
    * [Install without models](#install-without-models)
* [Execution](#execution)
  * [Data](#data)
  * [Predict WCPM](#predict-wcpm)
  * [Train Models](#train-models)
* [Algorithm and Performance so far](#algorithm-and-performance-so-far)
* [Repo Directory structure](#repo-directory-structure)

## Installation

### Install litreading package

This will install litreading package including the model trainer and grader, and some model configs that work well, but no trained models.

```
pip install git+https://github.com/julesbertrand/litreading-insight-project
```

Please check the `litreading/config.py` file to avoid any issues, especially MODELS_PATH. Note that you will have to train models as first thing.

> If you want to have access to already trained models that work well, you can either clone the repo (see below) or download the `models/`folders from the repo.

### Installing litreading by cloning the repo

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
You will need to clone the repo as stated above:
```
git clone https://github.com/julesbertrand/litreading-insight-project
```
Then install [streamlit](https://www.streamlit.io/):
```
pip install streamlit
```
Then go to the directory and run app:
```
cd ./litreading-insight-project
streamlit run apps/app.py
```

Then, give a prompt, a interpretation and a scored duration, and it will give you a WCPM grade.


## Execution

The package is made of three main files:
- dataset.py containing the Dataset base class for data preprocessing and feature engineering. You will not need to import it in a python script.
- grade.py with DataGrader class (Dataset class inheritance) for grading with an existing model.
- train.py with ModelTrainer class (Dataset class inheritance) for training new models and hyperparameters tuning (grid search).

### Data

Your data for prediction must be a pandas DataFrame with one columns for the original text, the asr transcript and the scored duration. The default names for this columns are `'prompt'`, `'asr_transcript'`, `'scored_duration'` but they can be adapted to your data by specigying the right name for each column when you instanciate a DataGrader(). For training, you will also need a `'human_wcpm'` column for labels.

### Predict WCPM

Once your data is in the form of a DataFrame `df` with the right column headers, you can either call the function `grade_wcpm(df)` to grade with default params or instanciate `DataGrader(df)` to do preprocessing, feature computation and prediction step by step ith customized params.

```python
from litreading import grade_wcpm

df = pd.read_csv('file_to_grade')
grades = grade_wcpm(df, model_type="default")
```

```python
from litreading import Grader

df = pd.read_csv('file_to_grade')
grader = DataGrader(model_filepath='models/xgb_model.pkl')
grades = grader.grade(df)
```

Please refer to the [examples jupyter notebook](https://github.com/julesbertrand/litreading-insight-project/blob/master/tutorial_litgrade.ipynb) in the github repo for more details.

### Train Models

To train a model, you will have to instanciate a Modeltrainer. You can either choose to retrain the same model with default params (already tuned for this problem for KNN, RF and XGBoost) or try a gridsearch for hyperparameters tuning.

```python
from litreading import Model

df = pd.read_csv('file_to_grade')
trainer = Model(
  estimator=LinearRegressor(),
  scaler=StandardScaler(),
  outliers_tolerance=.2,
  baseline_mode=False,
  verbose=True
)
trainer.prepare_train_test_set(df, test_size=.2)
trainer.fit()
trainer.evaluate()
trainer.save_model(overwrite=False)
```

Please refer to the [examples jupyter notebook](https://github.com/julesbertrand/litreading-insight-project/blob/master/tutorial_litreading.ipynb) in the github repo for more details.

## Algorithm and Performance so far

The model and data pipeline can be visualized as follows:

![](resources/model_pipeline_final.png?raw=true)

The algorithm is using Difflib to make a word to word comparison between the original text and teh ASR transcript. If the original text was not fully read, then the end is deleted to ensure a fair count. Then features are computed by counting the number of words correct (same word, same place), replaced(wrong word, same place), added, or removed, as well as the number of words in each transcript and the mean and standard deviation in word length for each text. An XGBoost finetuned model is then run on this features to have an estimation of the WCPM of the students.

Currently with XGBoost, the MAE is 2.57% (2.28 in absoute) on with 3000 datapoint used for training/test. The RMSE is 16.38, to compare to an average wcpm of around 120. Here is the distribution of errors for XGB, and a scatter plot with y=estimations and x=labels=human_wcpm.
![](resources/distribution_errors_xgb.png?raw=true)
![](resources/scatter_xgb.png?raw=true)

We can see that the etsimations are very good, however more data is needed to train an accurate algorithm for big values of wcpm.

## Repo Directory structure

The package is located in the `litreading/` folder. All the modules are located in it along side:
- a configuration file `config.py` where paths to the required files and model training default config are defined.
- The `litreading/models/` folder holds the models which are used for prediction and newly trained models - make sure that the MODELS_PATH variable in `config.py` is updated and points to the location of the `litreading/models/` folder!
- The `dataset.py` file hosts the Dataset class for preprocessing and data computation
- the `grade.py` file hosts the DataGrader class for wcpm estimation
- the `train.py` files hosts the ModelTrainer class for training default model, new models, or make an sklearn gridsearch.

Unit tests are located in the `tests/` folder and for them to run properly, the variable TEST_FOLDER in config.py should point to the `tests/test_data/` folder. This gitrepo is connected to Travis.CI to run tests.

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
│   │   ├── evaluation.py
│   │   ├── files.py
│   │   ├── logging.py
│   │   ├── text.py
│   │   └── visualization.py
│   ├── __init__.py
│   ├── base.py
│   ├── config.py
│   ├── grader.py
│   ├── main.py
│   ├── model.py
│   └── preprocessor.py
├── models/
├── resources/
├── tests/
├── .flake8
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── Makefile
├── mypy.ini
├── pyproject.toml
├── README.md
├── requirements_dev.in
├── requirements_dev.txt
├── requirements.in
├── requirements.txt
├── setup.py
└── tutorial_litreading.ipynb
```
