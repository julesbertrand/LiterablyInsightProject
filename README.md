# LitReadingAI 
### Automated assessment of students' reading skills and literacy
This is a consulting project done as part of Insight Artificial Intelligence Fellowship.

![GitHub](https://img.shields.io/github/license/julesbertrand/litreading-insight-project)
![Travis (.com)](https://img.shields.io/travis/com/julesbertrand/litreading-insight-project?label=TravisCI)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

## Background
We are trying to predict the true number of words correct spoken by a student reading out loud per minute, using a speech recognition-generated transcript of the student reading out loud and the original passage that the student is supposed to read.
This is a python module that can accept an ASR-generated transcript, the original text the student was supposed to read and the scored duration of the reading in order to estimate the number of words the student read correctly and therefore the number of words he can read correctly per minute.


## Installation

### Install with models
1. Clone the repo:
```
git clone https://github.com/julesbertrand/litreading-insight-project
cd ./litreading/insight/project
```
2. update the config file
Please go to your package directory and check if the paths variables are ok (especially MODELS_PATH).  
You can also modify the preprocessing steps you want to run by default on your data. In that case, you may want to retrain the model to be sure to have great results. Note that you can still decide what preprocessing steps you want to apply by specifying it when running the model.

3. Install as package
To make it accessible as a python package, Run in terminal / Bash
```
cd ./litreading/insight/project
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
streamlit run app.py
```

### Install without models

This is not the preferred way of installing the package. In terminal run 
```
pip install git+https://github.com/julesbertrand/litreading-insight-project
```
Please check the config file paths to avoid any issues, especially MODELS_PATH. Note that you will have to train models as first thing.

## Execution

The package is made of three main files: 
- dataset.py containing the Dataset base class for data preprocessing and feature engineering. You will not need to import it in a python script.
- grade.py with DataGrader class (Dataset class inheritance) for grading with an existing model.
- train.py with ModelTrainer class (Dataset class inheritance) for training new models and hyperparameters tuning (grid search).

### Data

Your data for prediction must be a pandas DataFrame with one columns for the original text, the asr transcript and the scored duration. The default names for this columns are `'prompt'`, `'asr_transcript'`, `'scored_duration'` but they can be adapted to your data by specigying the right name for each column when you instanciate a DataGrader(). For training, you will also need a `'human_wcpm'` column for labels.

### Prediction

Once your data is in the form of a DataFrame `df` with the right column headers, you can either call the function `grade_wcpm(df)` to grade with default params or instanciate `DataGrader(df)` to do preprocessing, feature computation and prediction step by step ith customized params. 

```python
from litreading.grade import grade_wcpm

df = pd.read_csv('file_to_grade')
grades = grade_wcpm(df)
```

```python
from litreading.grade import DataGrader

df = pd.read_csv('file_to_grade')
grader = DataGrader(df, model_type='XGB')
grades = grader.grade_wcpm(df)
```

Please refer to the [examples jupyter notebook](https://github.com/julesbertrand/litreading-insight-project/blob/master/tutorial_litgrade.ipynb) in the github repo for more details.

### Training models

To train a model, you will have to instanciate a Modeltrainer. You can either choose to retrain the same model with default params (already tuned for this problem for KNN, RF and XGBoost) or try a gridsearch for hyperparameters tuning.

```python
from litreading.train import ModelTrainer

df = pd.read_csv('file_to_grade')
trainer = ModelTrainer(df, model_type="XGB")
trainer.prepare_train_test_set(remove_outliers=True,
                               outliers_tol=.15,
                               test_set_size=.2,
                               inplace=True
                              )
trainer.train()
trainer.evaluate_model()
trainer.save_model(replace=False)
```

Please refer to the [examples jupyter notebook](https://github.com/julesbertrand/litreading-insight-project/blob/master/tutorial_litgrade.ipynb) in the github repo for more details.

## Algorithm and performance so far

The algorithm is using Difflib to make a word to word comparison between the original text and teh ASR transcript. If the original text was not fully read, then the end is deleted to ensure a fair count. Then features are computed by counting the number of words correct (same word, same place), replaced(wrong word, same place), added, or removed, as well as the number of words in each transcript and the mean and standard deviation in word length for each text. An XGBoost finetuned model is then run on this features to have an estimation of the WCPM of the students.

Currently with XGBoost, the MAE is 2.57% (2.28 in absoute) on with 3000 datapoint used for training/test. The RMSE is 16.38, to compare to an average wcpm of around 120. Here is the distrobution of errors for XGB, and a scatter plot with y=estimations and x=labels=human_wcpm. 
![](resources/distribution_errors_xgb.png?raw=true)
![](resources/scatter_xgb.png?raw=true)

We can see that the etsimations are very good, however more data is needed to train an accurate algorithm for big values of wcpm.

## Repo Directory structure
The package is located in the `litreading/` folder. All the modules are located in it along side a configuration file `config.py` where paths to the required files and model training default config are defined. The `litreading/models/` folder holds the models which are used for prediction and newly trained models - make sure that the MODELS_PATH variable in `config.py` is updated and points to the location of the `litreading/models/` folder! The `dataset.py` file hosts the Dataset class for preprocessing and data computation, the `grade.py` file hosts the DataGrader class for wcpm estimation, and the `train.py` files hosts the ModelTrainer class for training default model, new models, or make an sklearn gridsearch.  

Unit tests are located in the `tests/` folder and for them to run properly, the variable TEST_FOLDER in config.py should point to the `tests/test_data/` folder. This gitrepo is connected to Travis.CI to run tests.

```
.
├── litreading
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── grade.py
│   ├── train.py
│   ├── utils.py
│   └── models
│       ├── StandardScaler.joblib
│       ├── RF.joblib
│       ├── XGB.joblib
│       └── KNN.joblib
├── tests
│   ├── test_utils.py
│   ├── test_dataset.py
│   ├── test_grade.py
│   ├── test_train.py
│   └── test_data
├── resources
│   ├── appheader.md
│   ├── distribution_errors_xgb.png
│   └── scatter_xgb.png
├── .gitignore
├── .travis.yml
├── LICENSE  
├── README.md
├── requirements.txt
├── setup.py
└── tutorial_litreading.ipynb 
```