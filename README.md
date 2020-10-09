# LitReadingAI 
### Automated assessment of students' reading skills and literacy
This is a consulting project done as part of Insight Artificial Intelligence Fellowship.

## Background

## Installation

### Install package
In terminal run 
```
pip install git+https://github.com/julesbertrand/litreading-insight-project
```

### Update config file
Please go to your package directory (something like `appdata\local\programs\python\python37\lib\site-packages`) and check if the paths variables are ok (especially MODELS_PATH).
You can also modify the preprocessing steps you want to run on your data. In that case, you may want to retrain the model to be sure to have great results.

### Run streamlit app (for fun)

## Execution

from litreading.grade import grade_wcpm, DataGrader
from litreading.train import ModelTrainer

### Data

Your data for prediction must be a pandas DataFrame with one columns for the original text, the asr transcript and the scored duration. The default names for this columns are `'prompt'`, `'asr_transcript'`, `'scored_duration'` but they can be adapted to your data by specigying the right name for each column when you instanciate a DataGrader(). For training, you will also need a `'human_wcpm'` column for labels.

### Prediction
Once your data is in the form of a DataFrame `df` with the right column headers, you can either call the function `grade_wcpm(df)` to grade with default params or instanciate `DataGrader(df)` to do preprocessing, feature computation and prediction step by step ith customized params. 

### Retraining the model

### Training in details and hyperparameters tuning

## Algorithm and performance so far

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
│      ├── test_data_2.csv
│      └── test_data_1.csv
├── .gitignore
├── .travis.yml
├── LICENSE  
├── README.md
├── requirements.txt
├── setup.py
└── tutorial_litreading.ipynb 
```