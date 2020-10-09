# LitReadingAI - Automated assessment of students' reading skills and literacy
This is a consulting project done as part of Insight Artificial Intelligence Fellowship.

## Background

## Installation

## Execution

### Data

### Prediction

### Retraining the model

### Training in details and hyperparameters tuning

## Algorithm and performance so far

## Repo Directory structure
The package is located in the `litreading/` folder. All the modules are located in it along side a configuration file `config.py` where paths to the required files and model training default config are defined. The `litreading/models/` folder holds the models which are used for prediction and newly trained models - make sure that the MODELS_PATH variable in `config.py` is updated and points to the location of the `litreading/models/` folder! The `dataset.py` file hosts the Dataset class for preprocessing and data computation, the `grade.py` file hosts the DataGrader class for wcpm estimation, and the `train.py` files hosts the ModelTrainer class for training default model, new models, or make an sklearn gridsearch.

Unit tests are located in the `tests/` folder and for them to run properly, the variable TEST_FOLDER in config.py should point to the `tests/test_data/` folder.

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
├── LICENSE
├── tutorial_litreading.ipynb   
├── README.md
├── requirements.txt
└── setup.py
```