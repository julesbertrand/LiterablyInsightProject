import sys

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from litreading.model import Model


def main():
    logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])
    df = pd.read_csv("./data/larger_wcpm.csv")
    print(df.head())
    m = Model(estimator=RandomForestRegressor(), scaler=StandardScaler(), baseline_mode=False)
    m.prepare_train_test_set(df)
    m = m.fit()
    perfs = m.evaluate()
    print(perfs)


if __name__ == "__main__":
    main()
