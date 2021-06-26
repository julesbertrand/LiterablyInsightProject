import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from litreading.train import Model


def main():
    df = pd.read_csv("./data/larger_wcpm.csv")
    print(df.head())
    m = Model(estimator=RandomForestRegressor(), scaler=StandardScaler())
    m.prepare_train_test_set(df)
    m = m.fit()
    perfs = m.evaluate()
    print(perfs)


if __name__ == "__main__":
    main()
