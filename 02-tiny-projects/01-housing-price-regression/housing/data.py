import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

TARGET = "MedHouseVal"


def load_data() -> pd.DataFrame:
    return fetch_california_housing(as_frame=True).frame


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed)
