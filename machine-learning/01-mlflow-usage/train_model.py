import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

test_size = 0.2
n_estimators = 100
max_depth = 4

def performance(actual, pred):

    f1 = f1_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)

    return f1, accuracy, precision

if __name__ == "__main__":

    data = pd.read_csv(r"..//..//data//mlflow_data.csv")

    train, test = train_test_split(data, test_size=test_size)
    
    X_train = train.drop(["label"], axis=1)
    y_train = train["label"]

    X_test = test.drop(["label"], axis=1)
    y_test = test["label"]

    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    rf_model.fit(X_train, y_train)

    predictions = rf_model.predict(X_test)

    f1, accuracy, precision = performance(y_test, predictions)

    print(f"Random Forest: {n_estimators}, {max_depth}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")





