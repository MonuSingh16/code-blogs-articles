import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn # module provides an API for logging and loading scikit-learn models

test_size = 0.2
n_estimators = 50
max_depth = 5

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

    # Differen hypeparams values 
    #hyperparameters_values = [0.01, 0.1, 0.5, 1.0]
 
    #for lr in hyperparameters_values:
    # change tracking location
    # mlflow.set_tracking_uri(r"./model_tracking")
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5005")

    # create experiment 
    my_exp = mlflow.set_experiment("mlflow_tracking_server")
    with mlflow.start_run(experiment_id=my_exp.experiment_id, run_name="custom_name"):
        # intiates a new run within a active mlflow experiment
        # can continue an existing run id, run_id = ""

        # Mlflow logging automatically captures the code version, the framewrok used
        mlflow.autolog()
        
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf_model.fit(X_train, y_train)

        # # log model hyperparameters
        # mlflow.log_params({"n_estimators": n_estimators,
        #                    "max_depth": max_depth})
        
        predictions = rf_model.predict(X_test)
        f1, accuracy, precision = performance(y_test, predictions)
        
        # # log performance metrics
        # mlflow.log_metrics({"f1_score": f1,
        #                     "accuracy_score": accuracy,
        #                     "precision": precision})
        # # active run, return run object which is active
        # active_run = mlflow.active_run()
        # print(f"Active Run ID: {active_run.info.run_id}")
        # print(f"Active Run Name: {active_run.info.run_name}")
        # print(f"Active Run Parameters: {active_run.data.params}")
        # print(f"Active Run Metrics: {active_run.data.metrics}") # empty as logging & retrieval of params
        # # Happens before ml_flow.end_run()

        # # log model
        # mlflow.sklearn.log_model(rf_model, "model") # model & relative artifact

        # to gracefully conclude the run -> mlflow.end_run()
        # as context manager esnures this part, If not using then advise to use end_run()
        # context manager - __enter__ and __exit__ method is called. Ensuring resources are managerd


    print(f"Random Forest: {n_estimators}, {max_depth}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")





