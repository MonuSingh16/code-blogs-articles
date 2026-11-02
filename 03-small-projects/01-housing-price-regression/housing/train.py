import os

import mlflow
import numpy as np
from mlflow.models import infer_signature
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict

from housing.config import ARTIFACT_ROOT, EXPERIMENT, MODEL_NAME, TRACKING_URI
from housing.data import load_data, split_data
from housing.pipeline import make_pipe

ALPHA = 0.1
N_ITER = int(os.getenv("HOUSING_N_ITER", "25"))

PARAM_DIST = {
    "model__learning_rate": loguniform(0.02, 0.3),
    "model__max_iter": randint(200, 800),
    "model__max_leaf_nodes": randint(15, 80),
    "model__min_samples_leaf": randint(10, 60),
    "model__l2_regularization": loguniform(1e-3, 10),
}


def conformal_halfwidth(y_true, oof_pred, alpha=ALPHA):
    n = len(y_true)
    abs_res = np.abs(y_true - oof_pred)
    return float(np.quantile(abs_res, np.ceil((n + 1) * (1 - alpha)) / n))


def train():
    X_train, X_test, y_train, y_test = split_data(load_data())
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        make_pipe(HistGradientBoostingRegressor(random_state=42)),
        param_distributions=PARAM_DIST, n_iter=N_ITER, cv=cv,
        scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    oof_pred = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=-1)
    q = conformal_halfwidth(y_train.values, oof_pred)

    pred = model.predict(X_test)
    covered = (y_test.values >= pred - q) & (y_test.values <= pred + q)
    metrics = {
        "cv_rmse": -search.best_score_,
        "test_rmse": root_mean_squared_error(y_test, pred),
        "test_mae": mean_absolute_error(y_test, pred),
        "test_r2": r2_score(y_test, pred),
        "interval_halfwidth": q,
        "interval_coverage": float(covered.mean()),
    }

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="hgb-tuned") as run:
        mlflow.log_params(search.best_params_)
        mlflow.log_param("interval_alpha", ALPHA)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, name="model",
            registered_model_name=MODEL_NAME,
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(5),
        )
    return run.info.run_id


if __name__ == "__main__":
    print("run_id:", train())
