import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient

from api.schemas import HouseFeatures, Prediction
from housing.config import CHAMPION_ALIAS, MODEL_NAME, TRACKING_URI

state = {}


def _load_from_bundle(path: Path):
    meta = json.loads((path / "meta.json").read_text())
    model = mlflow.sklearn.load_model(str(path / "model"))
    return model, meta["version"], meta["interval_halfwidth"]


def _load_from_registry():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    halfwidth = client.get_run(mv.run_id).data.metrics["interval_halfwidth"]
    return model, str(mv.version), halfwidth


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.getenv("MODEL_DIR")
    model, version, halfwidth = (
        _load_from_bundle(Path(model_dir)) if model_dir else _load_from_registry()
    )
    state.update(model=model, version=version, halfwidth=halfwidth)
    yield
    state.clear()


app = FastAPI(title="Housing price API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": state["version"]}


@app.post("/predict", response_model=Prediction)
def predict(features: HouseFeatures):
    X = pd.DataFrame([features.model_dump()])
    pred = float(state["model"].predict(X)[0])
    q = state["halfwidth"]
    return Prediction(
        prediction=pred, lower=pred - q, upper=pred + q,
        model_version=state["version"],
    )
