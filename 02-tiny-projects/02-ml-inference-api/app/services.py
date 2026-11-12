"""Plain functions with no HTTP in them: run the model, log predictions."""

import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from app.schemas import IrisFeatures

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "predictions.jsonl"
FEATURES = list(IrisFeatures.model_fields)


def run_model(bundle: dict, X: pd.DataFrame) -> list[dict]:
    proba = bundle["model"].predict_proba(X[FEATURES])
    idx = proba.argmax(axis=1)
    return [
        {"label": bundle["classes"][i], "confidence": float(p[i])}
        for p, i in zip(proba, idx)
    ]


def log_predictions(labels: list[str]):
    LOG_PATH.parent.mkdir(exist_ok=True)
    row = {"ts": time.time(), "n": len(labels), "counts": dict(Counter(labels))}
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")
