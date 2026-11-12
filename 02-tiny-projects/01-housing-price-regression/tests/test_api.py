import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, state

VALID = {
    "MedInc": 8.3, "HouseAge": 41, "AveRooms": 7.0, "AveBedrms": 1.0,
    "Population": 322, "AveOccup": 2.5, "Latitude": 37.88, "Longitude": -122.23,
}


class StubModel:
    def predict(self, X):
        assert isinstance(X, pd.DataFrame)
        return [2.0]


@pytest.fixture
def client():
    state.update(model=StubModel(), version="1", halfwidth=0.5)
    yield TestClient(app)  # no `with`, so lifespan (registry load) is skipped
    state.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "1"}


def test_predict(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert r.json() == {
        "prediction": 2.0, "lower": 1.5, "upper": 2.5, "model_version": "1",
    }


@pytest.mark.parametrize("field,value", [
    ("AveRooms", 0), ("AveOccup", -1), ("Latitude", 100),
])
def test_rejects_invalid_values(client, field, value):
    r = client.post("/predict", json={**VALID, field: value})
    assert r.status_code == 422


def test_rejects_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "MedInc"}
    assert client.post("/predict", json=bad).status_code == 422
