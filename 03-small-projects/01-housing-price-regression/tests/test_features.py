import pandas as pd
import pytest

from housing.features import add_features


def sample():
    return pd.DataFrame({
        "MedInc": [8.3], "HouseAge": [41.0], "AveRooms": [7.0], "AveBedrms": [1.0],
        "Population": [322.0], "AveOccup": [2.5],
        "Latitude": [37.77], "Longitude": [-122.42],
    })


def test_adds_columns():
    out = add_features(sample())
    assert {"BedroomRatio", "RoomsPerPerson", "DistSF", "DistLA"} <= set(out.columns)


def test_values():
    out = add_features(sample())
    assert out["BedroomRatio"].iloc[0] == pytest.approx(1 / 7)
    assert out["RoomsPerPerson"].iloc[0] == pytest.approx(7 / 2.5)
    assert out["DistSF"].iloc[0] == pytest.approx(0, abs=1e-9)


def test_input_not_mutated():
    X = sample()
    cols = list(X.columns)
    add_features(X)
    assert list(X.columns) == cols
