import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "MedInc": rng.uniform(1, 10, n),
        "HouseAge": rng.uniform(1, 50, n),
        "AveRooms": rng.uniform(2, 9, n),
        "AveBedrms": rng.uniform(0.8, 1.5, n),
        "Population": rng.uniform(100, 4000, n),
        "AveOccup": rng.uniform(1.5, 5, n),
        "Latitude": rng.uniform(32.5, 42, n),
        "Longitude": rng.uniform(-124, -114, n),
    })
    y = 0.4 * X["MedInc"] + rng.normal(0, 0.1, n)
    return X, y
