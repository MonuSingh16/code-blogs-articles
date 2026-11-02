import pickle

import numpy as np
from sklearn.linear_model import Ridge

from housing.pipeline import make_pipe


def test_fit_predict_shape(toy_data):
    X, y = toy_data
    pipe = make_pipe(Ridge()).fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(X),)
    assert not np.isnan(pred).any()


def test_pickle_roundtrip(toy_data):
    X, y = toy_data
    pipe = make_pipe(Ridge()).fit(X, y)
    restored = pickle.loads(pickle.dumps(pipe))
    np.testing.assert_allclose(pipe.predict(X), restored.predict(X))
