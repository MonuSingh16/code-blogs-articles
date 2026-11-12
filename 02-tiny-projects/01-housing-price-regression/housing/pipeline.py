import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from housing.features import add_features

SKEWED = ["AveRooms", "AveBedrms", "Population", "AveOccup"]
OTHERS = ["MedInc", "HouseAge", "Latitude", "Longitude",
          "BedroomRatio", "RoomsPerPerson", "DistSF", "DistLA"]


def build_preprocessor():
    log_scale = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("skewed", log_scale, SKEWED),
        ("rest", StandardScaler(), OTHERS),
    ])


def make_pipe(model):
    return Pipeline([
        ("fe", FunctionTransformer(add_features)),
        ("prep", build_preprocessor()),
        ("model", model),
    ])
