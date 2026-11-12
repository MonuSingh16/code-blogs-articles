import numpy as np


def add_features(X):
    X = X.copy()
    X["BedroomRatio"] = X["AveBedrms"] / X["AveRooms"]
    X["RoomsPerPerson"] = X["AveRooms"] / X["AveOccup"]
    X["DistSF"] = np.hypot(X["Latitude"] - 37.77, X["Longitude"] + 122.42)
    X["DistLA"] = np.hypot(X["Latitude"] - 34.05, X["Longitude"] + 118.24)
    return X
