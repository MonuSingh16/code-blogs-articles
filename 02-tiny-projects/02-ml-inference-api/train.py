from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(as_frame=True, return_X_y=True)
X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
out = Path("models")
out.mkdir(exist_ok=True)
joblib.dump({"model": clf, "classes": list(load_iris().target_names)}, out/"iris.joblib")
print("saved", out/"iris.joblib")
