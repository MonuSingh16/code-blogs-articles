import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=5, n_informative=5, n_redundant=0,
                           n_repeated=0, n_classes=2,random_state=42
                           )

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

data = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4", "f5"])
data["label"] = y

data.to_csv(r"mlflow_data.csv", index=False)
