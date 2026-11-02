from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = (ROOT / "mlruns").as_uri()
EXPERIMENT = "housing-price-regression"
MODEL_NAME = "housing-regressor"
CHAMPION_ALIAS = "champion"
