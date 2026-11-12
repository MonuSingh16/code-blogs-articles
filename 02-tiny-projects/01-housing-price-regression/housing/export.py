import json
import shutil

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from housing.config import CHAMPION_ALIAS, MODEL_NAME, ROOT, TRACKING_URI

BUNDLE_DIR = ROOT / "model_bundle"


def export_champion(out_dir=BUNDLE_DIR):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    metrics = client.get_run(mv.run_id).data.metrics
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    mlflow.sklearn.save_model(model, out_dir / "model")
    (out_dir / "meta.json").write_text(json.dumps({
        "version": str(mv.version),
        "interval_halfwidth": metrics["interval_halfwidth"],
        "test_rmse": metrics["test_rmse"],
    }, indent=2))
    return out_dir


if __name__ == "__main__":
    print("exported to", export_champion())
