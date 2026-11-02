import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from housing.config import CHAMPION_ALIAS, MODEL_NAME, TRACKING_URI

GATE_METRIC = "cv_rmse"


def _score(client, version):
    return client.get_run(version.run_id).data.metrics[GATE_METRIC]


def promote_latest():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise SystemExit(f"No registered versions of '{MODEL_NAME}'. Run housing.train first.")
    candidate = max(versions, key=lambda v: int(v.version))

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        champion = None

    if champion is not None and candidate.version == champion.version:
        print(f"v{candidate.version} is already {CHAMPION_ALIAS}")
        return False

    if champion is None or _score(client, candidate) < _score(client, champion):
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, candidate.version)
        print(f"v{candidate.version} promoted to {CHAMPION_ALIAS}")
        return True

    print(f"v{candidate.version} did not beat v{champion.version}, champion unchanged")
    return False


if __name__ == "__main__":
    promote_latest()
