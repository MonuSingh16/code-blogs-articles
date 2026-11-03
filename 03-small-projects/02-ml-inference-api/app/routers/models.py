from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/models", tags=["models"])

MODELS = {
    "iris": {"task": "classification", "n_features": 4},
    "cancer": {"task": "classification", "n_features": 30},
    "housing": {"task": "regression", "n_features": 8},
}


@router.get("")
def list_models(task: str | None = None, limit: int = 10):
    items = [
        {"name": name, **meta}
        for name, meta in MODELS.items()
        if task is None or meta["task"] == task
    ]
    return items[:limit]


@router.get("/{name}")
def get_model(name: str):
    if name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return {"name": name, **MODELS[name]}
