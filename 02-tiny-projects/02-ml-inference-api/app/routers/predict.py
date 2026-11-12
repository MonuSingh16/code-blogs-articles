import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from pydantic import ValidationError

from app.deps import Bundle, require_api_key
from app.schemas import BatchRequest, IrisFeatures, Prediction
from app.services import FEATURES, log_predictions, run_model

# every route here needs the API key, so it is declared once on the router
router = APIRouter(
    prefix="/predict", tags=["predict"], dependencies=[Depends(require_api_key)]
)


# What response_model does: FastAPI validates and filters the returned data through Prediction. Why that matters: accidental leaks (internal scores, user IDs, a password hash) can't reach the client. The response contract is enforced on your side, not by trust.
@router.post("", response_model=Prediction)
def predict(features: IrisFeatures, bundle: Bundle):
    return run_model(bundle, pd.DataFrame([features.model_dump()]))[0]


@router.post("/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest, background_tasks: BackgroundTasks, bundle: Bundle):
    X = pd.DataFrame([f.model_dump() for f in req.items])
    preds = run_model(bundle, X)
    background_tasks.add_task(log_predictions, [p["label"] for p in preds])
    return preds


@router.post("/csv", response_model=list[Prediction])
def predict_csv(file: UploadFile, background_tasks: BackgroundTasks, bundle: Bundle):
    try:
        df = pd.read_csv(file.file)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Could not parse CSV")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")

    # reuse the batch schema so CSV rows get the same bounds as JSON rows
    try:
        BatchRequest(items=df[FEATURES].to_dict("records"))
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=e.errors(include_url=False, include_context=False, include_input=False),
        )

    preds = run_model(bundle, df)
    background_tasks.add_task(log_predictions, [p["label"] for p in preds])
    return preds


# load once at startup, keep it in shared state, validate the input, return a typed response, and keep CPU-bound prediction off the event loop
