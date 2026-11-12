# California Housing Price Regression

End-to-end ML project: tuned gradient boosting model with prediction intervals, tracked in MLflow, served with FastAPI, containerized, with drift/performance monitoring and CI.

## Results (held-out test set, target in $100k)

| Model | RMSE |
|---|---|
| Mean baseline | 1.145 |
| Ridge + engineered features | 0.677 |
| Random forest (CV) | 0.492 |
| **HistGradientBoosting, tuned** | **0.440** (MAE 0.284, R² 0.852) |

90% prediction interval: ±0.673, empirical coverage 0.905.

## Layout

    housing/   data, features, pipeline, train, registry, export, monitoring
    api/       FastAPI app and schemas
    tests/     unit tests (synthetic data, stub model)
    Dockerfile, requirements-serve.txt

## Run

    uv run python -m housing.train       # tune, evaluate, log + register in MLflow
    uv run python -m housing.registry    # promote to `champion` if CV RMSE improves
    uv run python -m housing.export      # write model_bundle/ for the container
    uv run uvicorn api.main:app          # serve (loads champion from the registry)
    docker build -t housing-api . && docker run --rm -p 8080:8000 housing-api
    uv run pytest -q

Example: `POST /predict` with `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude` returns the prediction, a 90% interval and the model version.

## Design decisions

- **One pipeline artifact:** feature engineering, preprocessing and model are a single sklearn Pipeline, so training and serving can't diverge.
- **Split conformal intervals** from out-of-fold residuals; the interval width is logged with the model.
- **Champion gate:** a new model is promoted only if its CV RMSE beats the current champion. The test set is never used for selection.
- **Bundle for serving:** the container ships an exported model, not a connection to the tracking DB.
- **Monitoring:** PSI/KS data drift, and live RMSE, bias and interval coverage against the training baseline.

## Known limitations

- Target is capped at 5.0 in the source data, so top-end prices are under-predicted.
- Intervals have constant width; coverage drops for expensive houses (~0.78 in the top deciles).
- Drift checks are functions, not yet a scheduled job.

## CI

GitHub Actions (path-filtered): lint, tests, then train with fast settings, build the image and smoke-test `/health` and `/predict` in the container.
