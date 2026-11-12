# ML Inference API

A small FastAPI service that serves an sklearn classifier and streams LLM responses. It exists to demonstrate the FastAPI features a data or ML engineer uses in practice.

## Endpoints

| Method | Path | Shows |
|---|---|---|
| GET | `/health` | health check |
| GET | `/models`, `/models/{name}` | path and query parameters, 404s |
| POST | `/predict` | Pydantic request and response models |
| POST | `/predict/batch` | vectorized batch inference, size limits, background logging |
| POST | `/predict/csv` | file upload, reusing the JSON schema for validation |
| POST | `/chat/stream` | async LLM call streamed as server-sent events |

## Run

    echo "APP_API_KEY=dev-secret" > .env
    uv run python train.py                 # trains and saves models/iris.joblib
    uv run uvicorn app.main:app --reload   # docs at /docs
    uv run pytest -q

For `/chat/stream`, export `ANTHROPIC_API_KEY` in the shell that runs uvicorn.

## Layout

    app/main.py        startup/shutdown and router wiring
    app/routers/       one file per feature
    app/deps.py        dependencies (model, LLM client, API key, settings)
    app/services.py    inference and logging, no HTTP
    app/settings.py    environment-based configuration
    tests/             stub model, stub LLM, dependency_overrides

## Notes

- The model loads once at startup (`lifespan`) and is shared through `Depends`.
- CPU-bound endpoints are plain `def` (thread pool). The LLM endpoint is `async def`.
- Errors after a stream starts are sent as events, since the status code is already 200.
