"""App entry point: startup/shutdown and wiring. Endpoints live in app/routers/."""

from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
import joblib
from fastapi import FastAPI

from app.routers import chat, health, models, predict
from app.state import state

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris.joblib"


# Why load in lifespan: unpickling takes time. Doing it once at startup means each request only pays for the prediction. If the file is missing, startup fails, and the app never serves with no model
@asynccontextmanager
async def lifespan(app: FastAPI):
    state["bundle"] = joblib.load(MODEL_PATH)
    state["llm"] = anthropic.AsyncAnthropic()
    yield
    await state["llm"].close()
    state.clear()


app = FastAPI(title="ML Inference API", lifespan=lifespan)

app.include_router(health.router)
app.include_router(models.router)
app.include_router(predict.router)
app.include_router(chat.router)
