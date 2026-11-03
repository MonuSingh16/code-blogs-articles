"""Dependencies: things endpoints ask FastAPI to prepare via Depends."""

import secrets
from typing import Annotated

import anthropic
from fastapi import Depends, Header, HTTPException

from app.settings import Settings, get_settings
from app.state import state

Cfg = Annotated[Settings, Depends(get_settings)]


def get_bundle():
    bundle = state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return bundle


def get_llm():
    llm = state.get("llm")
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM client not ready")
    return llm


def require_api_key(cfg: Cfg, x_api_key: str = Header()):
    if not secrets.compare_digest(x_api_key.encode(), cfg.api_key.encode()):
        raise HTTPException(status_code=401, detail="Invalid API key")


# Annotated aliases: endpoints write `bundle: Bundle` instead of `Depends(get_bundle)`
Bundle = Annotated[dict, Depends(get_bundle)]
LLM = Annotated[anthropic.AsyncAnthropic, Depends(get_llm)]
