import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.deps import get_bundle, get_llm
from app.main import app
from app.settings import Settings, get_settings

KEY = {"x-api-key": "test-key"}


class StubModel:
    def predict_proba(self, X):
        return np.tile([0.1, 0.8, 0.1], (len(X), 1))


class FakeStream:
    def __init__(self, chunks, stop="end_turn", boom=None):
        self.chunks, self.stop, self.boom = chunks, stop, boom
        self.text_stream = self._gen()

    async def _gen(self):
        for c in self.chunks:
            yield c
        if self.boom:
            raise self.boom

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return type("Msg", (), {"stop_reason": self.stop})()


class FakeLLM:
    def __init__(self, stream):
        self._stream = stream
        self.messages = self

    def stream(self, **kwargs):
        return self._stream


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr("app.services.LOG_PATH", path)
    return path


@pytest.fixture
def client(log_file):
    bundle = {"model": StubModel(), "classes": ["setosa", "versicolor", "virginica"]}
    app.dependency_overrides[get_bundle] = lambda: bundle
    app.dependency_overrides[get_settings] = lambda: Settings(api_key="test-key")
    yield TestClient(app)  # no `with`: lifespan (model file, LLM client) is skipped
    app.dependency_overrides.clear()


@pytest.fixture
def use_llm():
    def _use(stream):
        app.dependency_overrides[get_llm] = lambda: FakeLLM(stream)

    return _use
