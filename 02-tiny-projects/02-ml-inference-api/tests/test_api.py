import anthropic
import httpx
import pytest

from tests.conftest import KEY, FakeStream

ROW = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
CSV_HEADER = "sepal_length,sepal_width,petal_length,petal_width\n"


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_models_list_filter_and_404(client):
    assert len(client.get("/models").json()) == 3
    assert [m["name"] for m in client.get("/models?task=regression").json()] == ["housing"]
    assert client.get("/models/xyz").status_code == 404
    assert client.get("/models?limit=abc").status_code == 422


def test_predict_ok(client):
    r = client.post("/predict", json=ROW, headers=KEY)
    assert r.status_code == 200
    assert r.json() == {"label": "versicolor", "confidence": 0.8}


def test_predict_auth(client):
    assert client.post("/predict", json=ROW, headers={"x-api-key": "bad"}).status_code == 401
    assert client.post("/predict", json=ROW).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("sepal_length", 0), ("petal_width", -1), ("sepal_width", 11),
])
def test_predict_rejects_invalid_values(client, field, value):
    r = client.post("/predict", json={**ROW, field: value}, headers=KEY)
    assert r.status_code == 422


def test_batch_ok_and_logs(client, log_file):
    r = client.post("/predict/batch", json={"items": [ROW, ROW]}, headers=KEY)
    assert r.status_code == 200 and len(r.json()) == 2
    assert '"versicolor": 2' in log_file.read_text()  # background task ran


def test_batch_rejects_empty(client):
    assert client.post("/predict/batch", json={"items": []}, headers=KEY).status_code == 422


def test_csv_ok(client):
    body = (CSV_HEADER + "5.1,3.5,1.4,0.2\n").encode()
    r = client.post("/predict/csv", files={"file": ("a.csv", body, "text/csv")}, headers=KEY)
    assert r.status_code == 200 and len(r.json()) == 1


def test_csv_rejects_missing_column_and_bad_values(client):
    f1 = {"file": ("a.csv", b"sepal_length\n5.1\n", "text/csv")}
    assert client.post("/predict/csv", files=f1, headers=KEY).status_code == 422
    f2 = {"file": ("a.csv", (CSV_HEADER + "-1,3,1,1\n").encode(), "text/csv")}
    assert client.post("/predict/csv", files=f2, headers=KEY).status_code == 422


def test_chat_streams_tokens(client, use_llm):
    use_llm(FakeStream(["Hello", " world"]))
    r = client.post("/chat/stream", json={"message": "hi"}, headers=KEY)
    assert r.status_code == 200
    assert 'data: "Hello"' in r.text
    assert r.text.endswith("event: done\ndata: {}\n\n")


def test_chat_refusal_and_api_error_become_events(client, use_llm):
    use_llm(FakeStream([], stop="refusal"))
    assert "event: error" in client.post("/chat/stream", json={"message": "hi"}, headers=KEY).text
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))
    use_llm(FakeStream(["part"], boom=err))
    r = client.post("/chat/stream", json={"message": "hi"}, headers=KEY)
    assert 'data: "part"' in r.text and "event: error" in r.text


def test_chat_rejects_empty_message(client, use_llm):
    use_llm(FakeStream([]))
    assert client.post("/chat/stream", json={"message": ""}, headers=KEY).status_code == 422
