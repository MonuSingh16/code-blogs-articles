FROM python:3.12-slim

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY housing/ housing/
COPY api/ api/
COPY model_bundle/ model_bundle/

ENV MODEL_DIR=/app/model_bundle

RUN useradd -m app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
