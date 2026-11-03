from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")  # registers the function as the handler for GET /health
def health():
    return {"status": "ok"}  # FastAPI turns the dict into JSON


# GET is idempotent and safe. Repeating it causes no extra effect, so clients and proxies can retry and cache it freely. A POST can have side effects, so a retry might do the work twice, and that matters for payments or logging.
