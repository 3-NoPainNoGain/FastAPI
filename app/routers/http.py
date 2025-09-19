from fastapi import APIRouter

http_router = APIRouter()

@http_router.get("/")
def root():
    return {"message": "Hello, Handoc FastAPI"}


@http_router.get("/health")
def health():
    return {"status": "ok"}