from contextlib import asynccontextmanager
from typing import Any, Dict
import mediapipe as mp

from app.models.infer import load_model

ml_models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app):
    print("WS_PATCH v2025-09-01 | 모델/리소스 로드...")
    model, classes = load_model()
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    ml_models.update({"model": model, "classes": classes, "holistic": holistic})
    print("로드 완료.")
    yield
    print("리소스 해제...")
    ml_models["holistic"].close()
    ml_models.clear()
    print("해제 완료.")
