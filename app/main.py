import mediapipe as mp
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# AI 모델 관련 모듈 임포트
from models.predict import load_model

# 분리된 내부 모듈 임포트
from app.services.websocket_handler import WebSocketHandler
from app.routers import base  # HTTP 라우터 임포트

ml_models: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작 시 모델을 로드하고, 종료 시 리소스를 해제합니다."""
    print("WS_PATCH v2025-09-01")  # 버전 식별용 로그
    print("AI 모델 및 MediaPipe 리소스를 로드합니다...")
    
    model, classes = load_model()
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    ml_models.update({"model": model, "classes": classes, "holistic": holistic})
    print("로드 완료.")
    
    yield  # 애플리케이션 실행
    
    print("리소스를 해제합니다...")
    ml_models["holistic"].close()
    ml_models.clear()
    print("해제 완료.")

app = FastAPI(lifespan=lifespan)

# CORS 설정 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(base.router)

@app.websocket("/fastapi/ws")
async def ws(websocket: WebSocket):
    """
    웹소켓 연결을 수락하고, 모든 처리를 WebSocketHandler에게 위임합니다.
    """
    await websocket.accept()
    print("WebSocket 연결됨")
    
    # Lifespan에서 로드된 모델 리소스를 핸들러에 주입
    handler = WebSocketHandler(websocket, ml_models)
    
    try:
        # 핸들러가 연결의 모든 로직을 처리하도록 위임
        await handler.handle_connection()
        
    except Exception as e:
        # 핸들러 내부가 아닌, 연결 수락/위임 과정의 예외 처리
        print(f"웹소켓 연결 수준 에러 발생: {e}")
    finally:
        print("WebSocket 연결 종료 (main).")