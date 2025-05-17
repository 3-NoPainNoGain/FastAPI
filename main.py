from fastapi import FastAPI, WebSocket, WebSocketDisconnect 
from fastapi.middleware.cors import CORSMiddleware
import asyncio, json, numpy as np
import sys, os

# AI 모델 디렉토리 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "ai_model"))

from predict import load_model, predict_from_keypoints

app = FastAPI()

# 서버 시작 시 모델 한 번 로드 
model = load_model()

# CORS 설정 (프론트앤드에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결됨")

    try:
        while True:
            data = await websocket.receive_text()
            print("받은 메시지: ", data)
            message = json.loads(data)
            coordinates = message.get("coordinates", [])
            print("좌표 수:", len(coordinates))

            if len(coordinates) == 154:
                keypoints = np.array([coordinates])
                predicted_text = predict_from_keypoints(keypoints, model)
                await websocket.send_text(json.dumps({"text": predicted_text}))
            else:
                await websocket.send_text(json.dumps({"text": "좌표 개수 오류: 154개가 아님"}))

    except WebSocketDisconnect:
        print("WebSocket 연결 종료")