from fastapi import FastAPI, WebSocket, WebSocketDisconnect 
from fastapi.middleware.cors import CORSMiddleware
import time
import asyncio 
import json
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"message" : "Hello FastAPI"}

### 모델 연결 
# CORS 설정 (프론트앤드에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket:WebSocket):
    await websocket.accept()
    print("WebSocket 연결됨")

    try:
        while True:
            # 프론트에서 JSON 메시지 수신 
            data = await websocket.receive_text()
            print("받은 메시지: ",data)
            message = json.loads(data)
            coordinates = message.get("coordinates", [])
            print("좌표 수:", len(coordinates)) 

            # 좌표 수 확인 
            if len(coordinates) == 154:
                keypoints = np.array(coordinates).reshape(1, 154)

                # 여기서 수어 예측 모델 호출 
                predicted_text = "아프다"

                # 클라이언트에 응답 전송 
                await websocket.send_text(json.dumps({"text" : predicted_text}))
            else:
                await websocket.send_text(json.dumps({"text" :  "좌표 개수 오류 : 154개가 아님"}))
    
    except WebSocketDisconnect: 
        print("WebSocket 연결 종료")