import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.lifespan import ml_models
from app.services.ws_logic import (
    WebSocketState, process_frame, get_prediction, handle_sentence_logic,
    RECV_TIMEOUT_SEC, SEQUENCE_LENGTH
)

ws_router = APIRouter()

@ws_router.websocket("/fastapi/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket 연결됨")
    
    state = WebSocketState()
    last_error_msg = None
    
    try:
        while True:
            # 수신 타임아웃 + ping으로 연결 유지
            try:
                base64_data = await asyncio.wait_for(websocket.receive_text(), timeout=RECV_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
                continue

            keypoints, resp = await process_frame(base64_data, ml_models["holistic"])

            # 빈/깨진 프레임이면 경고만 보내고 다음 루프로
            if keypoints is None:
                await websocket.send_text(json.dumps(resp))
                continue
            
            state.sequence.append(keypoints)
            state.sequence = state.sequence[-SEQUENCE_LENGTH:]
            
            pred = await get_prediction(state.sequence, ml_models["model"], ml_models["classes"])
            
            if pred:
                resp["live"] = pred
                handle_sentence_logic(pred, state, resp)

            await websocket.send_text(json.dumps(resp))

    except WebSocketDisconnect:
        print("WebSocket 연결이 종료되었습니다.")
    except Exception as e:
        msg = str(e)
        if msg != last_error_msg:
            print(f"처리 중 에러 발생: {msg}")
            last_error_msg = msg