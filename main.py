from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import sys, os
import base64
import numpy as np
import cv2, json
import mediapipe as mp

# AI 모델 디렉토리 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "ai_model"))
from ai_model.predict import load_model, predict_from_keypoints

app = FastAPI()

# 모델 로드
model = load_model()
mp_holistic = mp.solutions.holistic

# CORS 설정
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

def decode_base64_image(base64_data: str):
    image_bytes = base64.b64decode(base64_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def extract_keypoints(results):
    # ✅ 전체 33개 pose 좌표 사용
    pose = np.array(
        [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
    ).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    # ✋ 왼손 keypoints
    lh = np.array(
        [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    # 🤚 오른손 keypoints
    rh = np.array(
        [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])  # ✅ shape = (258,)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sequence = []
    threshold = 0.95

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        try:
            while True:
                base64_data = await websocket.receive_text()
                frame = decode_base64_image(base64_data)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)

                keypoints = extract_keypoints(results)
                print("📐 추출된 keypoints shape:", keypoints.shape)  # 디버깅용 로그
                
                sequence.append(keypoints)
                sequence = sequence[-30:]  # 최근 30개만 유지

                if len(sequence) == 30:
                    pred = predict_from_keypoints(np.array(sequence), model)
                    # 💡 좌표 데이터 함께 전송
                    await websocket.send_text(json.dumps({
                        "result": pred,
                        "coordinates": keypoints.tolist()
                    }))
        except WebSocketDisconnect:
            print("WebSocket 연결 종료")
