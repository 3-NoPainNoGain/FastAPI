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
mp_holistic= mp.solutions.holistic

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

def decode_base64_image(base64_data: str):
    image_bytes = base64.b64decode(base64_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def extract_keypoints(results):
    # 상반신 pose index 7개: 0, 11, 12, 13, 14, 15, 16
    pose_indices = [0, 11, 12, 13, 14, 15, 16]
    pose = np.array(
        [[results.pose_landmarks.landmark[i].x,
          results.pose_landmarks.landmark[i].y,
          results.pose_landmarks.landmark[i].z,
          results.pose_landmarks.landmark[i].visibility]
         for i in pose_indices]
    ).flatten() if results.pose_landmarks else np.zeros(7 * 4)

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])

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
                sequence.append(keypoints)
                sequence = sequence[-30:]  # 최근 30개만 유지

                if len(sequence) == 30:
                    pred = predict_from_keypoints(np.array(sequence), model)
                    print("예측 결과 :", pred)
                    await websocket.send_text(json.dumps({"result": pred}))
        except WebSocketDisconnect:
            print("WebSocket 연결 종료")