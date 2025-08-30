import base64
import numpy as np
import cv2
import mediapipe as mp

# Mediapipe Holistic 초기화
mp_holistic = mp.solutions.holistic

# 프론트(WebSocket)에서 받은 base64 문자열을 OpenCV 이미지로 변환
def decode_base64_image(base64_data: str):
    image_bytes = base64.b64decode(base64_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# Mediapipe Holistic 결과에서 keypoints 추출 → 258차원 벡터 생성
# Pose(33*4) + LeftHand(21*3) + RightHand(21*3) = 258
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]) \
                     if results.pose_landmarks else np.zeros((33, 4))

    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]) \
                   if results.left_hand_landmarks else np.zeros((21, 3))

    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]) \
                   if results.right_hand_landmarks else np.zeros((21, 3))

    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])  # shape = (258,)
