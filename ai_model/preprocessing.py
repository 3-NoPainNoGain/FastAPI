# ai_model/preprocessing.py
import base64
import re
import numpy as np
import cv2
import mediapipe as mp

# (옵션) 모듈 전역에 홀리스틱 네임스페이스만 유지
mp_holistic = mp.solutions.holistic

# data URL 프리픽스 제거용 (예: "data:image/jpeg;base64,XXXX")
_DATA_URL_RE = re.compile(r'^data:image/\w+;base64,', re.I)

def _pad_base64(s: str) -> str:
    """개행/공백 제거 + 누락 패딩 보정('=' 채움)"""
    s = s.replace('\n', '').replace('\r', '').replace(' ', '')
    missing = (-len(s)) % 4
    if missing:
        s += "=" * missing
    return s

def decode_base64_image(base64_data: str, max_bytes: int = 2_000_000):
    """
    안전한 base64 → OpenCV BGR 이미지
      - data URL 프리픽스 제거
      - 개행/공백 제거
      - 패딩 보정
      - 사이즈 제한 (기본 2MB)
    실패 시 None 반환
    """
    if not base64_data:
        return None

    s = _DATA_URL_RE.sub('', base64_data.strip())
    s = _pad_base64(s)

    try:
        raw = base64.b64decode(s, validate=False)  # 깨진 패딩 허용
    except Exception:
        return None

    if len(raw) > max_bytes:
        return None

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img

def extract_keypoints(results):
    """
    Pose(33*4) + LeftHand(21*3) + RightHand(21*3) = 258
    """
    pose = (np.array([[lm.x, lm.y, lm.z, lm.visibility]
                      for lm in (results.pose_landmarks.landmark if results.pose_landmarks else [])])
            if results.pose_landmarks else np.zeros((33, 4), dtype=float))

    lh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in (results.left_hand_landmarks.landmark if results.left_hand_landmarks else [])])
          if results.left_hand_landmarks else np.zeros((21, 3), dtype=float))

    rh = (np.array([[lm.x, lm.y, lm.z]
                    for lm in (results.right_hand_landmarks.landmark if results.right_hand_landmarks else [])])
          if results.right_hand_landmarks else np.zeros((21, 3), dtype=float))

    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()]).astype(float)  # (258,)
