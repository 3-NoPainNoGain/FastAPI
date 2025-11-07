import asyncio
import json
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

# AI 모델 모듈 임포트
from models.predict import predict_from_keypoints
from models.preprocessing import extract_keypoints

# 분리된 내부 모듈 임포트
from app.core.config import (
    SEQUENCE_LENGTH, WIN, MAJ, INACTIVITY_SEC, 
    HARD_RESET_SEC, COOLDOWN_SEC, PAIR_MAX_BACK, RECV_TIMEOUT_SEC
)
from app.core.utils import safe_b64_to_bgr, run_in_thread
from app.domain.korean_utils import (
    to_text, FIXED_UTTERANCES, 
    try_make_sentence_from_buffer_by_distance
)

class WebSocketState:
    """웹소켓 연결별 상태를 관리하는 클래스"""
    def __init__(self):
        self.sequence: List[np.ndarray] = []
        self.recent_preds = deque(maxlen=WIN)
        self.word_buffer: List[str] = []
        self.last_confirm_time = 0.0
        self.last_sentence_time = 0.0
        self.last_sentence_text = ""
        self.boot_time = time.time()

async def process_frame(base64_data: str, holistic) -> Tuple[Optional[np.ndarray], Dict]:
    """프레임 처리 및 키포인트 추출 (블로킹 작업을 비동기화)"""
    # 안전 디코딩
    bgr = safe_b64_to_bgr(base64_data)
    if bgr is None:
        return None, {"warn": "empty_or_decode_fail"}

    # 색 공간 변환 가드
    try:
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return None, {"warn": f"cvtColor_error:{str(e)[:60]}"}

    # 블로킹 작업: 별도 스레드에서 실행하여 이벤트 루프 정체 방지
    try:
        results = await run_in_thread(holistic.process, image_rgb)
    except Exception as e:
        return None, {"warn": f"mp_process_error:{str(e)[:60]}"}

    # 키포인트 추출 가드
    try:
        keypoints = extract_keypoints(results)
    except Exception as e:
        return None, {"warn": f"keypoint_error:{str(e)[:60]}"}

    if keypoints is None:
        return None, {"warn": "no_keypoints"}

    return keypoints, {"coordinates": keypoints.tolist()}

async def get_prediction(sequence: List, model, classes) -> str:
    """AI 모델 예측 (블로킹 작업을 비동기화)"""
    if len(sequence) != SEQUENCE_LENGTH:
        return ""
    try:
        arr = np.array(sequence)
    except Exception:
        return ""
    raw_pred = await run_in_thread(
        predict_from_keypoints, arr, model, classes
    )
    return to_text(raw_pred)

def handle_sentence_logic(pred: str, state: WebSocketState, resp: Dict):
    """문장 조합 및 타임아웃 로직 처리 (CPU 바운드가 아니므로 async 불필요)"""
    now = time.time()

    if pred in FIXED_UTTERANCES:
        if (now - state.last_sentence_time) >= COOLDOWN_SEC or state.last_sentence_text != pred:
            resp["sentence"] = pred
        state.last_sentence_time, state.last_sentence_text = now, pred
        state.recent_preds.clear(); state.word_buffer.clear(); state.last_confirm_time = 0.0
        return

    state.recent_preds.append(pred)
    if len(state.recent_preds) >= 3:
        top_word, top_count = Counter(state.recent_preds).most_common(1)[0]
        if top_count >= MAJ and (not state.word_buffer or state.word_buffer[-1] != top_word):
            state.word_buffer.append(top_word)
            state.last_confirm_time = now
            
            sentence, consume = try_make_sentence_from_buffer_by_distance(state.word_buffer)
            if sentence:
                if (now - state.last_sentence_time) >= COOLDOWN_SEC or state.last_sentence_text != sentence:
                    resp["sentence"] = sentence
                    state.last_sentence_time, state.last_sentence_text = now, sentence
                if consume > 0: del state.word_buffer[:consume]
                state.recent_preds.clear()
                return

    if state.last_confirm_time:
        is_inactive = (now - state.last_confirm_time) > INACTIVITY_SEC
        is_hard_reset = (now - state.boot_time) > HARD_RESET_SEC and is_inactive
        if is_inactive or is_hard_reset:
            sentence, consume = try_make_sentence_from_buffer_by_distance(state.word_buffer)
            if sentence:
                if (now - state.last_sentence_time) >= COOLDOWN_SEC or state.last_sentence_text != sentence:
                    resp["sentence"] = sentence
                    state.last_sentence_time, state.last_sentence_text = now, sentence
                if consume > 0: del state.word_buffer[:consume]
            if not state.word_buffer:
                state.recent_preds.clear(); state.last_confirm_time = 0.0
