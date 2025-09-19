import asyncio
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.models import predict_from_keypoints, extract_keypoints
from app.utils.image import safe_b64_to_bgr
from app.utils.korean import (
    to_text, FIXED_UTTERANCES, try_make_sentence_from_buffer_by_distance
)

# 파라미터
SEQUENCE_LENGTH = 30
WIN             = 6
MAJ             = 4
INACTIVITY_SEC  = 1.2
HARD_RESET_SEC  = 5.0
COOLDOWN_SEC    = 1.0

RECV_TIMEOUT_SEC = 5.0

async def run_in_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

class WebSocketState:
    def __init__(self):
        self.sequence: List[np.ndarray] = []
        self.recent_preds = deque(maxlen=WIN)
        self.word_buffer: List[str] = []
        self.last_confirm_time = 0.0
        self.last_sentence_time = 0.0
        self.last_sentence_text = ""
        self.boot_time = time.time()

async def process_frame(base64_data: str, holistic) -> Tuple[Optional[np.ndarray], Dict]:
    bgr = safe_b64_to_bgr(base64_data)
    if bgr is None:
        return None, {"warn": "empty_or_decode_fail"}

    try:
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return None, {"warn": f"cvtColor_error:{str(e)[:60]}"}

    try:
        results = await run_in_thread(holistic.process, image_rgb)
    except Exception as e:
        return None, {"warn": f"mp_process_error:{str(e)[:60]}"}

    try:
        keypoints = extract_keypoints(results)
    except Exception as e:
        return None, {"warn": f"keypoint_error:{str(e)[:60]}"}

    if keypoints is None:
        return None, {"warn": "no_keypoints"}

    return keypoints, {"coordinates": keypoints.tolist()}

async def get_prediction(sequence: List, model, classes) -> str:
    if len(sequence) != SEQUENCE_LENGTH:
        return ""
    try:
        arr = np.array(sequence)
    except Exception:
        return ""
    raw = await run_in_thread(predict_from_keypoints, arr, model, classes)
    return to_text(raw)

def handle_sentence_logic(pred: str, state: WebSocketState, resp: Dict):
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

            sentence, consume = try_make_sentence_from_buffer_by_distance(state.word_buffer, pair_max_back=PAIR_MAX_BACK)
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
            sentence, consume = try_make_sentence_from_buffer_by_distance(state.word_buffer, pair_max_back=PAIR_MAX_BACK)
            if sentence:
                if (now - state.last_sentence_time) >= COOLDOWN_SEC or state.last_sentence_text != sentence:
                    resp["sentence"] = sentence
                    state.last_sentence_time, state.last_sentence_text = now, sentence
                if consume > 0: del state.word_buffer[:consume]
            if not state.word_buffer:
                state.recent_preds.clear(); state.last_confirm_time = 0.0
