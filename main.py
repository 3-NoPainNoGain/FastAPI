import asyncio
import base64
import json
import re
import time
import unicodedata
from collections import Counter, deque
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# AI 모델 관련 모듈 임포트
from ai_model.predict import load_model, predict_from_keypoints
from ai_model.preprocessing import extract_keypoints  # decode_base64_image 대신 안전 디코더 사용

# ==============================================================================
# 1. 파라미터 및 상수 정의
# ==============================================================================
SEQUENCE_LENGTH = 30   # 모델 입력 시퀀스 길이
WIN             = 6    # 최근 예측을 저장할 창 크기
MAJ             = 4    # 다수결 판정을 위한 임계값
INACTIVITY_SEC  = 1.2  # 비활성 상태로 간주할 시간 (초)
HARD_RESET_SEC  = 5.0  # 시스템을 초기화할 비활성 시간 (초)
COOLDOWN_SEC    = 1.0  # 동일 문장 반복 출력을 막기 위한 쿨다운
PAIR_MAX_BACK   = 6    # 동사-명사 조합을 찾을 최대 거리
RECV_TIMEOUT_SEC = 5.0 # WS 수신 타임아웃 (ping 유지용)

# ==============================================================================
# 2. FastAPI 앱 생애주기(Lifespan) 및 리소스 관리
# ==============================================================================
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

# ==============================================================================
# 3. CORS 미들웨어 설정
# ==============================================================================
# !! 중요: 배포 시에는 ["*"] 대신 실제 프론트엔드 도메인을 명시해야 합니다. !!
# 예: allow_origins=["https://your-frontend.com", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 4. 한국어 처리 및 문장 조합 유틸리티
# ==============================================================================
def to_text(x: Any) -> str:
    if x is None: s = ""
    elif isinstance(x, bytes):
        try: s = x.decode("utf-8", "ignore")
        except Exception: s = str(x)
    else:
        try: s = x.item()
        except Exception: s = x
        s = str(s)
    return unicodedata.normalize("NFC", s.strip())

_raw_FIXED = {"안녕하세요", "감사합니다"}
_raw_NOUNS = {"열", "콧물", "코", "기침"}
_raw_VERBS = {"있다", "없다", "막히다", "아프다"}

FIXED_UTTERANCES = {to_text(s) for s in _raw_FIXED}
NOUN_OVERRIDES   = {to_text(s) for s in _raw_NOUNS}
VERB_OVERRIDES   = {to_text(s) for s in _raw_VERBS}

def is_verb(w: str) -> bool:
    if not isinstance(w, str): return False
    if w in VERB_OVERRIDES: return True
    if w in NOUN_OVERRIDES: return False
    return w.endswith("다")

def has_jongseong(word: str) -> bool:
    if not word: return False
    ch = word[-1]
    base = ord('가')
    code = ord(ch) - base
    return 0 <= code <= 11171 and (code % 28) != 0

def subject_particle(noun: str) -> str:
    return '이' if has_jongseong(noun) else '가'

JUNGSEONG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
def _decompose(ch: str):
    code = ord(ch)
    if not (0xAC00 <= code <= 0xD7A3): return None
    s = code - 0xAC00
    c, v, f = s // 588, (s % 588) // 28, s % 28
    return c, v, f
def _compose(c_idx: int, v_idx: int, f_idx: int = 0) -> str:
    return chr(0xAC00 + c_idx*588 + v_idx*28 + f_idx)
def _last_vowel(s: str) -> str:
    if not s: return ''
    parts = _decompose(s[-1])
    if not parts: return ''
    _, v_idx, _ = parts
    return JUNGSEONG[v_idx]
def _has_jong(s: str) -> bool:
    parts = _decompose(s[-1]) if s else None
    return bool(parts and parts[2] != 0)
def _replace_last_vowel(s: str, new_vowel: str) -> str:
    if not s: return s
    parts = _decompose(s[-1])
    if not parts: return s
    c, _, f = parts
    v_idx = JUNGSEONG.index(new_vowel)
    return s[:-1] + _compose(c, v_idx, f)

def conjugate_to_polite(verb: str) -> str:
    verb = to_text(verb)
    if not verb.endswith("다"): return verb
    stem = verb[:-1]
    if _last_vowel(stem) == "ㅡ":
        base = stem[:-1]
        prev_v = _last_vowel(base)
        chosen = "ㅏ" if prev_v in ["ㅏ", "ㅗ"] else "ㅓ"
        new_stem = _replace_last_vowel(stem, chosen)
        return new_stem + "요"
    if not _has_jong(stem) and _last_vowel(stem) == "ㅣ":
        return _replace_last_vowel(stem, "ㅕ") + "요"
    if _last_vowel(stem) in ["ㅏ", "ㅗ"]:
        if not _has_jong(stem) and _last_vowel(stem) == "ㅏ": return stem + "요"
        return stem + "아요"
    else: return stem + "어요"

def format_noun_verb(noun: str, verb: str) -> str:
    if not noun or not verb: return ""
    polite_verb = conjugate_to_polite(verb)
    return f"{noun}{subject_particle(noun)} {polite_verb}"

def try_make_sentence_from_buffer_by_distance(buf: List[str]) -> Tuple[str, int]:
    if not buf: return "", 0
    v_idx = -1
    for i in range(len(buf) - 1, -1, -1):
        if is_verb(buf[i]):
            v_idx = i
            break
    if v_idx == -1: return "", 0
    start = max(0, v_idx - PAIR_MAX_BACK)
    n_idx = -1
    for j in range(v_idx - 1, start - 1, -1):
        if not is_verb(buf[j]):
            n_idx = j
            break
    if n_idx == -1: return "", 0
    sentence = format_noun_verb(buf[n_idx], buf[v_idx])
    return (sentence, v_idx + 1) if sentence else ("", 0)

# ==============================================================================
# 5. 안전 디코더 & 헬퍼
# ==============================================================================
_B64_DATAURL = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,")

def safe_b64_to_bgr(s: str) -> Optional[np.ndarray]:
    """dataURL/패딩/공백을 보정해 안전하게 BGR 이미지를 반환."""
    try:
        if not s:
            return None
        # dataURL 헤더 제거 + 공백/URL변형 보정 + 패딩 보정
        s = _B64_DATAURL.sub("", s).replace(" ", "+").replace("%2B", "+")
        pad = len(s) % 4
        if pad:
            s += "=" * (4 - pad)
        buf = base64.b64decode(s, validate=False)
        if not buf:
            return None
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size == 0:
            return None
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        return img
    except Exception:
        return None

async def run_in_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

# ==============================================================================
# 6. 웹소켓 로직을 위한 헬퍼 (상태 관리 클래스 및 비동기 함수)
# ==============================================================================
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

# ==============================================================================
# 7. 메인 API 엔드포인트
# ==============================================================================
@app.get("/")
def root():
    return {"message": "Handoc Backend API", "version": "v2025-09-01"}

@app.get("/health")
def health():
    # 헬스는 초경량: 모델/미디어파이프 접근 금지
    return {"status": "ok"}

@app.websocket("/fastapi/ws")
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
