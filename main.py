# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import asyncio, json, time, unicodedata
import numpy as np
import mediapipe as mp
import cv2
from collections import Counter, deque
from typing import Any, List, Tuple
from starlette.websockets import WebSocketState, CloseCode

from ai_model.predict import load_model, predict_from_keypoints
from ai_model.preprocessing import decode_base64_image, extract_keypoints

app = FastAPI()

# ---- Health (HEAD도 명시 지원) ----
@app.get("/health")
def health_get():
    return {"status": "ok"}

@app.head("/health")
def health_head():
    return Response(status_code=200)

# ====== 파라미터 ======
WIN            = 6     # 최근 예측 창 크기
MAJ            = 4     # 다수결 임계
INACTIVITY_SEC = 1.2   # 마지막 확정단어 이후 입력 뜸하면 강제 플러시
HARD_RESET_SEC = 5.0   # 하드 리셋
COOLDOWN_SEC   = 1.0   # 같은 문장/고정 문구 연타 방지
PAIR_MAX_BACK  = 6     # 동사 앞에서 최대 몇 개 안에서 명사를 찾을지
FRAME_INTERVAL_SEC = 0.12  # 프레임 수신 스로틀
RECV_TIMEOUT_SEC   = 30.0  # 유휴 수신 타임아웃

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 정규화 유틸 ======
def to_text(x: Any) -> str:
    if x is None:
        s = ""
    elif isinstance(x, bytes):
        try:
            s = x.decode("utf-8", "ignore")
        except Exception:
            s = str(x)
    else:
        try:
            s = x.item()
        except Exception:
            s = x
        s = str(s)
    return unicodedata.normalize("NFC", s.strip())

# ====== 라벨/품사 ======
_raw_FIXED = {"안녕하세요", "감사합니다"}
_raw_NOUNS = {"열", "콧물", "코", "기침"}
_raw_VERBS = {"있다", "없다", "막히다", "아프다"}

FIXED_UTTERANCES = {to_text(s) for s in _raw_FIXED}
NOUN_OVERRIDES   = {to_text(s) for s in _raw_NOUNS}
VERB_OVERRIDES   = {to_text(s) for s in _raw_VERBS}

def is_verb(w: str) -> bool:
    if not isinstance(w, str):
        return False
    if w in VERB_OVERRIDES: return True
    if w in NOUN_OVERRIDES: return False
    return w.endswith("다")

def has_jongseong(word: str) -> bool:
    if not word:
        return False
    ch = word[-1]
    base = ord('가')
    code = ord(ch) - base
    return 0 <= code <= 11171 and (code % 28) != 0

def subject_particle(noun: str) -> str:
    return '이' if has_jongseong(noun) else '가'

# ====== 한글 합성/분해 최소 유틸 ======
JUNGSEONG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']

def _decompose(ch: str):
    code = ord(ch)
    if not (0xAC00 <= code <= 0xD7A3):
        return None
    s = code - 0xAC00
    c = s // 588
    v = (s % 588) // 28
    f = s % 28
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
    if not verb.endswith("다"):
        return verb

    stem = verb[:-1]

    # ㅡ 불규칙
    if _last_vowel(stem) == "ㅡ":
        base = stem[:-1]
        prev_v = _last_vowel(base)
        chosen = "ㅏ" if prev_v in ["ㅏ", "ㅗ"] else "ㅓ"
        new_stem = _replace_last_vowel(stem, chosen)
        return new_stem + "요"

    # ㅣ + 어 → 여
    if not _has_jong(stem) and _last_vowel(stem) == "ㅣ":
        return _replace_last_vowel(stem, "ㅕ") + "요"

    if _last_vowel(stem) in ["ㅏ", "ㅗ"]:
        if not _has_jong(stem) and _last_vowel(stem) == "ㅏ":
            return stem + "요"
        return stem + "아요"
    else:
        return stem + "어요"

def format_noun_verb(noun: str, verb: str) -> str:
    if not noun or not verb:
        return ""
    polite_verb = conjugate_to_polite(verb)
    return f"{noun}{subject_particle(noun)} {polite_verb}"

# ====== 모델 ======
model, classes = load_model()
mp_holistic = mp.solutions.holistic

@app.get("/")
def root():
    return {"message": "Hello FastAPI"}

def try_make_sentence_from_buffer_by_distance(buf: List[str]) -> Tuple[str, int]:
    if not buf:
        return "", 0

    # 최근 동사
    v_idx = -1
    for i in range(len(buf) - 1, -1, -1):
        if is_verb(buf[i]):
            v_idx = i
            break
    if v_idx == -1:
        return "", 0

    # v_idx 앞에서 가까운 명사 (최대 PAIR_MAX_BACK)
    start = max(0, v_idx - PAIR_MAX_BACK)
    n_idx = -1
    for j in range(v_idx - 1, start - 1, -1):
        if not is_verb(buf[j]):
            n_idx = j
            break
    if n_idx == -1:
        return "", 0

    noun = buf[n_idx]
    verb = buf[v_idx]
    sentence = format_noun_verb(noun, verb)
    if not sentence:
        return "", 0

    return sentence, (v_idx + 1)

# ====== WebSocket ======
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("WebSocket 연결됨")

    sequence: List[np.ndarray] = []
    recent_preds = deque(maxlen=WIN)
    word_buffer: List[str] = []

    last_confirm_time = 0.0
    last_sentence_time = 0.0
    last_sentence_text = ""
    boot_time = time.time()
    last_sent_ts = 0.0

    last_error_msg = None  # 동일 에러 중복 로그 억제

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        try:
            while True:
                # 유휴 타임아웃 관리 + 프레임 스로틀
                try:
                    msg = await asyncio.wait_for(ws.receive_text(), timeout=RECV_TIMEOUT_SEC)
                except asyncio.TimeoutError:
                    if ws.application_state == WebSocketState.CONNECTED:
                        await ws.send_text('{"type":"keepalive"}')
                        continue
                    else:
                        break

                now = time.perf_counter()
                if now - last_sent_ts < FRAME_INTERVAL_SEC:
                    # 과도 프레임 드롭
                    continue
                last_sent_ts = now

                # base64 → 이미지
                frame = decode_base64_image(msg)
                if frame is None or (hasattr(frame, "size") and frame.size == 0):
                    # 깨진 프레임은 에코만
                    await ws.send_text('{"type":"error","reason":"bad_base64_or_empty"}')
                    continue

                # 추론 파이프라인
                try:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    # 드물게 디코드 실패 시
                    err = f"cv2.cvtColor error: {e}"
                    if err != last_error_msg:
                        print("loop error:", err)
                        last_error_msg = err
                    await ws.send_text('{"type":"error","reason":"cvtColor_failed"}')
                    continue

                results = holistic.process(image_rgb)
                keypoints = extract_keypoints(results)

                sequence.append(keypoints)
                sequence = sequence[-30:]

                resp = {"coordinates": keypoints.tolist()}

                if len(sequence) == 30:
                    try:
                        raw = predict_from_keypoints(np.array(sequence), model, classes)
                        pred = to_text(raw)
                        resp["live"] = pred
                    except Exception as e:
                        msg_e = f"predict error: {e}"
                        if msg_e != last_error_msg:
                            print("loop error:", msg_e)
                            last_error_msg = msg_e
                        await ws.send_text(json.dumps(resp))
                        continue

                    now_s = time.time()

                    # 고정 문구 즉시 출력(쿨다운)
                    if pred in FIXED_UTTERANCES:
                        if (now_s - last_sentence_time) >= COOLDOWN_SEC or last_sentence_text != pred:
                            resp["sentence"] = pred
                            last_sentence_time = now_s
                            last_sentence_text = pred
                        recent_preds.clear()
                        word_buffer.clear()
                        last_confirm_time = 0.0
                        await ws.send_text(json.dumps(resp))
                        continue

                    # 다수결 확정 → 버퍼 push
                    recent_preds.append(pred)
                    if len(recent_preds) >= 3:
                        cnt = Counter(recent_preds)
                        top_word, top_count = cnt.most_common(1)[0]
                        if top_count >= MAJ:
                            if not word_buffer or word_buffer[-1] != top_word:
                                word_buffer.append(top_word)
                                last_confirm_time = now_s
                                sentence, consume = try_make_sentence_from_buffer_by_distance(word_buffer)
                                if sentence:
                                    if (now_s - last_sentence_time) >= COOLDOWN_SEC or last_sentence_text != sentence:
                                        resp["sentence"] = sentence
                                        last_sentence_time = now_s
                                        last_sentence_text = sentence
                                    if consume > 0:
                                        del word_buffer[:consume]
                                    recent_preds.clear()

                    # 타임아웃 시 한 번 더 조립
                    if last_confirm_time:
                        inactive = (now_s - last_confirm_time) > INACTIVITY_SEC
                        hard = (now_s - boot_time) > HARD_RESET_SEC and inactive
                        if inactive or hard:
                            sentence, consume = try_make_sentence_from_buffer_by_distance(word_buffer)
                            if sentence:
                                if (now_s - last_sentence_time) >= COOLDOWN_SEC or last_sentence_text != sentence:
                                    resp["sentence"] = sentence
                                    last_sentence_time = now_s
                                    last_sentence_text = sentence
                                if consume > 0:
                                    del word_buffer[:consume]
                            if not word_buffer:
                                recent_preds.clear()
                                last_confirm_time = 0.0

                await ws.send_text(json.dumps(resp))

        except WebSocketDisconnect:
            print("WebSocket 연결 종료")
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close(code=CloseCode.NORMAL_CLOSURE, reason="client disconnect")
        except Exception as e:
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close(code=CloseCode.INTERNAL_ERROR, reason="server error")
            print("ws fatal:", e)
        finally:
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close(code=CloseCode.NORMAL_CLOSURE, reason="bye")
