import unicodedata
from typing import Any, List, Tuple 

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