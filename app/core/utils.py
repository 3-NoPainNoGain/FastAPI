import asyncio
import base64
import re
from typing import Optional

import cv2
import numpy as np

_B64_DATAURL = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,")

def safe_b64_to_bgr(s: str) -> Optional[np.ndarray]:
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