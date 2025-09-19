import base64, re
import numpy as np
import cv2
from typing import Optional

_B64_DATAURL = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,")

def safe_b64_to_bgr(s: str) -> Optional[np.ndarray]:
    try:
        if not s:
            return None
        s = _B64_DATAURL.sub("", s).replace(" ", "+").replace("%2B", "+")
        pad = len(s) % 4
        if pad: s += "=" * (4 - pad)
        buf = base64.b64decode(s, validate=False)
        if not buf: return None
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.size == 0: return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
