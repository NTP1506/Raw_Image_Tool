import cv2
import numpy as np

def apply_resize(img: np.ndarray, width: int = None, height: int = None, scale: float = None) -> np.ndarray:
    h, w = img.shape[:2]

    if scale is not None:
        new_w = int(w * scale)
        new_h = int(h * scale)
    elif width is not None:
        ratio = width / w
        new_w = width
        new_h = int(h * ratio)
    elif height is not None:
        ratio = height / h
        new_h = height
        new_w = int(w * ratio)
    else:
        return img

    interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LANCZOS4
    return cv2.resize(img, (new_w, new_h), interpolation=interp)
