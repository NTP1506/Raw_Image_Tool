import cv2
import numpy as np

def apply_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Gamma must be greater than 0")

    inv_gamma = 1.0 / gamma
    # Build lookup table for fast pixel-wise correction
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)

    return cv2.LUT(img, table)
