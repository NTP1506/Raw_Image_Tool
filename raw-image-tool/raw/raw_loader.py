import cv2
import numpy as np

from shared.constants import SUPPORTED_RAW
from shared.helpers import get_file_extension, load_raw_bgr

def load_raw_image(path: str) -> np.ndarray:
    ext = get_file_extension(path)

    if ext in SUPPORTED_RAW:
        try:
            return load_raw_bgr(path)
        except ValueError as raw_exc:
            # Some DNG files are not decodable by rawpy/LibRaw but may still
            # contain a preview image that OpenCV can open.
            img = cv2.imread(path)
            if img is not None:
                return img
            raise ValueError(
                f"{raw_exc}\nKhong mo duoc file nay duoi dang RAW hoac anh thuong."
            ) from raw_exc

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot open image: {path}")
    return img
