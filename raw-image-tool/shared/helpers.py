import os
import cv2
import numpy as np
import rawpy
from PIL import Image, ImageTk


def bgr_to_photoimage(bgr: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


def ensure_outputs_dir() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(base, exist_ok=True)
    return base


def get_file_extension(path: str) -> str:
    # Return lowercase file extension.
    return os.path.splitext(path)[-1].lower()


def load_raw_bgr(path: str) -> np.ndarray:
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=False,
                output_bps=8,
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except rawpy.LibRawFileUnsupportedError as exc:
        raise ValueError(
            "File RAW/DNG not supported. "
        ) from exc
    except rawpy.LibRawError as exc:
        raise ValueError(f"Cannot read RAW/DNG: {exc}") from exc
