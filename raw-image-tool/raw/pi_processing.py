import cv2
import numpy as np

from shared.constants import DEFAULT_PI_JPEG_QUALITY, PI_GAMMA, PI_WHITE_BALANCE_SCALE


def simulate_libcamera_processing(
    raw_image: np.ndarray,
    apply_demosaic: bool = True,
    apply_white_balance: bool = True,
    apply_gamma: bool = True,
) -> np.ndarray:
    """
    Local approximation of Raspberry Pi libcamera image pipeline.
    This does not require Pi hardware and is intended for side-by-side comparison.
    """
    img = raw_image.copy().astype(np.float32) / 255.0

    # 1. Demosaic (if grayscale, treat as Bayer pattern)
    if apply_demosaic and len(img.shape) == 2:
        # Simulate Bayer demosaic using OpenCV's demosaicing
        img_uint8 = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img_uint8, cv2.COLOR_BayerBG2BGR).astype(np.float32) / 255.0

    elif apply_demosaic and len(img.shape) == 3 and img.shape[2] == 3:
        # Already color, just ensure it's float
        pass

    # 2. White balance (gray-world with clamped gains)
    if apply_white_balance:
        avg_r = np.mean(img[:, :, 2])
        avg_g = np.mean(img[:, :, 1])
        avg_b = np.mean(img[:, :, 0])

        if avg_r > 0:
            gain_r = float(np.clip((avg_g / avg_r) * PI_WHITE_BALANCE_SCALE, 0.75, 1.45))
            img[:, :, 2] = img[:, :, 2] * gain_r
        if avg_b > 0:
            gain_b = float(np.clip((avg_g / avg_b) * PI_WHITE_BALANCE_SCALE, 0.75, 1.45))
            img[:, :, 0] = img[:, :, 0] * gain_b

        img = np.clip(img, 0, 1)

    # 3. Mild denoise in luma/chroma style to mimic ISP cleanup.
    img_u8 = (img * 255).astype(np.uint8)
    img_u8 = cv2.bilateralFilter(img_u8, d=5, sigmaColor=25, sigmaSpace=25)

    # 4. Local tone mapping using CLAHE on luminance.
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_u8 = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    img = img_u8.astype(np.float32) / 255.0

    # 5. Gamma correction
    if apply_gamma:
        img = np.power(img, PI_GAMMA)

    # 6. S-curve rolloff for highlights/shadows.
    img = 3 * img**2 - 2 * img**3
    img = np.clip(img, 0, 1)

    return (img * 255).astype(np.uint8)


def simulate_pi_raw_processing(
    raw_image: np.ndarray,
    apply_demosaic: bool = True,
    apply_white_balance: bool = True,
    apply_gamma: bool = True,
) -> np.ndarray:
    """Backward-compatible alias for legacy call sites."""
    return simulate_libcamera_processing(
        raw_image,
        apply_demosaic=apply_demosaic,
        apply_white_balance=apply_white_balance,
        apply_gamma=apply_gamma,
    )


def simulate_pi_jpeg_quality(image: np.ndarray, quality: int = DEFAULT_PI_JPEG_QUALITY) -> np.ndarray:
    # Encode and decode to simulate JPEG compression
    _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded
