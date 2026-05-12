import cv2
import numpy as np


def auto_white_balance(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Simple automatic white balance by scaling RGB channels.
    
    Args:
        img: BGR image
        strength: 0.0 = no change, 1.0 = full correction
    
    Returns:
        White-balanced BGR image
    """
    b, g, r = cv2.split(img.astype(np.float32))
    
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    
    # Scale R and B to match G
    kr = avg_g / (avg_r + 1e-5)
    kb = avg_g / (avg_b + 1e-5)
    
    # Apply correction with blend
    kr = 1.0 + (kr - 1.0) * strength
    kb = 1.0 + (kb - 1.0) * strength
    
    r_corrected = np.clip(r * kr, 0, 255)
    b_corrected = np.clip(b * kb, 0, 255)
    
    return cv2.merge([b_corrected, g, r_corrected]).astype(np.uint8)


def multi_scale_sharpen(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Multi-scale unsharp mask for natural-looking sharpening.
    Combines fine detail and medium contrast enhancement.
    
    Args:
        img: BGR image
        strength: 0.0-5.0, sharpening intensity
    
    Returns:
        Sharpened BGR image
    """
    if strength <= 0:
        return img

    # Sharpen in luminance only to avoid color halos.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32) / 255.0

    blurred_small = cv2.GaussianBlur(l_float, (3, 3), 0.6)
    detail_small = l_float - blurred_small

    blurred_medium = cv2.GaussianBlur(l_float, (5, 5), 1.2)
    detail_medium = l_float - blurred_medium

    # Edge-aware boost helps boundaries of objects stand out more clearly.
    lap = cv2.Laplacian(l_float, cv2.CV_32F, ksize=3)
    edge = np.abs(lap)
    edge = edge / (np.max(edge) + 1e-6)
    edge_mask = 0.50 + 0.75 * edge

    # Extra fine-edge pass for stronger perceived sharpness on object boundaries.
    edge_fine = l_float - cv2.GaussianBlur(l_float, (0, 0), 0.9)

    # Keep some sharpening in darker regions so objects remain distinguishable,
    # while still suppressing excessive noise amplification.
    # dark_mask = 0.34 + 0.66 * np.clip((l_float - 0.05) / 0.45, 0.0, 1.0)
    #   0.34: minimum sharpening kept in deep shadows (prevents losing all detail)
    #   0.66: the remaining amount increases as luminance rises
    #   0.05: cutoff threshold for deep shadow region
    #   0.45: transition range from shadow to brighter areas
    dark_mask = 0.34 + 0.66 * np.clip((l_float - 0.05) / 0.45, 0.0, 1.0)

    #   detail_small * 0.78: main contribution, emphasizes fine details
    #   detail_medium * 0.22: adds medium-scale details
    #   edge_fine * 0.28: boosts very fine edges, increases object boundary sharpness
    #   (strength * 0.92): slightly reduces overall amplitude to avoid excessive sharpening at high strength
    #   dark_mask: reduces sharpening in deep shadows to avoid noise amplification
    #   edge_mask: increases sharpening in strong edge regions
    combined = l_float + (
        detail_small * 0.78 +
        detail_medium * 0.22 +
        edge_fine * 0.28
    ) * (strength * 0.92) * dark_mask * edge_mask

    l_out = np.clip(combined * 255.0, 0, 255).astype(np.uint8)

    out_lab = cv2.merge([l_out, a, b])
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def recover_shadow_details(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Lift shadow detail while preserving highlights.

    Args:
        img: BGR image
        strength: 0.0-2.0, recovery intensity

    Returns:
        Detail-recovered BGR image
    """
    if strength <= 0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32) / 255.0

    # Shadow mask: strong in dark areas, minimal in highlights.
    #   0.65: upper bound for shadow region
    #   0.55: transition range from shadow to midtones
    shadow_mask = np.clip((0.65 - l_float) / 0.55, 0.0, 1.0)

    # Lift shadows with a soft gamma-like curve.
    #   0.78: gamma exponent, gently lifts shadows without blowing out midtones
    lifted = np.power(np.clip(l_float, 0.0, 1.0), 0.78)
    #   (0.55 * strength): controls how much shadow lifting is applied
    l_mix = l_float + (lifted - l_float) * shadow_mask * (0.55 * strength)

    # Add local contrast in shadows to reveal more objects.
    #   clipLimit=1.0 + 0.8 * strength: increases local contrast adaptively
    #   tileGridSize=(8,8): size of local regions for CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=1.0 + 0.8 * strength,
        tileGridSize=(8, 8),
    )
    l_local = clahe.apply((l_mix * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
    #   (0.35 * strength): blend ratio for local contrast
    l_mix = l_mix * (1.0 - 0.35 * strength) + l_local * (0.35 * strength)

    # Protect highlights from over-brightening.
    #   0.72: highlight threshold
    #   0.28: transition range for highlight protection
    hi_mask = np.clip((l_float - 0.72) / 0.28, 0.0, 1.0)
    l_mix = l_mix * (1.0 - hi_mask) + l_float * hi_mask

    l_out = np.clip(l_mix * 255.0, 0, 255).astype(np.uint8)
    out_lab = cv2.merge([l_out, a, b])
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def hdr_tone_mapping(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Simple HDR tone mapping to recover detail in shadows and highlights.
    Uses local contrast enhancement similar to HDR.
    
    Args:
        img: BGR image
        strength: 0.0-2.0, intensity of tone mapping
    
    Returns:
        Tone-mapped BGR image
    """
    if strength <= 0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32) / 255.0

    # Adaptive global exposure normalization for both dark and bright scenes.
    #   target_mid = 0.48: desired midtone after normalization
    #   gamma: computed to map mean luminance to target_mid
    #   gamma is clipped to [0.70, 1.35] to avoid extreme contrast changes
    mean_l = float(np.mean(l_float))
    target_mid = 0.48
    gamma = np.log(target_mid + 1e-6) / np.log(mean_l + 1e-6)
    gamma = float(np.clip(gamma, 0.70, 1.35))
    l_global = np.power(np.clip(l_float, 0.0, 1.0), gamma)

    # Local contrast enhancement with conservative blend.
    #   clipLimit=1.0 + 1.2 * strength: increases local contrast adaptively
    #   (0.45 * strength): blend ratio for local contrast
    clahe = cv2.createCLAHE(
        clipLimit=1.0 + 1.2 * strength,
        tileGridSize=(8, 8),
    )
    l_local = clahe.apply((l_global * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
    l_mix = l_global * (1.0 - 0.45 * strength) + l_local * (0.45 * strength)

    # Highlight roll-off keeps bright sky/clouds from clipping.
    #   0.7: highlight threshold
    #   0.3: transition range for highlight roll-off
    #   (0.15 * strength): roll-off strength
    highlight = np.clip((l_mix - 0.7) / 0.3, 0.0, 1.0)
    l_mix = l_mix - highlight * (0.15 * strength)
    l_mix = np.clip(l_mix, 0.0, 1.0)

    l_out = (l_mix * 255.0).astype(np.uint8)
    out_lab = cv2.merge([l_out, a, b])
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def denoise_bilateral_adaptive(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Bilateral + fastNlMeans denoise for best quality.
    Preserves edges while removing noise.
    
    Args:
        img: BGR image
        strength: 0.0-2.0, denoising intensity
    
    Returns:
        Denoised BGR image
    """
    if strength <= 0:
        return img
    
    # Bilateral filter: edge-preserving blur
    #   d: filter diameter, increases with strength (min 3)
    #   sigma_color, sigma_space: color and spatial sigmas, scale with strength
    d = max(3, int(5 * strength))
    sigma_color = 50 * strength
    sigma_space = 50 * strength
    img_bilateral = cv2.bilateralFilter(
        img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )
    
    # Optional: add fastNlMeansDenoising for additional smoothing
    #   h_param: denoising strength, increases with strength (min 3)
    #   templateWindowSize=7, searchWindowSize=21: patch sizes for denoising
    if strength > 0.5:
        h_param = max(3, int(10 * strength))
        img_bilateral = cv2.fastNlMeansDenoisingColored(
            img_bilateral, None, h=h_param, hForColorComponents=h_param, templateWindowSize=7,
            searchWindowSize=21
        )
    
    return img_bilateral


def apply_clarity(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Enhance local contrast and micro-details for punchy, crisp appearance.
    Uses high-pass filter effect for clarity without introducing halos.

    Args:
        img: BGR image
        strength: 0.0-2.0, clarity intensity

    Returns:
        Clarity-enhanced BGR image
    """
    if strength <= 0:
        return img

    # Work in LAB for luminance-only clarity
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32) / 255.0

    # High-pass filter: subtract Gaussian blur from original
    #   (31,31), 8.0: large kernel and sigma for strong blur, isolates fine details
    blurred = cv2.GaussianBlur(l_float, (31, 31), 8.0)
    high_pass = l_float - blurred

    # Create edge mask to apply clarity selectively
    #   ksize=3: small Laplacian kernel for edge detection
    #   edge_mask = 0.3 + 0.7 * edge_mask: ensures minimum effect in flat regions, boosts at edges
    lap = cv2.Laplacian(l_float, cv2.CV_32F, ksize=3)
    edge_mask = np.abs(lap) / (np.max(np.abs(lap)) + 1e-6)
    edge_mask = 0.3 + 0.7 * edge_mask

    # Apply clarity: add high-pass with sigmoid-like falloff
    #   0.8: scales clarity effect to avoid halos
    clarity = l_float + high_pass * strength * edge_mask * 0.8
    clarity = np.clip(clarity, 0.0, 1.0)

    l_out = np.clip(clarity * 255.0, 0, 255).astype(np.uint8)
    out_lab = cv2.merge([l_out, a, b])
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def apply_tone_curve(img: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """
    Apply S-curve to increase contrast and tonal separation.
    Creates punchier, more defined image tones.

    Args:
        img: BGR image
        strength: 0.0-1.0, curve intensity

    Returns:
        Tone-curved BGR image
    """
    if strength <= 0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32) / 255.0

    # S-curve: steeper in mid-tones, flattened at extremes
    #   x + strength * 0.5 * (x - x^3): increases contrast in midtones, gentle at shadows/highlights
    #   0.5: scales the S-curve effect for natural look
    x = l_float
    s_curve = x + strength * 0.5 * (x - x**3)
    s_curve = np.clip(s_curve, 0.0, 1.0)

    l_out = np.clip(s_curve * 255.0, 0, 255).astype(np.uint8)
    out_lab = cv2.merge([l_out, a, b])
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def enhance_object_edges(img: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """
    Edge enhancement tuned for object detection pre-processing.
    Emphasizes structural boundaries while limiting flat-area noise.

    Args:
        img: BGR image
        strength: 0.0-2.0, edge emphasis intensity

    Returns:
        Edge-enhanced BGR image
    """
    if strength <= 0:
        return img

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32) / 255.0

    # Light denoise on luminance avoids boosting sensor noise as false edges.
    #   (3,3), 0.8: small blur to suppress noise
    l_smooth = cv2.GaussianBlur(l_float, (3, 3), 0.8)

    # Sobel gradients for edge detection
    #   ksize=3: small kernel for fine edges
    gx = cv2.Sobel(l_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_norm = grad / (np.max(grad) + 1e-6)

    # Keep only confident edge regions to reduce background artifacts.
    #   0.045: edge threshold, 0.30: transition range
    #   (5,5), 0.8: smooths edge confidence mask
    edge_conf = np.clip((grad_norm - 0.045) / 0.30, 0.0, 1.0)
    edge_conf = cv2.GaussianBlur(edge_conf, (5, 5), 0.8)

    # Unsharp detail focused on detected edge regions.
    #   (0,0), 1.1: strong blur for unsharp masking
    #   1.1 * strength: scales detail boost
    blur = cv2.GaussianBlur(l_smooth, (0, 0), 1.1)
    detail = l_smooth - blur
    l_boost = l_float + detail * (1.1 * strength) * edge_conf

    # Add a small gradient-driven lift at edges for better object boundaries.
    #   0.08 * strength: small boost at edges
    l_boost = l_boost + grad_norm * (0.08 * strength) * edge_conf
    l_boost = np.clip(l_boost, 0.0, 1.0)

    l_out = np.clip(l_boost * 255.0, 0, 255).astype(np.uint8)
    out_lab = cv2.merge([l_out, a, b])
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


