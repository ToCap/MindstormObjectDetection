import cv2
import numpy as np


# ==========================
# GLOBAL MODEL STATE
# ==========================
_GROUND_MEAN = None
_GROUND_STD = None
_GROUND_ROI_RATIO = 0.6
_initialized = False


# ==========================
# INITIALIZATION - STRUCTURE
# ==========================
def initialize():
    """
    Initializes the ground model structure and parameters.
    Texture learning must be done separately from a frame.
    """
    global _GROUND_ROI_RATIO, _initialized

    _GROUND_ROI_RATIO = 0.6
    _initialized = False

    print("[GroundModel] Structure initialized")



# ==========================
# INITIALIZATION — PERCEPTION
# ==========================
def calibrate(ctx):
    """
    Learns ground color statistics from a single frame.
    """
    global _GROUND_MEAN, _GROUND_STD, _initialized

    print("[GroundModel] Learning ground texture...")

    frame_bgr = ctx.frame
    h, w = frame_bgr.shape[:2]

    roi = frame_bgr[int(h * _GROUND_ROI_RATIO):h, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Sample bottom part of ROI to avoid obstacles
    sample = hsv[int(roi.shape[0] * 0.7):, :]
    pixels = sample.reshape(-1, 3)

    _GROUND_MEAN = np.mean(pixels, axis=0)
    _GROUND_STD = np.std(pixels, axis=0) + 1e-6

    _initialized = True

    print("[GroundModel] Ground model ready")

def is_calibrated():
    return _initialized

# ==========================
# DETECTION — GROUND MASK
# ==========================
def detect(ctx, std_factor=2.0):
    """
    Detects ground pixels based on learned HSV statistics.
    """
    frame_bgr = ctx.frame

    h, w = frame_bgr.shape[:2]
    mask_full = np.zeros((h, w), dtype=np.uint8)

    roi = frame_bgr[int(h * _GROUND_ROI_RATIO):h, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = _GROUND_MEAN - std_factor * _GROUND_STD
    upper = _GROUND_MEAN + std_factor * _GROUND_STD

    lower = np.clip(lower, 0, 255).astype(np.uint8)
    upper = np.clip(upper, 0, 255).astype(np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask_full[int(h * _GROUND_ROI_RATIO):h, :] = mask
    return mask_full


# ==========================
# DRAWING
# ==========================
def draw(
    overlay,
    mask,
    color=(0, 255, 0),
    alpha=0.4
):
    """
    Draws the ground mask on the overlay.
    """
    color_layer = np.zeros_like(overlay)
    color_layer[mask > 0] = color

    cv2.addWeighted(
        color_layer,
        alpha,
        overlay,
        1 - alpha,
        0,
        overlay
    )
