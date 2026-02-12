import cv2
import numpy as np


# ==========================
# GLOBAL MODEL STATE
# ==========================
_GROUND_MEAN = None
_GROUND_STD = None
_GROUND_ROI_RATIO = 0.6


# ==========================
# INITIALIZATION — STRUCTURE
# ==========================
def init_ground_model_standard_texture():
    """
    Initializes the ground model structure and parameters.
    Texture learning must be done separately from a frame.
    """
    global _GROUND_ROI_RATIO

    _GROUND_ROI_RATIO = 0.6

    print("[GroundModel] Structure initialized")



# ==========================
# INITIALIZATION — PERCEPTION
# ==========================
def calibrate_ground_model_texture(frame_bgr):
    """
    Learns ground color statistics from a single frame.
    """
    global _GROUND_MEAN, _GROUND_STD

    print("[GroundModel] Learning ground texture...")

    h, w = frame_bgr.shape[:2]

    roi = frame_bgr[int(h * _GROUND_ROI_RATIO):h, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Sample bottom part of ROI to avoid obstacles
    sample = hsv[int(roi.shape[0] * 0.7):, :]
    pixels = sample.reshape(-1, 3)

    _GROUND_MEAN = np.mean(pixels, axis=0)
    _GROUND_STD = np.std(pixels, axis=0) + 1e-6

    print("[GroundModel] Ground model ready")


# ==========================
# DETECTION — GROUND MASK
# ==========================
def detect_ground_mask_standard_texture(frame_bgr, std_factor=2.0):
    """
    Detects ground pixels based on learned HSV statistics.
    """
    if not is_ground_model_initialized():
        raise RuntimeError(
            "Ground model not initialized from frame. "
            "Call init_ground_model_from_frame() first."
        )

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
# GEOMETRY — GROUND LINE
# ==========================
def detect_ground_line_standard_texture(mask):
    """
    Extracts the ground line as the median of the lowest
    detected ground pixel per column.
    """
    ys = []

    for x in range(mask.shape[1]):
        col = np.where(mask[:, x] > 0)[0]
        if len(col) > 0:
            ys.append(col[-1])

    if not ys:
        return None

    return int(np.median(ys))


# ==========================
# DRAWING
# ==========================
def draw_ground_mask_standard_texture(
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


def detect_ground_line_standard_texture(
    overlay,
    y,
    color=(0, 255, 0),
    thickness=3
):
    """
    Draws the detected ground line.
    """
    if y is None:
        return

    h, w = overlay.shape[:2]
    cv2.line(overlay, (0, y), (w, y), color, thickness)
