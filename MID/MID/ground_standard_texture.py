import cv2
import numpy as np


# ==========================
# GLOBAL MODEL STATE
# ==========================
_GROUND_MODELS = []   # list of (mean, std) in HSV, one per ground texture
_GROUND_ROI_RATIO = 0.6
_initialized = False

# Learning area: bottom strip, centred (fraction of the image width)
_SAMPLE_WIDTH_RATIO = 1.0

# The sample is split in 2 clusters (2 ground textures, e.g. floor + rug);
# a cluster smaller than this share of the sample is ignored
_N_TEXTURES = 2
_MIN_CLUSTER_RATIO = 0.1

# Feature weights (H, S, V) used for the clustering only
_CLUSTER_WEIGHTS = np.array([2.0, 1.0, 0.5], dtype=np.float32)

# Minimum standard deviation per channel (H, S, V): avoids a window that is
# too narrow on a uniform floor
_MIN_STD = np.array([5.0, 20.0, 30.0])

# V (brightness) tolerance multiplier: lighting varies more than colour
_V_TOLERANCE_FACTOR = 2.0


# ==========================
# INITIALIZATION - STRUCTURE
# ==========================
def initialize():
    """
    Initializes the ground model structure and parameters.
    Texture learning must be done separately from a frame.
    """
    global _GROUND_ROI_RATIO

    _GROUND_ROI_RATIO = 0.6

    print("[GroundModel] Structure initialized")



# ==========================
# INITIALIZATION � PERCEPTION
# ==========================
def calibrate(ctx):
    """
    Learns ground color statistics from a single frame.
    """
    global _GROUND_MODELS

    print("[GroundModel] Learning ground texture...")

    frame_bgr = ctx.frame
    h, w = frame_bgr.shape[:2]

    roi = frame_bgr[int(h * _GROUND_ROI_RATIO):h, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Sample bottom part of ROI to avoid obstacles
    # (centred area, see _SAMPLE_WIDTH_RATIO)
    x0 = int(w * (1.0 - _SAMPLE_WIDTH_RATIO) / 2.0)
    x1 = w - x0
    sample = hsv[int(roi.shape[0] * 0.7):, x0:x1]
    pixels = sample.reshape(-1, 3)

    # split the sample in textures (k-means on weighted HSV)
    data = np.ascontiguousarray(pixels, dtype=np.float32)
    models = []

    if len(data) >= _N_TEXTURES:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(
            data * _CLUSTER_WEIGHTS, _N_TEXTURES, None, criteria, 3,
            cv2.KMEANS_PP_CENTERS
        )
        labels = labels.ravel()

        for k in range(_N_TEXTURES):
            cluster = data[labels == k]
            if len(cluster) >= _MIN_CLUSTER_RATIO * len(data):
                models.append((
                    np.mean(cluster, axis=0),
                    np.maximum(np.std(cluster, axis=0), _MIN_STD),
                ))

    if not models:
        models.append((
            np.mean(data, axis=0),
            np.maximum(np.std(data, axis=0), _MIN_STD),
        ))

    _GROUND_MODELS = models

    print("[GroundModel] Ground model ready (%d texture(s))" % len(models))

def is_calibrated():
    return _initialized

# ==========================
# DETECTION � GROUND MASK
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

    factor = std_factor * np.array([1.0, 1.0, _V_TOLERANCE_FACTOR])
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for mean, std in _GROUND_MODELS:
        lower = np.clip(mean - factor * std, 0, 255).astype(np.uint8)
        upper = np.clip(mean + factor * std, 0, 255).astype(np.uint8)
        mask |= cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # keep only the regions connected to the bottom of the image
    _, labels = cv2.connectedComponents(mask)
    bottom = np.unique(labels[-1, :])
    mask = np.where(np.isin(labels, bottom[bottom > 0]), 255, 0).astype(np.uint8)

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
