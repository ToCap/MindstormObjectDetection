import cv2
import numpy as np

import parameters

_TAG = "GroundTexture"
GROUND_E_BAD_PARAMETER = 0x01


# ==========================
# GLOBAL MODEL STATE
# ==========================
_GROUND_MODELS = []   # list of (mean, std) in HSV, one per ground texture
_initialized = False

# ==========================
# ADJUSTABLE PARAMETERS (run time)
# ==========================
# Height of the camera above the floor, in meters
CAMERA_HEIGHT_M = 0.12

# Learning area: bottom of the image (fraction of the image height) and
# centred strip (fraction of the image width)
SAMPLE_BOTTOM_RATIO = 0.12
SAMPLE_WIDTH_RATIO = 1.0

# The sample is split in TEXTURE_COUNT_N clusters (ground textures, e.g.
# floor + rug); a cluster smaller than this share of the sample is ignored
TEXTURE_COUNT_N = 2
MIN_CLUSTER_RATIO = 0.1

# Two clusters whose centres are closer than this (in units of standard
# deviation, windows overlapping) are one texture: they are merged
MERGE_DISTANCE_SIGMA = 2.0

# Width of the colour window, in standard deviations
STD_FACTOR = 2.0

# V (brightness) tolerance multiplier: lighting varies more than colour
V_TOLERANCE_FACTOR = 1.5

# Cleaning of the mask
OPEN_KERNEL_SIZE_PX = 9      # removes thin objects (legs) and isolated spots
EDGE_DILATE_SIZE_PX = 3      # thickness of the contours cut out of the mask
EDGE_BLUR_SIZE_PX = 9        # smoothing before Canny (odd; ignores fine textures)
EDGE_MIN_LENGTH_PX = 25      # only contours at least this long
MAX_HOLE_AREA_PX2 = 400      # enclosed holes up to this area are filled

_PARAMETERS = (
    parameters.Parameter("CAMERA_HEIGHT_M", float, 0.0, 2.0, "m",
                         "Height of the camera above the floor; sets the start of the analysed area."),
    parameters.Parameter("SAMPLE_BOTTOM_RATIO", float, 0.01, 1.0, "ratio",
                         "Height of the learning area at the bottom of the image."),
    parameters.Parameter("SAMPLE_WIDTH_RATIO", float, 0.05, 1.0, "ratio",
                         "Width of the centred learning area."),
    parameters.Parameter("TEXTURE_COUNT_N", int, 1, 2, "count",
                         "Number of ground textures learned."),
    parameters.Parameter("MIN_CLUSTER_RATIO", float, 0.0, 0.5, "ratio",
                         "Smallest share of the sample for a texture to be kept."),
    parameters.Parameter("MERGE_DISTANCE_SIGMA", float, 0.0, 10.0, "sigma",
                         "Distance under which two textures are merged."),
    parameters.Parameter("STD_FACTOR", float, 0.5, 6.0, "factor",
                         "Width of the colour window, in standard deviations."),
    parameters.Parameter("V_TOLERANCE_FACTOR", float, 1.0, 5.0, "factor",
                         "Extra tolerance on the brightness channel."),
    parameters.Parameter("OPEN_KERNEL_SIZE_PX", int, 1, 51, "px",
                         "Size of the opening that removes thin objects."),
    parameters.Parameter("EDGE_DILATE_SIZE_PX", int, 1, 31, "px",
                         "Thickness of the colour contours cut out of the mask."),
    parameters.Parameter("EDGE_BLUR_SIZE_PX", int, 1, 31, "px",
                         "Smoothing before the contour detection.", True),
    parameters.Parameter("EDGE_MIN_LENGTH_PX", int, 1, 200, "px",
                         "Shortest contour that is kept."),
    parameters.Parameter("MAX_HOLE_AREA_PX2", int, 0, 100000, "px2",
                         "Largest enclosed hole that is filled."),
)

# Analysed area: from a fraction of the image height down to the bottom. The
# higher the camera, the lower the floor ends in the image, so the fraction
# grows with the camera height. Points (height in m, fraction), measured on
# captures; linear in between, constant outside
_ROI_BY_HEIGHT = ((0.12, 0.40), (0.60, 0.60))

# Feature weights (H, S, V) used for the clustering only
_CLUSTER_WEIGHTS = np.array([2.0, 1.0, 0.5], dtype=np.float32)

# Minimum standard deviation per channel (H, S, V): avoids a window that is
# too narrow on a uniform floor
_MIN_STD = np.array([5.0, 15.0, 15.0])

# Canny thresholds (low, high) on the Lab channels L, a, b: colour contours
# are cut out of the mask even when the grey levels are close
_EDGE_THRESHOLDS = ((10, 30), (4, 12), (4, 12))

# Colours (BGR) used by draw() for texture 1, 2, ...
_DRAW_COLORS = [(0, 255, 0), (255, 255, 0)]


# ==========================
# INITIALIZATION - STRUCTURE
# ==========================
def initialize():
    """
    Initializes the ground model structure.
    Texture learning must be done separately from a frame.
    """
    print("[GroundModel] Structure initialized (camera at %.2f m, analysed "
          "from %.0f %% of the height)"
          % (CAMERA_HEIGHT_M, 100.0 * _roi_ratio()))


def _roi_ratio():
    heights, ratios = zip(*_ROI_BY_HEIGHT)
    return float(np.interp(CAMERA_HEIGHT_M, heights, ratios))


# ==========================
# PARAMETERS
# ==========================
def get_parameters():
    return parameters.describe(globals(), _PARAMETERS)


def get_parameter(name):
    return parameters.get_value(globals(), _PARAMETERS, name, _TAG,
                                "GROUND_E_BAD_PARAMETER", GROUND_E_BAD_PARAMETER)


def set_parameter(name, value):
    parameters.set_value(globals(), _PARAMETERS, name, value, _TAG,
                         "GROUND_E_BAD_PARAMETER", GROUND_E_BAD_PARAMETER)


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

    # Sample the bottom of the image to avoid obstacles
    # (centred area, see SAMPLE_WIDTH_RATIO)
    hsv = cv2.cvtColor(frame_bgr[int(h * (1.0 - SAMPLE_BOTTOM_RATIO)):, :],
                       cv2.COLOR_BGR2HSV)
    x0 = int(w * (1.0 - SAMPLE_WIDTH_RATIO) / 2.0)
    x1 = w - x0
    sample = hsv[:, x0:x1]
    pixels = sample.reshape(-1, 3)

    # split the sample in textures (k-means on weighted HSV)
    data = np.ascontiguousarray(pixels, dtype=np.float32)
    models = []

    if len(data) >= TEXTURE_COUNT_N:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(
            data * _CLUSTER_WEIGHTS, TEXTURE_COUNT_N, None, criteria, 3,
            cv2.KMEANS_PP_CENTERS
        )
        labels = labels.ravel()

        for k in range(TEXTURE_COUNT_N):
            cluster = data[labels == k]
            if len(cluster) >= MIN_CLUSTER_RATIO * len(data):
                models.append((
                    np.mean(cluster, axis=0),
                    np.maximum(np.std(cluster, axis=0), _MIN_STD),
                ))

    # one texture seen as two clusters: merge them
    if len(models) == 2:
        (m1, s1), (m2, s2) = models
        scale = np.maximum(s1, s2) * np.array([1.0, 1.0, V_TOLERANCE_FACTOR])
        if np.max(np.abs(m1 - m2) / scale) < MERGE_DISTANCE_SIGMA:
            models = []

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
def detect(ctx):
    """
    Detects ground pixels based on learned HSV statistics.
    Returns a uint8 image of the frame size: 0 = not ground,
    1, 2, ... = index (from 1) of the ground texture of the pixel.
    """
    frame_bgr = ctx.frame

    h, w = frame_bgr.shape[:2]
    mask_full = np.zeros((h, w), dtype=np.uint8)
    y0 = int(h * _roi_ratio())

    roi = frame_bgr[y0:h, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_f = hsv.astype(np.float32)

    # one window per texture; a pixel takes the closest texture
    factor = STD_FACTOR * np.array([1.0, 1.0, V_TOLERANCE_FACTOR])
    best = np.full(hsv.shape[:2], np.inf, dtype=np.float32)
    texture = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for k, (mean, std) in enumerate(_GROUND_MODELS):
        dist = np.max(np.abs(hsv_f - mean) / (factor * std), axis=2)
        closer = (dist <= 1.0) & (dist < best)
        best[closer] = dist[closer]
        texture[closer] = k + 1

    mask = np.where(texture > 0, 255, 0).astype(np.uint8)

    # fill small holes, then cut the colour contours (base of the furniture)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # contours of the Lab channels, smoothed: fine textures (rug) vanish, long
    # contours (furniture, walls) stay
    lab = cv2.cvtColor(
        cv2.GaussianBlur(roi, (EDGE_BLUR_SIZE_PX,) * 2, 0), cv2.COLOR_BGR2Lab
    )
    edges = np.zeros(mask.shape, dtype=np.uint8)
    for channel, (low, high) in zip(cv2.split(lab), _EDGE_THRESHOLDS):
        edges |= cv2.Canny(channel, low, high)
    long_edges = cv2.morphologyEx(
        edges, cv2.MORPH_OPEN, np.ones((1, EDGE_MIN_LENGTH_PX), np.uint8)
    ) | cv2.morphologyEx(
        edges, cv2.MORPH_OPEN, np.ones((EDGE_MIN_LENGTH_PX, 1), np.uint8)
    )
    long_edges = cv2.dilate(
        long_edges, np.ones((EDGE_DILATE_SIZE_PX,) * 2, np.uint8)
    )
    mask[long_edges > 0] = 0

    # remove thin objects and isolated spots
    kernel = np.ones((OPEN_KERNEL_SIZE_PX,) * 2, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # region growing from the bottom: keep only the regions connected to the
    # bottom of the image
    _, labels = cv2.connectedComponents(mask)
    bottom = np.unique(labels[-1, :])
    keep = np.isin(labels, bottom[bottom > 0])

    # fill the small holes enclosed in the ground (dark spots)
    holes = np.where(keep, 0, 255).astype(np.uint8)
    n, hole_labels, stats, _ = cv2.connectedComponentsWithStats(
        holes, connectivity=4
    )
    x, y, w_h, h_h, area = (stats[:, i] for i in range(5))
    touches = (x == 0) | (y == 0) | (x + w_h == keep.shape[1]) \
        | (y + h_h == keep.shape[0])
    small = ~touches & (area <= MAX_HOLE_AREA_PX2)
    small[0] = False
    keep |= small[hole_labels]

    # pixels added by the closing or by the filling take the main texture
    if keep.any():
        main = int(np.bincount(texture[keep], minlength=3)[1:].argmax()) + 1
        texture[keep & (texture == 0)] = main

    mask_full[y0:h, :] = np.where(keep, texture, 0)
    return mask_full


# ==========================
# DRAWING
# ==========================
def draw(
    overlay,
    mask,
    colors=None,
    alpha=0.4
):
    """
    Draws the ground mask on the overlay, one colour per texture.
    """
    colors = colors or _DRAW_COLORS
    color_layer = np.zeros_like(overlay)
    for k in range(1, int(mask.max()) + 1):
        color_layer[mask == k] = colors[(k - 1) % len(colors)]

    cv2.addWeighted(
        color_layer,
        alpha,
        overlay,
        1 - alpha,
        0,
        overlay
    )
