import cv2

import parameters

_TAG = "BBoxContour"

# ==========================
# ADJUSTABLE PARAMETERS (run time)
# ==========================
MIN_AREA_PX2 = 200          # smallest contour area accepted
MAX_AREA_PX2 = 200000       # largest contour area accepted
MIN_ASPECT_RATIO = 0.1      # smallest width / height accepted
MAX_ASPECT_RATIO = 10.0     # largest width / height accepted
SCALE_RATIO = 1.0           # enlargement of each box around its centre (1.0 = none)

BBOX_E_BAD_PARAMETER = 0x01

_PARAMETERS = (
    parameters.Parameter("MIN_AREA_PX2", float, 0.0, 1.0e9, "px2",
                         "Smallest contour area accepted."),
    parameters.Parameter("MAX_AREA_PX2", float, 0.0, 1.0e9, "px2",
                         "Largest contour area accepted."),
    parameters.Parameter("MIN_ASPECT_RATIO", float, 0.001, 1000.0, "ratio",
                         "Smallest width / height accepted."),
    parameters.Parameter("MAX_ASPECT_RATIO", float, 0.001, 1000.0, "ratio",
                         "Largest width / height accepted."),
    parameters.Parameter("SCALE_RATIO", float, 0.1, 10.0, "ratio",
                         "Enlargement of each box around its centre; 1.0 means none."),
)


# ==========================
# INITIALIZATION
# ==========================
def initialize():
    """
    Minimum implementation to satisfy the common detection module interface.
    """
    print("[BBoxContour] initialize called")


def calibrate(ctx=None):
    """
    Minimum implementation to satisfy the common detection module interface.
    """
    print("[BBoxContour] calibrate called")


# ==========================
# PARAMETERS
# ==========================
def get_parameters():
    return parameters.describe(globals(), _PARAMETERS)


def get_parameter(name):
    return parameters.get_value(globals(), _PARAMETERS, name, _TAG,
                                "BBOX_E_BAD_PARAMETER", BBOX_E_BAD_PARAMETER)


def set_parameter(name, value):
    parameters.set_value(globals(), _PARAMETERS, name, value, _TAG,
                         "BBOX_E_BAD_PARAMETER", BBOX_E_BAD_PARAMETER)


# ==========================
# DETECTION - BOUNDING BOXES
# ==========================
def detect(ctx):
    edges = ctx.edges

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h_img, w_img = edges.shape
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX2 or area > MAX_AREA_PX2:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            continue

        cx, cy = x + w // 2, y + h // 2
        new_w, new_h = int(w * SCALE_RATIO), int(h * SCALE_RATIO)

        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w_img, cx + new_w // 2)
        y2 = min(h_img, cy + new_h // 2)

        boxes.append((x1, y1, x2, y2))

    return boxes


# ==========================
# DRAWING
# ==========================
def draw(
    overlay,
    boxes,
    color=(0, 255, 0),
    thickness=2
):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )
