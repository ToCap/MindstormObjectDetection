import cv2
import numpy as np


# ==========================
# INITIALIZATION
# ==========================
def init_geometry_model():
    """
    Initializes the geometry model.
    Currently empty but kept for architectural consistency.
    """
    print("[GeometryModel] Initialized")


# ==========================
# DETECTION
# ==========================
def detect_lines_hough(
    edges,
    threshold=100,
    min_line_length=0,
    max_line_gap=0
):
    """
    Detects straight line segments using the probabilistic Hough transform.

    This function performs NO semantic filtering.
    Intended for debug / visualization purposes only.

    Parameters:
    - edges: binary edge image (Canny output)

    Returns:
    - list of detected lines (raw Hough output)
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        return []

    return lines


# ==========================
# DRAWING
# ==========================
def draw_lines_hough(
    overlay,
    lines,
    color=(0, 255, 0),
    thickness=2
):
    """
    Draws detected Hough lines on the overlay.
    """
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )
