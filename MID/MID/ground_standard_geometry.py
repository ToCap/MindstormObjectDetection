import cv2
import numpy as np


# ==========================
# INITIALIZATION
# ==========================
def initialize():
    """
    Initializes the geometric ground model.
    Currently stores only configuration parameters.
    """
    global _GROUND_ROI_RATIO
    _GROUND_ROI_RATIO = 0.6

    print("[GroundGeometry] Model initialized")


def calibrate(ctx=None):
    """
    Placeholder for interface consistency.
    """
    pass


# ==========================
# DETECTION — Hough candidates
# ==========================
def _detect_hough_lines(
    edges,
    threshold,
    min_dx_ratio,
    max_line_gap=50
):
    h_img, w_img = edges.shape

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=int(w_img * min_dx_ratio),
        maxLineGap=max_line_gap
    )

    return [] if lines is None else lines


# ==========================
# DETECTION — Ground line
# ==========================
def detect(
    ctx,
    roi_ratio=0.6,
    dy_thresh=15,
    min_dx_ratio=0.3,
    strategy="best"  # "best" | "lowest"
):
    """
    Detects the ground as a horizontal line in the lower part of the image.

    Strategies:
    - "best"   : longest + flattest line
    - "lowest" : lowest valid horizontal line
    """
    edges = ctx.edges
    h_img, w_img = edges.shape

    # Bottom region of interest (ROI)
    roi_y = int(h_img * roi_ratio)
    edges_roi = edges[roi_y:h_img, :]

    lines = _detect_hough_lines(
        edges_roi,
        threshold=80 if strategy == "best" else 60,
        min_dx_ratio=min_dx_ratio
    )

    best_y = None
    best_score = -np.inf

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Reproject coordinates to full image
        y1 += roi_y
        y2 += roi_y

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Nearly horizontal and wide enough
        if dy > dy_thresh or dx < w_img * min_dx_ratio:
            continue

        y_avg = (y1 + y2) // 2

        if strategy == "lowest":
            score = y_avg
        else:  # "best"
            score = dx - 5 * dy

        if score > best_score:
            best_score = score
            best_y = y_avg

    return best_y


# ==========================
# DRAWING
# ==========================
def draw(
    overlay,
    y,
    color=(255, 0, 0),
    thickness=4,
    label="GROUND"
):
    """
    Draws the detected ground line.
    """
    if y is None:
        return

    h, w = overlay.shape[:2]

    cv2.line(
        overlay,
        (0, y),
        (w, y),
        color,
        thickness
    )

    cv2.putText(
        overlay,
        label,
        (20, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )
