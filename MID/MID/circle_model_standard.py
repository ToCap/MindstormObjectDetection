import cv2
import numpy as np


# ==========================
# INITIALIZATION
# ==========================
def init_circle_model_standard():
    """
    Initialize the circle detection model.
    Stateless placeholder.
    """
    print("[CircleModel] init_model called (placeholder)")


def calibrate_circle_model_standard(frame=None):
    """
    Optional initialization from a frame.
    """
    print("[CircleModel] calibrate_model called (placeholder)")


# ==========================
# DETECTION — HOUGH CIRCLES
# ==========================
def detect_circle_standard(
    gray,
    dp=1,
    min_dist=20,
    param1=100,
    param2=30,
    min_radius=0,
    max_radius=0
):
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return []

    return np.round(circles[0, :]).astype("int")


# ==========================
# DRAWING
# ==========================
def draw_circle_standard(
    overlay,
    circles,
    color=(255, 0, 0)
):
    for (x, y, r) in circles:
        cv2.circle(overlay, (x, y), r, color, 2)
        cv2.circle(overlay, (x, y), 2, color, 3)
