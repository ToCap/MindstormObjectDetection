import cv2
import numpy as np


# ==========================
# INITIALIZATION
# ==========================
def initialize():
    """
    Initialize the circle detection model.
    Stateless placeholder.
    """
    print("[CircleModel] initialize called (placeholder)")


def calibrate(ctx=None):
    """
    Optional initialization from a frame.
    """
    print("[CircleModel] calibrate called (placeholder)")


# ==========================
# DETECTION — HOUGH CIRCLES
# ==========================
def detect(
    ctx,
    dp=1,
    min_dist=20,
    param1=100,
    param2=30,
    min_radius=0,
    max_radius=0
):
    gray = ctx.gray

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
def draw(
    overlay,
    circles,
    color=(255, 0, 0)
):
    for (x, y, r) in circles:
        cv2.circle(overlay, (x, y), r, color, 2)
        cv2.circle(overlay, (x, y), 2, color, 3)
