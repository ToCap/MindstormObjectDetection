
import cv2


# ==========================
# INITIALIZATION
# ==========================
def initialize():
    """
    Initialize the contour model.
    Stateless placeholder.
    """
    print("[ContourModel] initialize called (placeholder)")


def calibrate(ctx=None):
    """
    Optional initialization from a frame.
    """
    print("[ContourModel] calibrate called (placeholder)")


# ==========================
# DETECTION — CONTOURS
# ==========================
def detect(ctx):
    edges = ctx.edges

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


# ==========================
# DRAWING
# ==========================
def draw(
    overlay,
    contours,
    color=(0, 0, 255),
    thickness=2
):
    for cnt in contours:
        cv2.drawContours(
            overlay,
            [cnt],
            -1,
            color,
            thickness
        )
