
import cv2


# ==========================
# INITIALIZATION
# ==========================
def init_contour_model_standard():
    """
    Initialize the contour model.
    Stateless placeholder.
    """
    print("[ContourModel] init_model called (placeholder)")


def calibrate_contour_model_standard(frame=None):
    """
    Optional initialization from a frame.
    """
    print("[ContourModel] calibrate_model called (placeholder)")


# ==========================
# DETECTION — CONTOURS
# ==========================
def detect_contour_standard(edges):
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


# ==========================
# DRAWING
# ==========================
def draw_contour_standard(
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
