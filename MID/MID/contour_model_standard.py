
import cv2


# ==========================
# INITIALIZATION
# ==========================
def init_model():
    """
    Initialize the contour model.
    Stateless placeholder.
    """
    print("[ContourModel] init_model called (placeholder)")


def init_model_from_frame(frame=None):
    """
    Optional initialization from a frame.
    """
    print("[ContourModel] init_model_from_frame called (placeholder)")


# ==========================
# DETECTION — CONTOURS
# ==========================
def detect_contours(edges):
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


# ==========================
# DRAWING
# ==========================
def draw_contours(
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
