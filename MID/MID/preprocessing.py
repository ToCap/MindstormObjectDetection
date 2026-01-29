import cv2
import numpy as np


def preprocess_frame(frame):
    """
    Convertit une frame BGR en :
    - gray
    - edges (Canny + morphologie)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Canny
    edges = cv2.Canny(gray, 30, 100)

    # Nettoyage morphologique
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)

    return gray, edges

