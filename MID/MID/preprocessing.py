from dataclasses import dataclass

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


@dataclass
class FrameContext:
    """
    Images shared by all detectors for one frame.
    Built once per frame so that every detect(ctx) has the same signature.
    """
    frame: np.ndarray  # raw BGR frame
    gray: np.ndarray   # grayscale
    edges: np.ndarray  # Canny + morphology


def build_context(frame):
    """
    Runs the preprocessing once and returns the shared FrameContext.
    """
    gray, edges = preprocess_frame(frame)
    return FrameContext(frame=frame, gray=gray, edges=edges)

