import cv2
import numpy as np




def draw_hough_lines(
    edges,
    overlay,
    threshold=100,
    min_line_length=0,
    max_line_gap=0
):
    """
    Detects straight line segments using the probabilistic Hough transform.

    This function draws ALL detected lines and does not apply
    any semantic filtering (debug / visualization only).

    Parameters:
    - edges: binary edge image (Canny output)
    - overlay: image used for drawing results
    - threshold: accumulator threshold for line detection
    - min_line_length: minimum accepted line length
    - max_line_gap: maximum allowed gap between line segments

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

    # Draw detected lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    return lines
