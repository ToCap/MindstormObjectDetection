import cv2
import numpy as np


def detect_ground(
    edges,
    overlay,
    roi_ratio=0.6,
    dy_thresh=15,
    min_dx_ratio=0.3
):
    """
    Detects the ground as the best horizontal line
    in the lower part of the image.

    Selection criteria:
    - search only in a bottom ROI
    - nearly horizontal lines
    - score based on line length and flatness
    """
    h_img, w_img = edges.shape

    # Bottom region of interest (ROI)
    roi_y = int(h_img * roi_ratio)
    edges_roi = edges[roi_y:h_img, :]

    # Probabilistic Hough transform
    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(w_img * min_dx_ratio),
        maxLineGap=50
    )

    best_line = None
    best_score = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Reproject coordinates to full image
            y1 += roi_y
            y2 += roi_y

            dy = abs(y1 - y2)
            dx = abs(x2 - x1)

            # Nearly horizontal and long enough
            if dy < dy_thresh and dx > 50:
                score = dx - dy * 5  # favor long, flat lines

                if score > best_score:
                    best_score = score
                    best_line = (x1, y1, x2, y2)

    if best_line is not None:
        _, y1, _, y2 = best_line
        y = (y1 + y2) // 2

        # Draw ground line
        cv2.line(
            overlay,
            (0, y),
            (w_img, y),
            (255, 0, 0),
            4
        )

        cv2.putText(
            overlay,
            "GROUND",
            (20, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        return y  # detected ground position (optional)

    return None


def detect_ground_hough(
    edges,
    overlay,
    roi_ratio=0.6,
    dy_thresh=15,
    min_dx_ratio=0.3
):
    """
    Ground detection using a simpler Hough strategy:
    - bottom ROI
    - nearly horizontal lines
    - select the lowest valid line
    """
    h_img, w_img = edges.shape

    # Bottom region of interest (ROI)
    roi_y = int(h_img * roi_ratio)
    edges_roi = edges[roi_y:h_img, :]

    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=int(w_img * min_dx_ratio),
        maxLineGap=50
    )

    ground_y = None
    best_y = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Reproject coordinates to full image
            y1 += roi_y
            y2 += roi_y

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            # Nearly horizontal and wide enough
            if dy < dy_thresh and dx > w_img * min_dx_ratio:
                y_avg = (y1 + y2) // 2

                # Keep the lowest line (ground assumption)
                if y_avg > best_y:
                    best_y = y_avg
                    ground_y = y_avg

    if ground_y is not None:
        cv2.line(
            overlay,
            (0, ground_y),
            (w_img, ground_y),
            (255, 0, 0),
            4
        )

        cv2.putText(
            overlay,
            "GROUND (Hough)",
            (20, ground_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

    return ground_y
