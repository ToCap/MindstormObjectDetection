import cv2
import numpy as np


def draw_bounding_boxes(
    edges,
    overlay,
    min_area=200,
    max_area=200000,
    min_aspect=0.1,
    max_aspect=10,
    scale=1.3
):
    """
    Detects contours and draws filtered bounding boxes.
    """
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h_img, w_img = overlay.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = w / float(h)
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue

        cx, cy = x + w // 2, y + h // 2
        new_w, new_h = int(w * scale), int(h * scale)

        x2 = max(0, cx - new_w // 2)
        y2 = max(0, cy - new_h // 2)
        x3 = min(w_img, cx + new_w // 2)
        y3 = min(h_img, cy + new_h // 2)

        cv2.rectangle(
            overlay,
            (x2, y2),
            (x3, y3),
            (0, 255, 0),
            2
        )


def draw_raw_contours(edges, overlay):
    """
    Draws raw contours for debugging.
    """
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 2)


def draw_circle_boxes(
    gray,
    overlay,
    dp=1,
    min_dist=20,
    param1=100,
    param2=30,
    min_radius=0,
    max_radius=0
):
    """
    Detects circles using the Hough Circle Transform and draws them.
    """
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

    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        cv2.circle(overlay, (x, y), r, (255, 0, 0), 2)
        cv2.circle(overlay, (x, y), 2, (255, 0, 0), 3)

    return circles
