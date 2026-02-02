import cv2
import numpy as np


def detect_geometry_bounding_boxes(
    edges,
    min_area=200,
    max_area=200000,
    min_aspect=0.1,
    max_aspect=10,
    scale=1.3
):
    """
    Detects bounding boxes from contours using geometric heuristics.

    Returns:
        List of boxes: [(x1, y1, x2, y2), ...]
    """
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h_img, w_img = edges.shape
    boxes = []

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

        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(w_img, cx + new_w // 2)
        y2 = min(h_img, cy + new_h // 2)

        boxes.append((x1, y1, x2, y2))

    return boxes


def draw_geometry_bounding_boxes(
    overlay,
    boxes,
    color=(0, 255, 0),
    thickness=2
):
    """
    Draws bounding boxes on the overlay.
    """
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )


def detect_geometry_contours(edges):
    """
    Detects raw contours (debug / analysis).
    """
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def draw_geometry_contours(
    overlay,
    contours,
    color=(0, 0, 255),
    thickness=2
):
    """
    Draws contours on the overlay.
    """
    for cnt in contours:
        cv2.drawContours(overlay, [cnt], -1, color, thickness)


def detect_geometry_circle_boxes(
    gray,
    dp=1,
    min_dist=20,
    param1=100,
    param2=30,
    min_radius=0,
    max_radius=0
):
    """
    Detects circles using the Hough Circle Transform.

    Returns:
        List of circles: [(x, y, r), ...]
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

    return np.round(circles[0, :]).astype("int")


def draw_geometry_circles(
    overlay,
    circles,
    color=(255, 0, 0)
):
    """
    Draws detected circles.
    """
    for (x, y, r) in circles:
        cv2.circle(overlay, (x, y), r, color, 2)
        cv2.circle(overlay, (x, y), 2, color, 3)

