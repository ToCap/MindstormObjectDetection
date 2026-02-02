import cv2
import numpy as np

from preprocessing import preprocess_frame

from bbox_detection import (
    detect_geometry_bounding_boxes,
    draw_geometry_bounding_boxes,
    detect_geometry_contours,
    draw_geometry_contours,
    detect_geometry_circle_boxes,
    draw_geometry_circles
)

from ground_detection import (
    detect_ground,
    detect_ground_hough
)

from object_detection_tensorflow import (
    detect_objects,
    draw_object_boxes
)

# ==========================
# Camera initialization
# ==========================
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Error: unable to open camera")

# ==========================
# Main loop
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from camera")
        break

    raw_frame = frame.copy()
    overlay = frame.copy()

    # ==========================
    # Preprocessing
    # ==========================
    gray, edges = preprocess_frame(raw_frame)

    # ==========================
    # ENABLE / DISABLE MODULES
    # ==========================

    if 1:
        # ==========================
        # TensorFlow object detection
        # ==========================
        boxes, scores, classes = detect_objects(raw_frame)
        draw_object_boxes(overlay, boxes, scores, classes)

    if 0:
        # ==========================
        # Geometry: circles (Hough)
        # ==========================
        circles = detect_geometry_circle_boxes(gray)
        # draw is already included OR:
        # draw_geometry_circle_boxes(overlay, circles)

    if 0:
        # ==========================
        # Geometry: raw contours
        # ==========================
        contours = detect_geometry_contours(edges)
        draw_geometry_contours(overlay, contours)

    if 0:
        # ==========================
        # Geometry: bounding boxes
        # ==========================
        boxes = detect_geometry_bounding_boxes(edges)
        draw_geometry_bounding_boxes(overlay, boxes)

    if 0:
        # ==========================
        # Ground detection (simple)
        # ==========================
        detect_ground(edges, overlay)

    if 1:
        # ==========================
        # Ground detection (Hough)
        # ==========================
        detect_ground_hough(edges, overlay)

    # ==========================
    # Display
    # ==========================
    cv2.imshow("Frame + Detections", overlay)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# Cleanup
# ==========================
cap.release()
cv2.destroyAllWindows()
