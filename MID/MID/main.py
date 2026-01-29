import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from bbox_detection import (
    draw_bounding_boxes,
    draw_circle_boxes,
    draw_raw_contours
)
from ground_detection import detect_ground, detect_ground_hough
from preprocessing import preprocess_frame


from object_detection_tensorflow import (
    detect_objects,
    draw_object_boxes
)


# open internal camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error : unable to open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error : Unable to read frame from camera")
        break

    raw_frame = frame.copy()
    overlay = frame.copy()

    # Preprocessing (gray, edges)
    gray, edges = preprocess_frame(raw_frame)

    # =========================
    # CHOICES: ENABLE / DISABLE
    # =========================*

    if 0:
        boxes, scores, classes = detect_objects(frame)
        draw_object_boxes(overlay, boxes, scores, classes)

    if 0:
        # Draw circles detected by Hough
        draw_circle_boxes(gray, overlay)

    if 0:
        # Draw raw contours
        draw_raw_contours(edges, overlay)

    if 0:
        # Draw filtered bounding boxes
        draw_bounding_boxes(edges, overlay)

    if 0:
        # Detect ground using method 1
        detect_ground(edges, overlay)

    if 1:
        # Detect ground using Hough (best line)
        detect_ground_hough(edges, overlay)

    # Display
    cv2.imshow("Frame + Edges", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
