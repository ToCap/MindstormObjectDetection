import cv2
import numpy as np

from preprocessing import preprocess_frame

# ==========================
# GEOMETRY MODELS
# ==========================
from geometry.bbox_model import (
    init_model as init_bbox_model,
    detect_bounding_boxes,
    draw_bounding_boxes
)

from geometry.contour_model import (
    init_model as init_contour_model,
    detect_contours,
    draw_contours
)

from geometry.circle_model import (
    init_model as init_circle_model,
    detect_circles,
    draw_circles
)

# ==========================
# GROUND MODELS
# ==========================
from ground_model_standard_geometry import detect_ground_hough
from ground_model_standard_texture import (
    init_ground_model,
    init_ground_model_from_frame,
    detect_ground_mask,
    detect_ground_line,
    draw_ground_mask,
    draw_ground_line
)

# ==========================
# OBJECT DETECTION — TENSORFLOW
# ==========================
from object_detection_tensorflow import (
    detect_objects,
    draw_object_boxes
)


# ==========================
# SETTINGS
# ==========================
USE_TEXTURE_MODEL = True  # True = HSV/texture model, False = Hough geometry


# ==========================
# INITIALIZATION
# ==========================
init_bbox_model()
init_contour_model()
init_circle_model()
init_ground_model()  # initializes structure for texture model


# ==========================
# CAMERA INITIALIZATION
# ==========================
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Error: unable to open camera")


# ==========================
# MAIN LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from camera")
        break

    raw_frame = frame.copy()
    overlay = frame.copy()

    # ==========================
    # PREPROCESSING
    # ==========================
    gray, edges = preprocess_frame(raw_frame)

    # ==========================
    # OBJECT DETECTION
    # ==========================
    if 1:
        boxes, scores, classes = detect_objects(raw_frame)
        draw_object_boxes(overlay, boxes, scores, classes)

    # ==========================
    # GEOMETRY DETECTIONS
    # ==========================
    if 0:
        circles = detect_circles(gray)
        draw_circles(overlay, circles)

    if 0:
        contours = detect_contours(edges)
        draw_contours(overlay, contours)

    if 0:
        boxes = detect_bounding_boxes(edges)
        draw_bounding_boxes(overlay, boxes)

    # ==========================
    # GROUND DETECTION
    # ==========================
    if USE_TEXTURE_MODEL:
        # Learn ground texture on first frame if needed
        if not init_ground_model_from_frame.__globals__['_GROUND_READY']:
            init_ground_model_from_frame(raw_frame)

        ground_mask = detect_ground_mask(raw_frame)
        ground_y = detect_ground_line(ground_mask)
        draw_ground_mask(overlay, ground_mask)
        draw_ground_line(overlay, ground_y)
    else:
        ground_y = detect_ground_hough(edges, overlay)
        # detect_ground_hough draws line directly


    # ==========================
    # DISPLAY
    # ==========================
    cv2.imshow("Frame + Detections", overlay)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()
