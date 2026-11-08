import cv2


from preprocessing import preprocess_frame

# ==========================
# GEOMETRY MODELS
# ==========================
from bbox_standard_contour import (
    init_bbox_standard_contour,
    detect_bbox_standard_contour,
    draw_bbox_standard_contour
)

from circle_standard_hough import (
    init_circle_standard_hough,
    detect_circle_standard_hough,
    draw_circle_standard_hough
)

from contour_standard_canny import (
    init_contour_standard_canny,
    detect_contour_standard_canny,
    draw_contour_standard_canny
)

# ==========================
# GROUND MODELS
# ==========================
from ground_standard_geometry import (
    init_ground_standard_geometry,
    detect_ground_standard_geometry,
    draw_ground_standard_geometry
)

from ground_standard_texture import (
    init_ground_standard_texture,
    calibrate_ground_standard_texture,
    is_ground_standard_texture_ready,
    detect_ground_mask_standard_texture,
    detect_ground_line_standard_texture,
    draw_ground_mask_standard_texture,
    draw_ground_line_standard_texture
)

# ==========================
# OBJECT DETECTION — TENSORFLOW
# ==========================
#from object_tensorflow_ssd import (
#    init_object_tensorflow_ssd,
#    calibrate_object_tensorflow_ssd,
#    detect_object_tensorflow_ssd,
#    draw_object_tensorflow_ssd
#)

# ==========================
# SETTINGS
# ==========================
USE_TEXTURE_MODEL = True  # True = HSV texture, False = Hough geometry

# ==========================
# INITIALIZATION
# ==========================
init_bbox_standard_contour()
init_contour_standard_canny()
init_circle_standard_hough()
init_ground_standard_texture()

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
    #if 1:
    #    boxes, scores, classes = detect_object_tensorflow_ssd(raw_frame)
    #    draw_object_tensorflow_ssd(overlay, boxes, scores, classes)

    # ==========================
    # GEOMETRY DETECTIONS
    # ==========================
    if 0:
        circles = detect_circle_standard_hough(gray)
        draw_circle_standard_hough(overlay, circles)

    if 0:
        contours = detect_contour_standard_canny(edges)
        draw_contour_standard_canny(overlay, contours)

    if 0:
        boxes = detect_bbox_standard_contour(edges)
        draw_bbox_standard_contour(overlay, boxes)

    # ==========================
    # GROUND DETECTION
    # ==========================
    if 1:

        # calibration automatique au premier frame
        if not is_ground_standard_texture_ready():
            calibrate_ground_standard_texture(raw_frame)

        ground_mask = detect_ground_mask_standard_texture(raw_frame)
        ground_y = detect_ground_line_standard_texture(ground_mask)

        draw_ground_mask_standard_texture(overlay, ground_mask)
        ##draw_ground_line_standard_texture(overlay, ground_y)

    else:
        # modèle géométrique
        ground_y = detect_ground_standard_geometry(edges, overlay)

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
