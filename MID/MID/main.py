import cv2


from preprocessing import build_context

# ==========================
# GEOMETRY MODELS
# ==========================
import bbox_standard_contour as bbox
import circle_standard_hough as circle
import contour_standard_canny as contour

# ==========================
# GROUND MODELS
# ==========================
import ground_standard_geometry as ground_geom
import ground_standard_texture as ground_tex

# ==========================
# OBJECT DETECTION — TENSORFLOW
# ==========================
#import object_tensorflow_ssd as object_tf

# ==========================
# SETTINGS
# ==========================
USE_TEXTURE_MODEL = True  # True = HSV texture, False = Hough geometry

# ==========================
# INITIALIZATION
# ==========================
bbox.initialize()
contour.initialize()
circle.initialize()
ground_tex.initialize()
#object_tf.initialize()

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
    ctx = build_context(raw_frame)

    # ==========================
    # OBJECT DETECTION
    # ==========================
    #if 1:
    #    objects = object_tf.detect(ctx)
    #    object_tf.draw(overlay, objects)

    # ==========================
    # GEOMETRY DETECTIONS
    # ==========================
    if 0:
        circles = circle.detect(ctx)
        circle.draw(overlay, circles)

    if 0:
        contours = contour.detect(ctx)
        contour.draw(overlay, contours)

    if 0:
        boxes = bbox.detect(ctx)
        bbox.draw(overlay, boxes)

    # ==========================
    # GROUND DETECTION
    # ==========================
    if 1:

        # calibration automatique au premier frame
        if not ground_tex.is_calibrated():
            ground_tex.calibrate(ctx)

        ground_mask = ground_tex.detect(ctx)
        ground_y = ground_tex.detect_line(ground_mask)

        ground_tex.draw(overlay, ground_mask)
        ##ground_tex.draw_line(overlay, ground_y)

    else:
        # modèle géométrique
        ground_y = ground_geom.detect(ctx)

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
