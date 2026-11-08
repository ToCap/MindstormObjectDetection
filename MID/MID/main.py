import cv2


from preprocessing import build_context

# ==========================
# GEOMETRY MODELS
# ==========================
import bbox_standard_contour as bbox
import circle_standard_hough as circle
import contour_standard_canny as contour
import lines_standard_hough as lines

# ==========================
# GROUND MODELS
# ==========================
import ground_standard_geometry as ground_geom
import ground_standard_texture as ground_tex

# ==========================
# OBJECT DETECTION — TENSORFLOW
# ==========================
#import object_tensorflow_ssd as object_tf
#import ground_tensorflow_mobilenet as ground_tf

# ==========================
# SETTINGS
# ==========================
# Two views of the same camera frame, processed one after the other.
# Each view applies its own list of detectors (modules imported above).
VIEWS = [
    [ground_tex],   # left view : HSV texture
    [ground_geom],  # right view: Hough geometry
]

# ==========================
# INITIALIZATION
# ==========================
# each module is initialized once, even if it is used in several views
for module in dict.fromkeys(m for view in VIEWS for m in view):
    module.initialize()

# ==========================
# VIEWS
# ==========================
def render_view(detectors, ctx):
    """
    Applies a list of detectors on a copy of the frame
    and returns the annotated image.
    """
    overlay = ctx.frame.copy()

    for module in detectors:
        # auto-calibration (modules without is_calibrated are skipped)
        if hasattr(module, "is_calibrated") and not module.is_calibrated():
            module.calibrate(ctx)

        module.draw(overlay, module.detect(ctx))

    label = ", ".join(m.__name__ for m in detectors)
    cv2.putText(
        overlay, label, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    return overlay

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

    # ==========================
    # PREPROCESSING
    # ==========================
    ctx = build_context(frame)

    # ==========================
    # VIEWS (same frame, one method per view)
    # ==========================
    views = [render_view(detectors, ctx) for detectors in VIEWS]

    # ==========================
    # DISPLAY
    # ==========================
    cv2.imshow("Frame + Detections", cv2.hconcat(views))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()
