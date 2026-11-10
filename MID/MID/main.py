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
# MAP MODELS
# ==========================
import map_standard_fusion as map_fusion

# ==========================
# OBJECT DETECTION — TENSORFLOW
# ==========================
#import object_tensorflow_ssd as object_tf
#import ground_tensorflow_mobilenet as ground_tf

# ==========================
# SETTINGS
# ==========================
# Views of the same camera frame, processed one after the other and
# displayed as a grid: one list per row, one list of detectors per view.
# An empty list is a placeholder view (black).
VIEWS = [
    [[ground_tex], [ground_geom]],   # top    : HSV texture | Hough geometry
    [[map_fusion], []],              # bottom : 2D occupancy grid | AI results
]

# ==========================
# INITIALIZATION
# ==========================
# each module is initialized once, even if it is used in several views
for module in dict.fromkeys(m for row in VIEWS for view in row for m in view):
    module.initialize()

# ==========================
# VIEWS
# ==========================
# modules whose calibration was already attempted (a cancelled
# calibration is not requested again at every frame)
_calibration_tried = set()


def render_view(detectors, ctx):
    """
    Applies a list of detectors on a copy of the frame
    and returns the annotated image.
    """
    overlay = ctx.frame.copy()
    if not detectors:
        overlay[:] = 0

    for module in detectors:
        # auto-calibration (modules without is_calibrated are skipped)
        if (hasattr(module, "is_calibrated") and not module.is_calibrated()
                and module not in _calibration_tried):
            _calibration_tried.add(module)
            module.calibrate(ctx)

        module.draw(overlay, module.detect(ctx))

    label = ", ".join(m.__name__ for m in detectors) or "AI results (todo)"
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
    rows = [
        cv2.hconcat([render_view(detectors, ctx) for detectors in row])
        for row in VIEWS
    ]

    # ==========================
    # DISPLAY
    # ==========================
    cv2.imshow("Frame + Detections", cv2.vconcat(rows))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
cv2.destroyAllWindows()
